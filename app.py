import AutoPylot
from flask import Flask, render_template, request, jsonify, send_file, Response
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import google.generativeai as genai
import json
import os
import io
import base64
import matplotlib.pyplot as plt
from datetime import datetime
import zipfile
from dotenv import load_dotenv
import logging  # Import the logging module

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG,  # Set the desired logging level
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # Create a logger instance

# Load API key from environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

# Global variables to store model and data
current_model = None
current_data = None
current_target = None
scaler = StandardScaler()
label_encoder = LabelEncoder()
training_config = None
training_in_progress = False  # Track if training is already running
stop_training_flag = False  # Track if user wants to stop training
pause_training_flag = False  # Track if user wants to pause training

@app.route('/')
def index():
    logger.debug("Rendering index.html")
    return render_template('index.html', datasetInfo=None, layers=[])

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    logger.debug("Entering upload_dataset route")
    if 'file' not in request.files:
        logger.warning("No file uploaded in upload_dataset")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    target_column = request.form.get('target_column')
    dataset_type = request.form.get('dataset_type', 'classification')  # Get user-selected dataset type
    
    logger.debug(f"Dataset type selected by user: {dataset_type}")

    try:
        # Read dataset
        logger.debug(f"Attempting to read CSV file: {file.filename}")
        df = pd.read_csv(file)
        logger.debug(f"CSV file read successfully. Shape: {df.shape}")

        # Store data and target
        global current_data, current_target
        if (target_column in df.columns):
            current_target = df[target_column]
            current_data = df.drop(columns=[target_column])
            logger.debug(f"Target column found: {target_column}.  current_data shape: {current_data.shape}")
        else:
            logger.error(f"Target column not found: {target_column}")
            return jsonify({'error': 'Target column not found'}), 400

        # Use user-selected dataset type instead of auto-detecting
        n_classes = None
        if dataset_type == 'classification':
            n_classes = len(current_target.unique())
            logger.debug(f"Classification dataset with {n_classes} classes")
        
        # Generate model architecture suggestion using Gemini
        dataset_info = {
            'n_samples': len(df),
            'n_features': len(current_data.columns),
            'target_type': dataset_type,  # Use user-selected type
            'n_classes': n_classes
        }
        logger.debug(f"Dataset info: {dataset_info}")

        prompt = f"""
        Given a dataset with following properties:
        - Number of samples: {dataset_info['n_samples']}
        - Number of features: {dataset_info['n_features']}
        - Task type: {dataset_info['target_type']}
        - Number of classes: {dataset_info['n_classes']}"""
        
        prompt += """
        Suggest an optimal neural network architecture for this task.  Provide the response as a *pure*, valid JSON string with *no* surrounding text or explanations. Do start with "{" and do not use newline codes or tabs. The JSON should have a "layers" field, an "optimizer" field, a "learning_rate" field, a "loss" field, a "metrics" field, a "batch_size" field, and an "epochs" field.
        """
        logger.debug(f"Gemini prompt: {prompt}")

        try:
            response = model.generate_content(prompt)
            result = response.text.strip()
            result = result.replace("\n", "").replace("\t", "")
            result = result.replace("```json\n", "").replace("```", "").replace("json{", "{").replace("}", "}")
            suggested_architecture = json.loads(result)
            logger.debug(f"Gemini suggested architecture: {suggested_architecture}")
        except Exception as e:
            logger.exception("Error calling Gemini API.  Falling back to default architecture.")
            # Fallback to default architecture if Gemini API fails
            if dataset_type == 'classification':
                output_units = n_classes if n_classes > 2 else 1
                output_activation = 'softmax' if n_classes > 2 else 'sigmoid'
                suggested_architecture = {
                    'layers': [
                        {'type': 'Dense', 'units': 64, 'activation': 'relu'},
                        {'type': 'Dropout', 'rate': 0.2},
                        {'type': 'Dense', 'units': 32, 'activation': 'relu'},
                        {'type': 'Dense', 'units': output_units, 'activation': output_activation}
                    ],
                    'optimizer': 'adam',
                    'learning_rate': 0.001,
                    'loss': 'sparse_categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy',
                    'metrics': ['accuracy'],
                    'batch_size': 32,
                    'epochs': 20
                }
            else:  # regression
                suggested_architecture = {
                    'layers': [
                        {'type': 'Dense', 'units': 64, 'activation': 'relu'},
                        {'type': 'Dropout', 'rate': 0.2},
                        {'type': 'Dense', 'units': 32, 'activation': 'relu'},
                        {'type': 'Dense', 'units': 1, 'activation': 'linear'}
                    ],
                    'optimizer': 'adam',
                    'learning_rate': 0.01,  # Higher learning rate for regression
                    'loss': 'mse',
                    'metrics': ['mae'],
                    'batch_size': 32,
                    'epochs': 50  # More epochs for regression
                }

        return jsonify({
            'architecture': suggested_architecture,
            'dataset_info': dataset_info,
            'sample_data': df.head().to_dict()
        })

    except Exception as e:
        logger.exception("Exception in upload_dataset route.")
        return jsonify({'error': str(e)}), 500

@app.route('/build_model', methods=['POST'])
def build_model():
    global current_model
    logger.debug("Entering build_model route")
    try:
        model_config = request.json
        logger.debug(f"Model config received from frontend: {json.dumps(model_config, indent=2)}")
        
        # Check if dataset has been processed
        if current_data is None:
            logger.error("No dataset available for building model")
            return jsonify({
                'error': 'No dataset has been uploaded and analyzed. Please upload a dataset and select a target column first.'
            }), 400
            
        # Validate architecture structure
        if not model_config.get('architecture'):
            logger.error("Missing architecture in model_config")
            return jsonify({'error': 'Missing architecture configuration'}), 400
            
        if not model_config['architecture'].get('layers'):
            logger.error("Missing layers in architecture")
            return jsonify({'error': 'No layers defined in architecture'}), 400
            
        logger.debug(f"Building model with {len(model_config['architecture']['layers'])} layers")
        for i, layer in enumerate(model_config['architecture']['layers']):
            logger.debug(f"Layer {i}: {layer}")

        # Check if input shape is available
        input_features = current_data.shape[1]
        logger.debug(f"Dataset has {input_features} input features")

        # Create Sequential model
        model = tf.keras.Sequential()
        
        # For regression problems, automatically add BatchNormalization after the input
        # This helps with feature scaling and training stability
        dataset_type = model_config.get('dataset_type', 'classification')
        if dataset_type == 'regression' or model_config.get('loss') == 'mse' or model_config.get('loss') == 'mae':
            logger.debug("Adding BatchNormalization layer after input for regression problem stability")
            model.add(tf.keras.layers.BatchNormalization(input_shape=(input_features,)))
        
        # Add layers based on config
        for i, layer in enumerate(model_config['architecture']['layers']):
            # Ensure each layer has a type
            layer_type = layer.get('type', 'Dense')  # Default to Dense if type not specified
            logger.debug(f"Adding layer of type: {layer_type}")
            
            # Get regularization if specified
            regularizer = None
            if layer.get('regularization'):
                reg_type = layer.get('regularization')
                if reg_type == 'l1':
                    regularizer = tf.keras.regularizers.l1(0.01)
                elif reg_type == 'l2':
                    regularizer = tf.keras.regularizers.l2(0.01)
                elif reg_type == 'l1_l2':
                    regularizer = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                logger.debug(f"Using regularization {reg_type} for layer {i}")
            
            # Use better weight initialization - use He for ReLU and Glorot for others
            kernel_initializer = None
            if layer.get('activation') == 'relu':
                kernel_initializer = 'he_uniform'
            else:
                kernel_initializer = 'glorot_uniform'
                
            # Get the units for Dense/LSTM/GRU layers
            units = layer.get('units', 32)
            if layer_type == 'Dropout' or layer_type == 'BatchNormalization':
                # No units needed for these layers, but keep units variable for code clarity
                pass
            elif i == 0 and (layer_type == 'Dense' or layer_type == 'LSTM' or layer_type == 'GRU'):
                # For first layer, ensure enough representation power
                # Use at least 2x the number of input features or the specified units, whichever is larger
                suggested_units = max(input_features * 2, units)
                if suggested_units > units:
                    logger.debug(f"Increasing first layer units from {units} to {suggested_units} for better representation")
                    units = suggested_units
            
            logger.debug(f"Layer {i} configuration: Type={layer_type}, Units={units if layer_type not in ['Dropout', 'BatchNormalization'] else 'N/A'}")
            
            # First layer needs input shape for most layer types
            if i == 0:
                if layer_type == 'Dense':
                    model.add(tf.keras.layers.Dense(
                        units=units, 
                        activation=layer['activation'],
                        kernel_regularizer=regularizer,
                        kernel_initializer=kernel_initializer,
                        input_shape=(input_features,)
                    ))
                elif layer_type == 'Dropout':
                    # Dropout as first layer is unusual but possible
                    model.add(tf.keras.layers.Dropout(
                        rate=layer.get('rate', 0.2),
                        input_shape=(input_features,)
                    ))
                elif layer_type == 'LSTM':
                    # For LSTM, we need to reshape the input to be sequential
                    # Assume each feature is a time step
                    model.add(tf.keras.layers.Reshape(
                        (input_features, 1),
                        input_shape=(input_features,)
                    ))
                    model.add(tf.keras.layers.LSTM(
                        units=units,
                        return_sequences=layer.get('return_sequences', False),
                        kernel_initializer='glorot_uniform'
                    ))
                elif layer_type == 'GRU':
                    # For GRU, we need to reshape the input to be sequential
                    # Assume each feature is a time step
                    model.add(tf.keras.layers.Reshape(
                        (input_features, 1),
                        input_shape=(input_features,)
                    ))
                    model.add(tf.keras.layers.GRU(
                        units=units,
                        return_sequences=layer.get('return_sequences', False),
                        kernel_initializer='glorot_uniform'
                    ))
                elif layer_type == 'BatchNormalization':
                    model.add(tf.keras.layers.BatchNormalization(
                        input_shape=(input_features,)
                    ))
            else:
                # For subsequent layers, no input_shape needed
                if layer_type == 'Dense':
                    model.add(tf.keras.layers.Dense(
                        units=units,
                        activation=layer['activation'],
                        kernel_regularizer=regularizer,
                        kernel_initializer=kernel_initializer
                    ))
                elif layer_type == 'Dropout':
                    model.add(tf.keras.layers.Dropout(
                        rate=layer.get('rate', 0.2)
                    ))
                elif layer_type == 'Flatten':
                    model.add(tf.keras.layers.Flatten())
                elif layer_type == 'LSTM':
                    model.add(tf.keras.layers.LSTM(
                        units=units,
                        return_sequences=layer.get('return_sequences', False),
                        kernel_initializer='glorot_uniform'
                    ))
                elif layer_type == 'GRU':
                    model.add(tf.keras.layers.GRU(
                        units=units,
                        return_sequences=layer.get('return_sequences', False),
                        kernel_initializer='glorot_uniform'
                    ))
                elif layer_type == 'BatchNormalization':
                    model.add(tf.keras.layers.BatchNormalization())

        # Get dataset type from model config - prioritize user selection
        dataset_type = model_config.get('dataset_type', 'classification')
        logger.debug(f"Using dataset type from model config: {dataset_type}")
        
        # Store the dataset type in a global variable for use in training
        global training_config
        if training_config is None:
            training_config = {}
        training_config['dataset_type'] = dataset_type
        logger.debug(f"Updated training_config with dataset_type: {dataset_type}")
            
        # Determine appropriate loss function based on dataset type and output layer
        loss = model_config.get('loss')
        if not loss:
            # Get output layer configuration
            output_layer = model_config['architecture']['layers'][-1]
            output_activation = output_layer.get('activation', 'linear')
            
            if dataset_type == 'classification':
                # Classification task
                logger.debug("Setting classification loss function")
                unique_values = len(current_target.unique())
                if unique_values == 2:  # Binary classification
                    loss = 'binary_crossentropy'
                    logger.debug("Using binary_crossentropy for binary classification")
                else:  # Multi-class classification
                    loss = 'sparse_categorical_crossentropy'
                    logger.debug(f"Using sparse_categorical_crossentropy for multi-class classification with {unique_values} classes")
            else:  # Regression
                logger.debug("Setting regression loss function")
                # Choose appropriate regression loss function based on the activation
                if output_activation == 'linear':
                    loss = 'mse'  # Mean Squared Error is standard for regression
                    logger.debug("Using MSE loss for regression with linear activation")
                elif output_activation == 'sigmoid' or output_activation == 'relu':
                    # For bounded outputs
                    loss = 'mae'  # Mean Absolute Error can be better for bounded regression
                    logger.debug(f"Using MAE loss for regression with {output_activation} activation")
                else:
                    loss = 'mse'  # Default to MSE
                    logger.debug(f"Using default MSE loss for regression with {output_activation} activation")
                
            logger.debug(f"Selected loss function: {loss} for activation: {output_activation}")
        
        # Compile model with optimizer from config or default
        optimizer_name = model_config.get('optimizer', 'adam')
        learning_rate = model_config.get('learning_rate')
        
        # Set appropriate learning rate based on dataset type
        if not learning_rate:
            if dataset_type == 'regression':
                learning_rate = 0.01  # Higher learning rate for regression problems
                logger.debug(f"Using higher default learning rate {learning_rate} for regression problem")
            else:
                learning_rate = 0.001  # Standard learning rate for classification
                logger.debug(f"Using standard default learning rate {learning_rate} for classification problem")
        
        # Use fixed learning rate
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        else:
            optimizer = optimizer_name  # Use string name if not a special case
        
        # Set appropriate metrics based on dataset type
        if dataset_type == 'classification':
            # For classification tasks, use accuracy
            metrics = ['accuracy']
            logger.debug(f"Using classification metrics: {metrics}")
        else:
            # For regression tasks, use MAE as the primary metric
            # Only use one metric to avoid confusion in frontend display
            metrics = ['mae']
            logger.debug(f"Using regression metrics: {metrics}")
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        logger.debug("Model compilation complete.")

        # Store model globally
        global current_model
        current_model = model

        # Get model summary
        stringio = io.StringIO()
        model.summary(print_fn=lambda x: stringio.write(x + '\n'))
        summary_string = stringio.getvalue()
        stringio.close()
        logger.debug(f"Model summary: {summary_string}")

        return jsonify({
            'success': True,
            'model_summary': summary_string,
            'dataset_type': dataset_type  # Return dataset_type to frontend
        })

    except Exception as e:
        logger.exception("Exception in build_model route.")
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        # Check if training is already in progress
        global training_in_progress, training_config
        
        if training_in_progress:
            logger.warning("Training is already in progress, cannot start another session")
            return jsonify({
                'success': False, 
                'error': 'Training is already in progress. Please wait for the current training to complete.'
            }), 400
            
        # Extract training parameters
        data = request.json
        training_config = {
            'optimizer': data.get('optimizer', 'adam'),
            'learning_rate': data.get('learning_rate', 0.001),
            'batch_size': data.get('batch_size', 32),
            'epochs': data.get('epochs', 20),
            'test_size': data.get('test_size', 0.2),  # Default to 20% test data
            'infinite_training': data.get('infinite_training', False),  # New option for infinite training
            'dataset_type': data.get('dataset_type', 'classification')  # Preserve user-selected dataset type
        }
        
        logger.debug(f"Dataset type in training config: {training_config['dataset_type']}")
        
        # Set training flag to true
        training_in_progress = True
        
        logger.debug(f"Received training config: {training_config}")
        
        return jsonify({
            'success': True,
            'message': 'Training started. Connect to training stream for updates.'
        })
        
    except Exception as e:
        logger.exception("Exception in train route.")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/pause_training', methods=['POST'])
def pause_training():
    """Endpoint to pause/resume training"""
    global pause_training_flag
    
    # Toggle the pause flag
    pause_training_flag = not pause_training_flag
    
    action = "paused" if pause_training_flag else "resumed"
    logger.info(f"User requested to {action} training")
    
    return jsonify({
        'success': True, 
        'paused': pause_training_flag,
        'message': f'Training {action}. It will {action} after the current batch completes.'
    })

@app.route('/stop_training', methods=['POST'])
def stop_training():
    """Endpoint to stop training in infinite mode"""
    global stop_training_flag
    
    logger.info("User requested to stop training")
    stop_training_flag = True
    return jsonify({
        'success': True, 
        'message': 'Training stop requested. It will stop after the current epoch completes.'
    })

@app.route('/train_stream')
def train_stream():
    def generate():
        global training_in_progress, stop_training_flag,training_config
        # Reset stop flag at the beginning of training
        stop_training_flag = False
        
        try:
            # Check if we have data to train on
            if current_data is None or current_target is None:
                logger.error("No dataset available for training")
                error_data = {
                    'status': 'error',
                    'message': 'No dataset has been uploaded and analyzed. Please upload a dataset and select a target column first.'
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                # Reset training flag since we're returning early
                training_in_progress = False
                return
                
            if current_model is None:
                logger.error("No model has been built yet")
                error_data = {
                    'status': 'error',
                    'message': 'No model has been built yet. Please build a model first.'
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                # Reset training flag since we're returning early
                training_in_progress = False
                return
            
            # Preprocess data
            X = current_data
            y = current_target
            
            logger.debug(f"Training data shape: {X.shape}, target shape: {y.shape}")

            # Handle categorical features
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = LabelEncoder().fit_transform(X[col])

            # Handle categorical target
            if y.dtype == 'object' or y.dtype.name == 'category':
                y = label_encoder.fit_transform(y)

            # Scale features - improve this to ensure better normalization
            X_scaled = scaler.fit_transform(X)
            
            # Store original target values for regression problems (for prediction conversion later)
            y_min, y_max = None, None
            
            # Get the dataset type from the model configuration or training config
            # This ensures we use the user-selected type instead of auto-detecting
            dataset_type = training_config.get('dataset_type', 'classification')
            
            # Check if this is a regression problem based on user selection
            is_regression = dataset_type == 'regression'
            logger.debug(f"User-selected dataset type: {dataset_type}, is_regression: {is_regression}")
            
            if is_regression:
                logger.debug("Processing regression problem, ensuring target is properly scaled if needed")
                # For regression, normalize the target if it has a large range
                target_range = np.max(y) - np.min(y)
                
                # Always store the original range for later reference
                y_min, y_max = np.min(y), np.max(y)
                
                if target_range > 10:  # If range is large, scale the target
                    logger.debug(f"Target range is large ({target_range}), scaling target values from [{y_min}, {y_max}] to [0,1]")
                    
                    # Scale to [0,1] range for better training stability
                    y = (y - y_min) / (y_max - y_min)
                    logger.debug(f"Target scaled, new range: [{np.min(y)}, {np.max(y)}]")
            else:
                logger.debug("Processing classification problem, no target scaling needed")

            # Get training parameters
            # global training_config
            if not training_config:
                training_config = {
                    'optimizer': 'adam',
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 20,
                    'test_size': 0.2,
                    'infinite_training': False
                }
            
            # Split data with configurable test size
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=training_config['test_size'], random_state=42
            )

            # Create a queue for communication between callback and generator
            from queue import Queue
            update_queue = Queue()

            # Create callback for epoch updates
            class StreamCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    
                    # Log all available keys for debugging
                    logger.debug(f"Epoch {epoch} logs: {logs}")
                    
                    # Get the dataset type from the training config
                    dataset_type = training_config.get('dataset_type', 'classification')
                    is_regression = dataset_type == 'regression'
                    
                    if is_regression:
                        # For regression models, we'll use mae as our main metric
                        acc = logs.get('mae', 0)
                        val_acc = logs.get('val_mae', 0)
                        logger.debug(f"Regression metrics - mae: {acc}, val_mae: {val_acc}")
                    else:
                        # For classification models, use standard accuracy
                        acc = logs.get('accuracy', logs.get('acc', 0))
                        val_acc = logs.get('val_accuracy', logs.get('val_acc', 0))
                        logger.debug(f"Classification metrics - accuracy: {acc}, val_accuracy: {val_acc}")
                    
                    # Create data packet for frontend
                    data = {
                        'status': 'epoch',
                        'epoch': epoch,
                        'loss': float(logs.get('loss', 0)),
                        'val_loss': float(logs.get('val_loss', 0)),
                        'acc': float(acc),
                        'val_acc': float(val_acc),
                        'is_regression': is_regression
                    }
                    
                    logger.debug(f"Sending to frontend: {data}")
                    update_queue.put(data)
                    
                    # Check if we should stop training (for infinite mode)
                    if stop_training_flag:
                        logger.info("Stopping training as requested by user")
                        self.model.stop_training = True
                
                def on_batch_end(self, batch, logs=None):
                    # Check if training should be paused
                    global pause_training_flag
                    while pause_training_flag and not stop_training_flag:
                        # Sleep briefly to avoid CPU spinning
                        import time
                        time.sleep(0.1)
                        
                        # If stop flag is set while paused, exit the pause loop
                        if stop_training_flag:
                            logger.info("Stopping training while paused as requested by user")
                            self.model.stop_training = True
                            break

            # Train model with callbacks
            model_callback = StreamCallback()
            
            # Only use the stream callback - no early stopping as requested
            callbacks = [model_callback]
            
            # No early stopping callback - removed as requested by user
            
            # Add reduce learning rate on plateau
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
            callbacks.append(reduce_lr)
            
            # Start training in a separate thread
            from threading import Thread
            
            def train_model():
                global training_in_progress
                try:
                    # Log the model configuration before training
                    logger.debug(f"Starting model training with metrics: {current_model.metrics_names}")
                    logger.debug(f"Model loss function: {current_model.loss}")
                    
                    # Get the dataset type from the training config
                    dataset_type = training_config.get('dataset_type', 'classification')
                    is_classification = dataset_type == 'classification'
                    logger.debug(f"Task type from config: {'Classification' if is_classification else 'Regression'}")
                    
                    # Check data shapes and types
                    logger.debug(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
                    logger.debug(f"Training data types: X_train={X_train.dtype}, y_train={y_train.dtype}")
                    logger.debug(f"Unique target values: {np.unique(y_train)}")
                    
                    if is_classification:
                        logger.debug(f"Number of classes: {len(np.unique(y_train))}")
                    
                    # Determine epochs based on infinite training setting
                    epochs = 9999 if training_config.get('infinite_training', False) else training_config['epochs']
                    logger.debug(f"Training mode: {'Infinite' if training_config.get('infinite_training', False) else 'Fixed'}, epochs={epochs}")
                    
                    # Start the training process
                    history = current_model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=training_config['batch_size'],
                        validation_data=(X_test, y_test),
                        callbacks=callbacks,
                        verbose=1
                    )
                    
                    # Log training completion
                    logger.debug(f"Training completed. History keys: {history.history.keys()}")
                    
                    # Get the dataset type from the training config
                    dataset_type = training_config.get('dataset_type', 'classification')
                    is_regression = dataset_type == 'regression'
                    logger.debug(f"Task is regression: {is_regression}")
                    
                    # Evaluate model - different handling for regression and classification
                    if is_regression:
                        # For regression, we want MAE (mean absolute error)
                        # Single evaluation call, extract required metrics
                        evaluation = current_model.evaluate(X_test, y_test, verbose=0)
                        
                        # Handle different return types (scalar or array)
                        if isinstance(evaluation, list):
                            # If multiple metrics, first is always loss, then metrics in order
                            test_loss = evaluation[0]
                            # If MAE is available use it
                            if 'mae' in current_model.metrics_names:
                                mae_idx = current_model.metrics_names.index('mae')
                                test_mae = evaluation[mae_idx]
                            elif len(evaluation) > 1:
                                # Use the second metric as our accuracy equivalent
                                test_mae = evaluation[1]
                            else:
                                # If no MAE, just use the loss
                                test_mae = test_loss
                        else:
                            # If just a scalar returned, it's the loss
                            test_loss = evaluation
                            test_mae = evaluation
                        
                        logger.debug(f"Regression evaluation - Loss: {test_loss}, MAE: {test_mae}")
                        
                        # For frontend display, MAE is our "accuracy" equivalent 
                        # Lower is better, but we need to transform it for consistent UI display
                        # Convert it to a range of 0-1 where 1 is best (opposite of error)
                        # First, ensure we don't divide by zero by adding a small epsilon
                        epsilon = 1e-10
                        # Transform MAE to a 0-1 scale where 1 is best (using exponential decay)
                        # This gives a more intuitive representation in the UI
                        normalized_mae = np.exp(-test_mae)
                        
                        completion_data = {
                            'status': 'completed',
                            'test_accuracy': float(normalized_mae),  # Transformed value for UI consistency
                            'test_loss': float(test_loss),
                            'test_mae_raw': float(test_mae),  # Include the raw MAE for reference
                            'is_regression': True
                        }
                    else:
                        # For classification, we want accuracy
                        test_loss, test_acc = current_model.evaluate(X_test, y_test, verbose=0)
                        logger.debug(f"Classification evaluation - Loss: {test_loss}, Accuracy: {test_acc}")
                        
                        completion_data = {
                            'status': 'completed',
                            'test_accuracy': float(test_acc),
                            'test_loss': float(test_loss),
                            'is_regression': False
                        }
                    
                    logger.debug(f"Sending completion data to frontend: {completion_data}")
                    update_queue.put(completion_data)
                except Exception as e:
                    logger.exception("Error during model training")
                    update_queue.put({
                        'status': 'error',
                        'message': str(e)
                    })
                finally:
                    # Reset the training flags
                    global stop_training_flag
                    training_in_progress = False
                    stop_training_flag = False
                    # Mark the end of updates
                    update_queue.put(None)

            # Start training thread
            Thread(target=train_model, daemon=True).start()

            # Generate events from queue
            while True:
                update = update_queue.get()
                if update is None:  # End of updates
                    break
                yield f"data: {json.dumps(update)}\n\n"

        except Exception as e:
            logger.exception("Error in train_stream")
            error_data = {
                'status': 'error',
                'message': str(e)
            }
            # Reset training flag on error
            training_in_progress = False  # We already declared global at the function start
            yield f"data: {json.dumps(error_data)}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/download_model', methods=['GET'])
def download_model():
    logger.debug("Entering download_model route")
    try:
        # Create temporary directory for files
        temp_dir = f'temp_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(temp_dir, exist_ok=True)
        logger.debug(f"Created temporary directory: {temp_dir}")

        # Save model
        model_path = os.path.join(temp_dir, 'model.h5')
        current_model.save(model_path)
        logger.debug(f"Model saved to: {model_path}")

        # Generate and save Python code
        model_code = generate_model_code()
        code_path = os.path.join(temp_dir, 'model_code.py')
        with open(code_path, 'w') as f:
            f.write(model_code)
        logger.debug(f"Model code saved to: {code_path}")

        # Create zip file
        zip_path = f'model_package_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(model_path, 'model.h5')
            zipf.write(code_path, 'model_code.py')
        logger.debug(f"Model package created: {zip_path}")

        # Clean up temporary files
        os.remove(model_path)
        os.remove(code_path)
        os.rmdir(temp_dir)
        logger.debug(f"Temporary files cleaned up.")

        # Send file and then delete it
        response = send_file(
            zip_path,
            as_attachment=True,
            download_name='model_package.zip',
            mimetype='application/zip'
        )
        logger.debug("Sending model package to client.")

        # Schedule file deletion after sending
        @response.call_on_close
        def cleanup():
            try:
                os.remove(zip_path)
                logger.debug(f"Deleted zip file: {zip_path}")
            except Exception as e:
                logger.error(f"Error deleting zip file: {e}")
                pass

        return response

    except Exception as e:
        logger.exception("Exception in download_model route.")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_code', methods=['GET'])
def generate_code():
    global current_model
    logger.debug("Entering generate_code route")
    try:
        # Generate code using the model if it exists, or build one on the fly if not
        if current_model is None:
            logger.info("No model exists yet, attempting to build one on-the-fly for code generation")
            # We need to have processed a dataset at least
            if current_data is None or current_target is None:
                return jsonify({'error': 'No dataset has been analyzed yet. Please upload and analyze a dataset first.'}), 400
                
            # Try to build a temporary model just for code generation
            # This supports the "Generate Code Without Training" feature
            try:
                from tensorflow import keras
                temp_model = keras.Sequential()
                input_shape = current_data.shape[1]
                
                # Use a simple default architecture if we don't have one
                temp_model.add(keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)))
                temp_model.add(keras.layers.Dense(32, activation='relu'))
                
                # Detect if we're doing classification or regression
                if current_target.dtype == 'object' or current_target.dtype.name == 'category':
                    n_classes = len(current_target.unique())
                    if n_classes == 2:  # Binary classification
                        temp_model.add(keras.layers.Dense(1, activation='sigmoid'))
                        loss = 'binary_crossentropy'
                    else:  # Multi-class classification
                        temp_model.add(keras.layers.Dense(n_classes, activation='softmax'))
                        loss = 'sparse_categorical_crossentropy'
                    metrics = ['accuracy']
                else:  # Regression
                    temp_model.add(keras.layers.Dense(1, activation='linear'))
                    loss = 'mse'
                    metrics = ['mae']
                
                # Compile the model
                temp_model.compile(optimizer='adam', loss=loss, metrics=metrics)
                
                # Store it temporarily for code generation
                current_model = temp_model
                logger.info("Temporary model built successfully for code generation")
            except Exception as e:
                logger.exception("Failed to build temporary model")
                return jsonify({'error': f'Failed to generate code: {str(e)}'}), 500
        
        # Generate code using the model
        code = generate_model_code()
        
        # Check for common issues in the code and fix them
        code = validate_and_fix_code(code)
        
        return jsonify({
            'success': True,
            'code': code
        })
    except Exception as e:
        logger.exception("Exception in generate_code route.")
        return jsonify({'error': str(e)}), 500

@app.route('/back_to_model', methods=['POST'])
def back_to_model():
    """
    Handle going back to the model from the generated code view.
    This maintains the current model and training state.
    """
    logger.debug("Entering back_to_model route")
    try:
        # Keep the current model configuration and return info needed for the model view
        response = {
            'success': True,
            'message': 'Returned to model view from code generation'
        }
        
        # Add model_summary if available
        if current_model is not None:
            # Get model summary
            stringio = io.StringIO()
            current_model.summary(print_fn=lambda x: stringio.write(x + '\n'))
            summary_string = stringio.getvalue()
            stringio.close()
            
            response['model_summary'] = summary_string
            
            # Check if model has been trained
            if hasattr(current_model, 'history') and current_model.history is not None:
                response['trained'] = True
            else:
                response['trained'] = False
        
        # Add dataset info if available
        if current_data is not None and current_target is not None:
            dataset_info = {
                'n_samples': len(current_data) + len(current_target),
                'n_features': len(current_data.columns),
                'target_type': 'classification' if len(current_target.unique()) < 20 else 'regression',
                'n_classes': len(current_target.unique()) if len(current_target.unique()) < 20 else None
            }
            response['dataset_info'] = dataset_info
            
        return jsonify(response)
    
    except Exception as e:
        logger.exception("Exception in back_to_model route.")
        return jsonify({'error': str(e)}), 500

@app.route('/go_back', methods=['POST'])
def go_back():
    """
    Handle going back to the model architecture customization stage.
    This resets the training status but keeps the dataset and model architecture.
    """
    global training_in_progress, training_config
    
    logger.debug("Entering go_back route")
    try:
        # Reset training progress if needed
        if training_in_progress:
            training_in_progress = False
            logger.info("Cancelled ongoing training process")
        
        # Keep the current model configuration but indicate we're back to customization stage
        response = {
            'success': True,
            'message': 'Returned to model customization stage'
        }
        
        # Add dataset_info if available
        if current_data is not None and current_target is not None:
            dataset_info = {
                'n_samples': len(current_data) + len(current_target),
                'n_features': len(current_data.columns),
                'target_type': 'classification' if len(current_target.unique()) < 20 else 'regression',
                'n_classes': len(current_target.unique()) if len(current_target.unique()) < 20 else None
            }
            response['dataset_info'] = dataset_info
        
        # Return model architecture details if available
        if current_model is not None:
            # Get model summary
            stringio = io.StringIO()
            current_model.summary(print_fn=lambda x: stringio.write(x + '\n'))
            summary_string = stringio.getvalue()
            stringio.close()
            
            response['model_summary'] = summary_string
        
        return jsonify(response)
        
    except Exception as e:
        logger.exception("Exception in go_back route.")
        return jsonify({'error': str(e)}), 500

def validate_and_fix_code(code):
    """Validate and fix any issues in the generated code"""
    # Fix common issues in the code
    
    # 1. Ensure train_test_split uses the right parameters
    if 'test_size=0.2' in code and training_config and 'test_size' in training_config:
        code = code.replace('test_size=0.2', f"test_size={training_config['test_size']}")
    
    # 2. Ensure epochs and batch_size match user configuration
    if 'epochs=50' in code and training_config and 'epochs' in training_config:
        code = code.replace('epochs=50', f"epochs={training_config['epochs']}")
    
    if 'batch_size=32' in code and training_config and 'batch_size' in training_config:
        code = code.replace('batch_size=32', f"batch_size={training_config['batch_size']}")
    
    # 3. Fix optimizer if specified
    if training_config and 'optimizer' in training_config:
        # Find the line with model.compile
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if 'model.compile(' in line and "optimizer='adam'" in line:
                lines[i] = line.replace("optimizer='adam'", f"optimizer='{training_config['optimizer']}'")
        code = '\n'.join(lines)
    
    # Add any additional validation or fixes as needed
    
    return code

def generate_model_code():
    """Generate Python code to recreate the current model with the original architecture"""
    if current_model is None:
        logger.warning("No current model to generate code for.")
        return ""

    # Get current loss function and metrics
    loss = current_model.loss if hasattr(current_model, 'loss') else "'sparse_categorical_crossentropy'"
    metrics = "['accuracy']" if hasattr(current_model, 'metrics_names') and 'accuracy' in current_model.metrics_names else "['mae']"
    
    # Get training configuration for code
    global training_config
    if not training_config:
        training_config = {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 20,
            'test_size': 0.2
        }
    
    optimizer = training_config.get('optimizer', 'adam')
    learning_rate = training_config.get('learning_rate', 0.001)
    batch_size = training_config.get('batch_size', 32)
    epochs = training_config.get('epochs', 20)
    test_size = training_config.get('test_size', 0.2)

    # Get layer configuration from the model
    layer_configs = []
    for layer in current_model.layers:
        config = layer.get_config()
        layer_type = layer.__class__.__name__
        layer_configs.append({
            'type': layer_type,
            'config': config
        })
    logger.debug(f"Layer configurations: {layer_configs}")

    code = f"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, GRU, BatchNormalization, Reshape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load and preprocess your data
def preprocess_data(df, target_column):
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle categorical features
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Handle categorical target
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, le

# Build model with your custom architecture
def build_model(input_dim={current_data.shape[1] if current_data is not None else 'your_input_dimension'}):
    model = Sequential()
"""

    # Generate layer code based on the actual model layers
    for i, layer_config in enumerate(layer_configs):
        layer_type = layer_config['type']
        
        if layer_type == 'Dense':
            units = layer_config['config']['units']
            activation = layer_config['config']['activation']
            
            # Get regularization if present
            regularization = ""
            if 'kernel_regularizer' in layer_config['config'] and layer_config['config']['kernel_regularizer']:
                reg_config = layer_config['config']['kernel_regularizer']
                reg_type = reg_config.get('class_name', '').lower()
                
                if 'l1' in reg_type:
                    l1_value = reg_config.get('config', {}).get('l1', 0.01)
                    regularization = f", kernel_regularizer=tf.keras.regularizers.l1({l1_value})"
                elif 'l2' in reg_type:
                    l2_value = reg_config.get('config', {}).get('l2', 0.01)
                    regularization = f", kernel_regularizer=tf.keras.regularizers.l2({l2_value})"
                elif 'l1_l2' in reg_type:
                    l1_value = reg_config.get('config', {}).get('l1', 0.01)
                    l2_value = reg_config.get('config', {}).get('l2', 0.01)
                    regularization = f", kernel_regularizer=tf.keras.regularizers.l1_l2(l1={l1_value}, l2={l2_value})"
            
            # Use better weight initialization - use He for ReLU and Glorot for others
            kernel_initializer = None
            activation = layer_config['config'].get('activation', 'linear')
            if activation == 'relu':
                kernel_initializer = 'he_uniform'
            else:
                kernel_initializer = 'glorot_uniform'
            
            # Get the units for Dense/LSTM/GRU layers
            units = layer_config['config'].get('units', 32)
            
            # Add input dimension check for first layer
            input_dim = current_data.shape[1] if current_data is not None else 'input_dim'
            
            logger.debug(f"Layer {i} configuration: Type={layer_type}, Units={units if layer_type not in ['Dropout', 'BatchNormalization'] else 'N/A'}")
            
            # First layer needs input shape for most layer types
            if i == 0:
                code += f"    model.add(Dense({units}, activation='{activation}', input_shape=(input_dim,){regularization}, kernel_initializer='{kernel_initializer}'))\n"
            else:
                code += f"    model.add(Dense({units}, activation='{activation}'{regularization}, kernel_initializer='{kernel_initializer}'))\n"
        
        elif layer_type == 'Dropout':
            rate = layer_config['config']['rate']
            code += f"    model.add(Dropout({rate}))\n"
        
        elif layer_type == 'Reshape':
            target_shape = layer_config['config']['target_shape']
            code += f"    model.add(Reshape({target_shape}, input_shape=(input_dim,)))\n"
            
        elif layer_type == 'Flatten':
            code += f"    model.add(Flatten())\n"
            
        elif layer_type == 'LSTM':
            units = layer_config['config']['units']
            return_sequences = layer_config['config'].get('return_sequences', False)
            if i == 0:
                code += f"    # For LSTM with tabular data, reshaping input features as time steps\n"
                code += f"    model.add(Reshape((input_dim, 1), input_shape=(input_dim,)))\n"
                code += f"    model.add(LSTM({units}, return_sequences={return_sequences}, kernel_initializer='glorot_uniform'))\n"
            else:
                code += f"    model.add(LSTM({units}, return_sequences={return_sequences}, kernel_initializer='glorot_uniform'))\n"
                
        elif layer_type == 'GRU':
            units = layer_config['config']['units']
            return_sequences = layer_config['config'].get('return_sequences', False)
            if i == 0:
                code += f"    # For GRU with tabular data, reshaping input features as time steps\n"
                code += f"    model.add(Reshape((input_dim, 1), input_shape=(input_dim,)))\n"
                code += f"    model.add(GRU({units}, return_sequences={return_sequences}, kernel_initializer='glorot_uniform'))\n"
            else:
                code += f"    model.add(GRU({units}, return_sequences={return_sequences}, kernel_initializer='glorot_uniform'))\n"
                
        elif layer_type == 'BatchNormalization':
            code += f"    model.add(BatchNormalization())\n"
            
        else:
            # For any other layer types
            code += f"    # Layer type '{layer_type}' included in original model\n"

    # Use the current optimizer and loss function
    optimizer_config = f"'{optimizer}'"
    # Add learning rate if not the default
    if optimizer != 'adam' or learning_rate != 0.001:
        optimizer_config = f"tf.keras.optimizers.{optimizer.capitalize()}(learning_rate={learning_rate})"

    code += f"""
    model.compile(optimizer={optimizer_config}, loss={loss}, metrics={metrics})
    return model

# Train model
def train_model(model, X_train, y_train, X_test, y_test, epochs={epochs}, batch_size={batch_size}, is_infinite=False):
    # No early stopping callback - removed as requested
    callbacks = []
    if is_infinite:
        print("Training in infinite mode. Press Ctrl+C to stop training when satisfied.")
    
    try:
        history = model.fit(
            X_train, y_train,
            epochs=999999 if is_infinite else epochs,  # Use very large number for infinite mode
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=callbacks  # Empty callbacks list - no early stopping
        )
    except KeyboardInterrupt:
        print("Training stopped by user.")
    
    return history

# Evaluate model
def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {{test_acc:.4f}}")
    print(f"Test loss: {{test_loss:.4f}}")
    return test_loss, test_acc

# Plot training history
def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Check if accuracy metrics exist in history
    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
    elif 'mae' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Make predictions
def predict(model, X_new, scaler, label_encoder=None):
    # Scale new data using the same scaler
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
    
    # For classification with label encoder, convert back to original labels
    if label_encoder is not None and hasattr(label_encoder, 'inverse_transform'):
        if predictions.shape[1] > 1:  # Multi-class classification
            predictions = np.argmax(predictions, axis=1)
        else:  # Binary classification
            predictions = (predictions > 0.5).astype(int).flatten()
        predictions = label_encoder.inverse_transform(predictions)
    
    return predictions

# Visualize predictions (for regression)
def visualize_predictions(X_test, y_test, predictions):
    plt.figure(figsize=(10, 6))
    
    # Sort the data for line plot
    sort_idx = np.argsort(X_test[:, 0])
    X_sorted = X_test[sort_idx, 0]
    y_sorted = y_test[sort_idx]
    predictions_sorted = predictions[sort_idx]
    
    plt.scatter(X_sorted, y_sorted, label='Actual values', alpha=0.6)
    plt.plot(X_sorted, predictions_sorted, color='red', linewidth=2, label='Predictions')
    plt.title('Model Predictions vs Actual Values')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.savefig('prediction_visualization.png')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('your_data.csv')

    # Preprocess
    X, y, scaler, label_encoder = preprocess_data(df, 'target_column')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)

    # Build and train model
    model = build_model(input_dim=X.shape[1])  # Pass input dimension dynamically
    
    # Choose whether to use infinite training mode
    use_infinite_training = False  # Change to True for infinite training
    
    history = train_model(model, X_train, y_train, X_test, y_test, is_infinite=use_infinite_training)

    # Evaluate
    test_loss, test_acc = evaluate_model(model, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)

    # Save model
    model.save('model.h5')
    
    # Quick model summary
    model.summary()
    
    # Make predictions on test data
    predictions = predict(model, X_test, scaler, label_encoder)
    
    # Visualize predictions if it's a regression task
    if len(np.unique(y)) > 10:  # Assuming regression if many unique values
        visualize_predictions(X_test, y_test, predictions)
"""

    return code

@app.route('/predict', methods=['POST'])
def make_prediction():
    """Endpoint for making predictions using the trained model"""
    global current_model, current_data, scaler, label_encoder,training_config
    
    try:
        if current_model is None:
            return jsonify({'error': 'No model has been trained yet'}), 400
            
        # Get prediction data from request
        data = request.json
        input_data = data.get('input_data')
        
        if not input_data:
            # If no specific input data is provided, use a sample from the test data
            # Create test data if not already done
            if current_data is None:
                return jsonify({'error': 'No dataset has been uploaded'}), 400
                
            # Preprocess the data
            X = current_data
            y = current_target
            
            # Handle categorical features
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = LabelEncoder().fit_transform(X[col])
                
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Split to get test data
            _, X_test, _, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Use a small sample for prediction
            sample_size = min(10, len(X_test))
            sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test[sample_indices]
            y_sample = y_test.iloc[sample_indices] if hasattr(y_test, 'iloc') else y_test[sample_indices]
            
            # Make predictions
            predictions = current_model.predict(X_sample)
            
            # Get the dataset type from training config
            global training_config
            dataset_type = training_config.get('dataset_type', 'classification')
            is_regression = dataset_type == 'regression'
            logger.debug(f"Making predictions for dataset type: {dataset_type}")
            
            # Convert predictions to proper format based on dataset type
            if not is_regression:
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    # Multi-class classification
                    pred_labels = np.argmax(predictions, axis=1)
                    if hasattr(label_encoder, 'inverse_transform'):
                        pred_labels = label_encoder.inverse_transform(pred_labels)
                    # Create probabilities dictionary for multi-class
                    probabilities = {}
                    for i, prob in enumerate(predictions[0]):
                        class_name = str(label_encoder.inverse_transform([i])[0]) if hasattr(label_encoder, 'inverse_transform') else f'Class {i}'
                        probabilities[class_name] = float(prob)
                else:
                    # Binary classification
                    pred_labels = (predictions > 0.5).astype(int).flatten()
                    if hasattr(label_encoder, 'inverse_transform'):
                        pred_labels = label_encoder.inverse_transform(pred_labels)
                    # Create probabilities dictionary for binary classification
                    probabilities = {}
                    class_names = label_encoder.classes_ if hasattr(label_encoder, 'classes_') else ['Class 0', 'Class 1']
                    probabilities[str(class_names[0])] = float(1 - predictions[0][0])
                    probabilities[str(class_names[1])] = float(predictions[0][0])
            else:
                # Regression
                pred_labels = predictions
                # Create a dummy probabilities dictionary for regression to ensure consistent response format
                probabilities = {'Predicted Value': float(predictions[0][0]) if len(predictions.shape) > 1 else float(predictions[0])}
            
            # Generate visualization based on dataset type
            visualization_url = None
            if is_regression:
                visualization_url = generate_regression_visualization(X_sample, y_sample, predictions)
            else:  # Classification
                if len(np.unique(y)) == 2:  # Binary classification
                    visualization_url = generate_classification_visualization(X_sample, y_sample, predictions)
                else:  # Multi-class classification
                    # No visualization for multi-class yet
                    pass
            
            # Format response
            result = {
                'success': True,
                'prediction': pred_labels[0] if hasattr(pred_labels, '__len__') else pred_labels,
                'predictions': pred_labels.tolist() if hasattr(pred_labels, 'tolist') else pred_labels,
                'actual_values': y_sample.tolist() if hasattr(y_sample, 'tolist') else y_sample,
                'visualization_url': visualization_url,
                'probabilities': probabilities
            }
            
            return jsonify(result)
        else:
            # Process custom input data
            # Convert input data to DataFrame if it's not already
            if not isinstance(input_data, pd.DataFrame):
                input_df = pd.DataFrame(input_data)
            else:
                input_df = input_data
                
            # Make sure it has the same columns as the training data
            if current_data is not None:
                missing_cols = set(current_data.columns) - set(input_df.columns)
                if missing_cols:
                    return jsonify({'error': f'Input data missing required columns: {missing_cols}'}), 400
            
            # Preprocess input data
            # Handle categorical features
            for col in input_df.select_dtypes(include=['object']).columns:
                input_df[col] = LabelEncoder().fit_transform(input_df[col])
                
            # Scale features
            input_scaled = scaler.transform(input_df)
            
            # Make predictions
            predictions = current_model.predict(input_scaled)
            
            # Get the dataset type from training config
            # global training_config
            dataset_type = training_config.get('dataset_type', 'classification')
            is_regression = dataset_type == 'regression'
            logger.debug(f"Making predictions for dataset type: {dataset_type}")
            
            # Convert predictions to proper format based on dataset type
            if not is_regression:
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    # Multi-class classification
                    pred_labels = np.argmax(predictions, axis=1)
                    if hasattr(label_encoder, 'inverse_transform'):
                        pred_labels = label_encoder.inverse_transform(pred_labels)
                    # Create probabilities dictionary for multi-class
                    probabilities = {}
                    for i, prob in enumerate(predictions[0]):
                        class_name = str(label_encoder.inverse_transform([i])[0]) if hasattr(label_encoder, 'inverse_transform') else f'Class {i}'
                        probabilities[class_name] = float(prob)
                else:
                    # Binary classification
                    pred_labels = (predictions > 0.5).astype(int).flatten()
                    if hasattr(label_encoder, 'inverse_transform'):
                        pred_labels = label_encoder.inverse_transform(pred_labels)
                    # Create probabilities dictionary for binary classification
                    probabilities = {}
                    class_names = label_encoder.classes_ if hasattr(label_encoder, 'classes_') else ['Class 0', 'Class 1']
                    probabilities[str(class_names[0])] = float(1 - predictions[0][0])
                    probabilities[str(class_names[1])] = float(predictions[0][0])
            else:
                # Regression
                pred_labels = predictions
                # Create a dummy probabilities dictionary for regression to ensure consistent response format
                probabilities = {'Predicted Value': float(predictions[0][0]) if len(predictions.shape) > 1 else float(predictions[0])}
            
            return jsonify({
                'success': True,
                'prediction': pred_labels[0] if hasattr(pred_labels, '__len__') else pred_labels,
                'predictions': pred_labels.tolist() if hasattr(pred_labels, 'tolist') else pred_labels,
                'probabilities': probabilities
            })
    
    except Exception as e:
        logger.exception("Error in prediction endpoint")
        return jsonify({'error': str(e)}), 500

def generate_regression_visualization(X_sample, y_sample, predictions):
    """Generate a visualization for regression predictions"""
    try:
        plt.figure(figsize=(10, 6))
        
        # Sort the data for line plot if X has only one feature
        if X_sample.shape[1] == 1:
            sort_idx = np.argsort(X_sample[:, 0])
            X_sorted = X_sample[sort_idx, 0]
            y_sorted = y_sample[sort_idx] if isinstance(y_sample, np.ndarray) else y_sample.values[sort_idx]
            predictions_sorted = predictions[sort_idx]
            
            plt.scatter(X_sorted, y_sorted, label='Actual values', alpha=0.6)
            plt.plot(X_sorted, predictions_sorted, color='red', linewidth=2, label='Predictions')
        else:
            # If more than one feature, plot actual vs predicted
            plt.scatter(y_sample, predictions, alpha=0.6)
            
            # Add a perfect prediction line
            min_val = min(np.min(y_sample), np.min(predictions))
            max_val = max(np.max(y_sample), np.max(predictions))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
        
        plt.title('Model Predictions vs Actual Values')
        plt.legend()
        
        # Save the plot to a temporary file
        img_name = f'prediction_viz_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        img_path = os.path.join('static', img_name)
        
        # Make sure the static directory exists
        os.makedirs('static', exist_ok=True)
        
        plt.savefig(img_path)
        plt.close()
        
        return f'/static/{img_name}'
    
    except Exception as e:
        logger.exception("Error generating regression visualization")
        return None

def generate_classification_visualization(X_sample, y_sample, predictions):
    """Generate a visualization for classification predictions"""
    try:
        # For now, we'll just create a simple bar chart comparing actual vs predicted
        plt.figure(figsize=(10, 6))
        
        # Convert predictions to binary if they're probabilities
        if len(predictions.shape) > 1:
            pred_labels = (predictions > 0.5).astype(int).flatten()
        else:
            pred_labels = predictions
            
        # Count correct and incorrect predictions
        correct = np.sum(pred_labels == y_sample)
        incorrect = len(y_sample) - correct
        
        # Create bar chart
        plt.bar(['Correct', 'Incorrect'], [correct, incorrect])
        plt.title('Prediction Accuracy')
        plt.ylabel('Count')
        
        # Add a pie chart for class distribution
        plt.figure(figsize=(8, 8))
        class_names = ['Class 0', 'Class 1']
        if hasattr(label_encoder, 'classes_'):
            class_names = label_encoder.classes_
            
        plt.pie([np.sum(pred_labels == 0), np.sum(pred_labels == 1)], 
                labels=class_names,
                autopct='%1.1f%%',
                shadow=True,
                startangle=90)
        plt.axis('equal')
        plt.title('Predicted Class Distribution')
        
        # Save the plots to a temporary file
        img_name = f'classification_viz_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        img_path = os.path.join('static', img_name)
        
        # Make sure the static directory exists
        os.makedirs('static', exist_ok=True)
        
        plt.savefig(img_path)
        plt.close('all')
        
        return f'/static/{img_name}'
    
    except Exception as e:
        logger.exception("Error generating classification visualization")
        return None

@app.route('/get_training_plot')
def get_training_plot():
    """Generate and return a plot of training history"""
    global current_model
    
    try:
        if not hasattr(current_model, 'history') or not current_model.history or not current_model.history.history:
            return jsonify({'error': 'No training history available'}), 404
            
        history = current_model.history.history
        
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'])
        if 'val_loss' in history:
            plt.plot(history['val_loss'])
            plt.legend(['Training', 'Validation'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Plot accuracy or MAE
        plt.subplot(1, 2, 2)
        if 'accuracy' in history:
            plt.plot(history['accuracy'])
            if 'val_accuracy' in history:
                plt.plot(history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.legend(['Training', 'Validation'])
        elif 'mae' in history:
            plt.plot(history['mae'])
            if 'val_mae' in history:
                plt.plot(history['val_mae'])
            plt.title('Model MAE')
            plt.ylabel('MAE')
            plt.legend(['Training', 'Validation'])
        
        plt.xlabel('Epoch')
        plt.tight_layout()
        
        # Save the plot to a temporary file
        img_name = f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        img_path = os.path.join('static', img_name)
        
        # Make sure the static directory exists
        os.makedirs('static', exist_ok=True)
        
        plt.savefig(img_path)
        plt.close()
        
        return jsonify({
            'success': True,
            'plot_url': f'/static/{img_name}'
        })
        
    except Exception as e:
        logger.exception("Error generating training plot")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)  # Keep debug=True for development
    