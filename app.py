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

        # Generate model architecture suggestion using Gemini
        dataset_info = {
            'n_samples': len(df),
            'n_features': len(current_data.columns),
            'target_type': 'classification' if len(current_target.unique()) < 20 else 'regression',
            'n_classes': len(current_target.unique()) if len(current_target.unique()) < 20 else None
        }
        logger.debug(f"Dataset info: {dataset_info}")

        prompt = f"""
        Given a dataset with following properties:
        - Number of samples: {dataset_info['n_samples']}
        - Number of features: {dataset_info['n_features']}
        - Task type: {dataset_info['target_type']}
        - Number of classes: {dataset_info['n_classes']}"""
        
        prompt += """
        Suggest an optimal neural network architecture for this classification task.  Provide the response as a *pure*, valid JSON string with *no* surrounding text or explanations. Do start with "{" and do not use newline codes or tabs. The JSON should have a "layers" field, an "optimizer" field, a "learning_rate" field, a "loss" field, a "metrics" field, a "batch_size" field, and an "epochs" field.
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
            suggested_architecture = {
                'layers': [
                    {'type': 'Dense', 'units': 64, 'activation': 'relu'},
                    {'type': 'Dropout', 'rate': 0.2},
                    {'type': 'Dense', 'units': 32, 'activation': 'relu'},
                    {'type': 'Dense', 'units': dataset_info['n_classes'] or 1,
                     'activation': 'softmax' if dataset_info['target_type'] == 'classification' else 'linear'}
                ]
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

        # Add layers based on config
        for i, layer in enumerate(model_config['architecture']['layers']):
            # Ensure each layer has a type
            layer_type = layer.get('type', 'Dense')  # Default to Dense if type not specified
            
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
                
            if i == 0:
                # First layer needs input shape
                if layer_type == 'Dense':
                    model.add(tf.keras.layers.Dense(
                        units=layer['units'],
                        activation=layer['activation'],
                        kernel_regularizer=regularizer,
                        input_shape=(input_features,)  # Use the detected input features
                    ))
                else:
                    if layer_type == 'Dense':
                        model.add(tf.keras.layers.Dense(
                            units=layer['units'],
                            activation=layer['activation'],
                            kernel_regularizer=regularizer
                        ))
                    elif layer_type == 'Dropout':
                        # Handle the case where rate might not be provided
                        dropout_rate = layer.get('rate', 0.2)  # Default to 0.2 if not specified
                        model.add(tf.keras.layers.Dropout(rate=dropout_rate))

        # Determine appropriate loss function based on output layer
        loss = model_config.get('loss')
        if not loss:
            # Auto-detect loss function based on output layer and data type
            output_layer = model_config['architecture']['layers'][-1]
            output_activation = output_layer.get('activation', 'linear')
            
            if current_target.dtype == 'object' or current_target.dtype.name == 'category':
                unique_values = len(current_target.unique())
                if unique_values == 2:  # Binary classification
                    loss = 'binary_crossentropy'
                else:  # Multi-class classification
                    loss = 'sparse_categorical_crossentropy'
            else:  # Regression
                loss = 'mse'
                
            logger.debug(f"Auto-detected loss function: {loss}")
        
        # Compile model with optimizer from config or default
        optimizer_name = model_config.get('optimizer', 'adam')
        learning_rate = model_config.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        else:
            optimizer = optimizer_name  # Use string name if not a special case
        
        # Set appropriate metrics based on loss function
        if loss.endswith('entropy'):
            # For classification tasks, use multiple metrics to ensure compatibility
            metrics = ['accuracy']
            logger.debug(f"Using classification metrics: {metrics}")
        else:
            # For regression tasks, use MAE
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
            'model_summary': summary_string
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
            'test_size': data.get('test_size', 0.2)  # Default to 20% test data
        }
        
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

@app.route('/train_stream')
def train_stream():
    def generate():
        global training_in_progress
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

            # Scale features
            X_scaled = scaler.fit_transform(X)

            # Get training parameters
            global training_config
            if not training_config:
                training_config = {
                    'optimizer': 'adam',
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 20,
                    'test_size': 0.2
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
                    
                    # Extract accuracy metrics, handling different possible metric names
                    acc = logs.get('accuracy', logs.get('acc', 0))
                    val_acc = logs.get('val_accuracy', logs.get('val_acc', 0))
                    
                    # For regression models or if accuracy is not found, calculate from loss
                    if acc == 0 and 'loss' in logs:
                        # Infer if this is a classification model based on loss function name
                        loss_name = current_model.loss
                        if isinstance(loss_name, str) and 'entropy' in loss_name:
                            # This is likely a classification model, but accuracy wasn't calculated
                            # We'll log this issue but keep the 0 value
                            logger.warning(f"Classification model detected but accuracy metrics not found in logs: {logs.keys()}")
                    
                    logger.debug(f"Extracted accuracy: {acc}, val_accuracy: {val_acc}")
                    
                    # Create data packet for frontend
                    data = {
                        'status': 'epoch',
                        'epoch': epoch,
                        'loss': float(logs.get('loss', 0)),
                        'val_loss': float(logs.get('val_loss', 0)),
                        'acc': float(acc),
                        'val_acc': float(val_acc)
                    }
                    
                    logger.debug(f"Sending to frontend: {data}")
                    update_queue.put(data)

            # Train model with callback
            model_callback = StreamCallback()
            
            # Start training in a separate thread
            from threading import Thread
            
            def train_model():
                global training_in_progress
                try:
                    # Log the model configuration before training
                    logger.debug(f"Starting model training with metrics: {current_model.metrics_names}")
                    logger.debug(f"Model loss function: {current_model.loss}")
                    
                    # Check if we're doing classification or regression
                    is_classification = 'accuracy' in current_model.metrics_names or current_model.loss.endswith('entropy')
                    logger.debug(f"Task type: {'Classification' if is_classification else 'Regression'}")
                    
                    # Check data shapes and types
                    logger.debug(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
                    logger.debug(f"Training data types: X_train={X_train.dtype}, y_train={y_train.dtype}")
                    logger.debug(f"Unique target values: {np.unique(y_train)}")
                    
                    if is_classification:
                        logger.debug(f"Number of classes: {len(np.unique(y_train))}")
                    
                    # Start the training process
                    history = current_model.fit(
                        X_train, y_train,
                        epochs=training_config['epochs'],
                        batch_size=training_config['batch_size'],
                        validation_data=(X_test, y_test),
                        callbacks=[model_callback],
                        verbose=1
                    )
                    
                    # Log training completion
                    logger.debug(f"Training completed. History keys: {history.history.keys()}")
                    
                    # Put completion message in queue
                    test_loss, test_acc = current_model.evaluate(X_test, y_test, verbose=0)
                    logger.debug(f"Final test metrics - loss: {test_loss}, accuracy: {test_acc}")
                    
                    completion_data = {
                        'status': 'completed',
                        'test_accuracy': float(test_acc),
                        'test_loss': float(test_loss)
                    }
                    update_queue.put(completion_data)
                except Exception as e:
                    logger.exception("Error during model training")
                    update_queue.put({
                        'status': 'error',
                        'message': str(e)
                    })
                finally:
                    # Reset the training in progress flag
                    training_in_progress = False
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
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import numpy as np
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

    return X_scaled, y, scaler

# Build model with your custom architecture
def build_model(input_dim={current_data.shape[1] if current_data is not None else 'your_input_dimension'}):
    model = Sequential()
"""

    # Generate layer code based on the actual model layers
    for i, layer_config in enumerate(layer_configs):
        if layer_config['type'] == 'Dense':
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
            
            if i == 0:
                code += f"    model.add(Dense({units}, activation='{activation}', input_shape=(input_dim,){regularization}))\n"
            else:
                code += f"    model.add(Dense({units}, activation='{activation}'{regularization}))\n"
        elif layer_config['type'] == 'Dropout':
            rate = layer_config['config']['rate']
            code += f"    model.add(Dropout({rate}))\n"

    # Use the current optimizer and loss function
    optimizer_config = f"'{optimizer}'"
    # Add learning rate if not the default
    if optimizer != 'adam' or learning_rate != 0.001:
        optimizer_config = f"tf.keras.optimizers.{optimizer.capitalize()}(learning_rate={learning_rate})"

    code += f"""
    model.compile(optimizer={optimizer_config}, loss={loss}, metrics={metrics})
    return model

# Train model
def train_model(model, X_train, y_train, X_test, y_test, epochs={epochs}, batch_size={batch_size}):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )
    return history

# Evaluate model
def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return test_loss, test_acc

# Make predictions
def predict(model, X_new, scaler):
    # Scale new data using the same scaler
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
    return predictions

# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('your_data.csv')

    # Preprocess
    X, y, scaler = preprocess_data(df, 'target_column')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)

    # Build and train model
    model = build_model(input_dim=X.shape[1])  # Pass input dimension dynamically
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Evaluate
    test_loss, test_acc = evaluate_model(model, X_test, y_test)
    print(f"Test Accuracy: {{test_acc:.4f}}")
    print(f"Test Loss: {{test_loss:.4f}}")

    # Save model
    model.save('model.h5')
    
    # Quick model summary
    model.summary()
"""

    return code

if __name__ == '__main__':
    app.run(debug=False)  # Keep debug=True for development