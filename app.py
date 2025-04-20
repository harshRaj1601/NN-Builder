from flask import Flask, render_template, request, jsonify, send_file
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
        if target_column in df.columns:
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
    logger.debug("Entering build_model route")
    try:
        model_config = request.json
        logger.debug(f"Model config: {model_config}")

        # Create Sequential model
        model = tf.keras.Sequential()

        # Add layers based on config
        for i, layer in enumerate(model_config['architecture']['layers']):
            if i == 0:
                # First layer needs input shape
                if layer['type'] == 'Dense':
                    model.add(tf.keras.layers.Dense(
                        layer['units'],
                        activation=layer['activation'],
                        input_shape=(current_data.shape[1],)
                    ))
            else:
                if layer['type'] == 'Dense':
                    model.add(tf.keras.layers.Dense(
                        layer['units'],
                        activation=layer['activation']
                    ))
                elif layer['type'] == 'Dropout':
                    model.add(tf.keras.layers.Dropout(layer['rate']))
                # Add more layer types as needed

        # Compile model
        model.compile(
            optimizer=model_config.get('optimizer', 'adam'),
            loss=model_config.get('loss', 'sparse_categorical_crossentropy'),
            metrics=['accuracy']
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

@app.route('/train_model', methods=['POST'])
def train_model():
    logger.debug("Entering train_model route")
    try:
        training_config = request.json
        logger.debug(f"Training config: {training_config}")

        # Preprocess data
        X = current_data
        y = current_target

        #Handle categorical features
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        #Handle categorical target
        if y.dtype == 'object' or y.dtype.name == 'category':
            y = label_encoder.fit_transform(y)

        # Scale features
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        logger.debug("Data preprocessing and splitting complete.")

        # Train model
        history = current_model.fit(
            X_train, y_train,
            epochs=training_config.get('epochs', 50),
            batch_size=training_config.get('batch_size', 32),
            validation_data=(X_test, y_test),
            verbose=1
        )
        logger.debug("Model training complete.")

        # Generate loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Convert plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        logger.debug("Loss plot generated.")

        # Evaluate model
        test_loss, test_acc = current_model.evaluate(X_test, y_test, verbose=0)
        logger.debug(f"Model evaluation complete. Test loss: {test_loss}, Test accuracy: {test_acc}")

        return jsonify({
            'success': True,
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'training_plot': image_base64
        })

    except Exception as e:
        logger.exception("Exception in train_model route.")
        return jsonify({'error': str(e)}), 500

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

def generate_model_code():
    """Generate Python code to recreate the current model"""
    if current_model is None:
        logger.warning("No current model to generate code for.")
        return ""

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

# Build model
def build_model():
    model = Sequential()
"""

    for i, layer_config in enumerate(layer_configs):
        if layer_config['type'] == 'Dense':
            if i == 0:
                code += f"    model.add(Dense({layer_config['config']['units']}, activation='{layer_config['config']['activation']}', input_shape=({current_data.shape[1]},)))\n"
            else:
                code += f"    model.add(Dense({layer_config['config']['units']}, activation='{layer_config['config']['activation']}'))\n"
        elif layer_config['type'] == 'Dropout':
            code += f"    model.add(Dropout({layer_config['config']['rate']}))\n"

    code += """
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model
def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train model
    model = build_model()
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Evaluate
    test_loss, test_acc = evaluate_model(model, X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Save model
    model.save('model.h5')
"""

    return code

if __name__ == '__main__':
    app.run(debug=False)  # Keep debug=True for development