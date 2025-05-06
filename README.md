# Neural Network Builder

A web-based application for building, training, and deploying neural networks with a user-friendly interface. This tool allows users to easily create, customize, and train neural networks for both classification and regression tasks.

## Features

- **Interactive Model Building**: Visually design neural network architectures
- **Dataset Handling**: Upload and preprocess CSV datasets automatically
- **Smart Architecture Suggestions**: Get AI-powered suggestions for network architecture based on your dataset
- **Real-time Training Monitoring**: Watch training progress with live metrics and visualizations
- **Flexible Training Options**: 
  - Standard training with configurable epochs
  - Infinite training mode with manual stopping
  - Pause/Resume training capability
- **Model Export**: Download trained models with generated Python code for deployment
- **Visualization Tools**: 
  - Training history plots
  - Prediction visualization
  - Model architecture visualization

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Flask
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd nn-builder
```

2. Set up a Python virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Follow these steps in the interface:
   - Upload your dataset (CSV format)
   - Select the target column
   - Choose between classification or regression
   - Customize the suggested architecture or use it as-is
   - Configure training parameters
   - Start training and monitor progress
   - Export the trained model when satisfied

## Code Generation

The application can generate ready-to-use Python code for:
- Data preprocessing
- Model architecture
- Training configuration
- Prediction functionality
- Result visualization

## Project Structure

```
NN Builder/
├── app.py              # Main Flask application
├── static/            
│   ├── css/           # Stylesheets
│   └── js/            # JavaScript files
├── templates/         
│   └── index.html     # Main web interface
└── README.md          # This file
```

## Training Options

### Standard Training
- Set number of epochs
- Configure batch size
- Choose optimizer and learning rate
- Monitor training metrics in real-time

### Infinite Training Mode
- Train without a fixed epoch limit
- Stop manually when satisfied with performance
- Pause and resume capability
- Real-time metric monitoring

## Model Export

The exported model package includes:
- Trained model in HDF5 format
- Generated Python code for deployment
- Model architecture description
- Training configuration

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the amazing deep learning framework
- Flask team for the web framework
- Google for the Gemini API integration