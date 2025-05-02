// Prediction Interface
class PredictionInterface {
    constructor(config) {
        this.config = {
            containerId: 'prediction-container',
            inputsContainerId: 'prediction-inputs',
            resultsContainerId: 'prediction-results',
            runButtonId: 'run-prediction-btn',
            features: []
        };
        
        // Override default config with provided values
        Object.assign(this.config, config);
        
        this.container = document.getElementById(this.config.containerId);
        this.inputsContainer = document.getElementById(this.config.inputsContainerId);
        this.resultsContainer = document.getElementById(this.config.resultsContainerId);
        this.runButton = document.getElementById(this.config.runButtonId);
        
        this.inputValues = {};
        this.lastPrediction = null;
        
        this.init();
    }
    
    init() {
        // Set up event listeners
        if (this.runButton) {
            this.runButton.addEventListener('click', () => this.runPrediction());
        }
    }
    
    setFeatures(features) {
        this.config.features = features;
        this.renderInputs();
    }
    
    renderInputs() {
        if (!this.inputsContainer) return;
        
        // Clear existing inputs
        this.inputsContainer.innerHTML = '';
        
        // Create inputs for each feature
        this.config.features.forEach(feature => {
            // Create form group
            const formGroup = document.createElement('div');
            formGroup.className = 'form-group';
            
            // Create label
            const label = document.createElement('label');
            label.className = 'form-label';
            label.setAttribute('for', `prediction-input-${feature.name}`);
            label.textContent = feature.label || feature.name;
            
            // Create input
            const input = document.createElement('input');
            input.type = feature.type || 'number';
            input.className = 'form-control';
            input.id = `prediction-input-${feature.name}`;
            input.name = feature.name;
            input.placeholder = feature.placeholder || '';
            
            // Set min/max/step for numeric inputs
            if (input.type === 'number') {
                if (feature.min !== undefined) input.min = feature.min;
                if (feature.max !== undefined) input.max = feature.max;
                if (feature.step !== undefined) input.step = feature.step;
            }
            
            // Set default value if provided
            if (feature.defaultValue !== undefined) {
                input.value = feature.defaultValue;
                this.inputValues[feature.name] = feature.defaultValue;
            }
            
            // Add event listener to update inputValues
            input.addEventListener('input', e => {
                this.inputValues[feature.name] = e.target.type === 'number' 
                    ? parseFloat(e.target.value) 
                    : e.target.value;
            });
            
            // Append to form group
            formGroup.appendChild(label);
            formGroup.appendChild(input);
            
            // Append to inputs container
            this.inputsContainer.appendChild(formGroup);
        });
    }
    
    getInputValues() {
        return { ...this.inputValues };
    }
    
    async runPrediction() {
        if (!this.runButton || !this.resultsContainer) return;
        
        // Disable run button and show loading state
        this.runButton.disabled = true;
        this.runButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Predicting...';
        
        try {
            // Get all input values
            const inputData = this.getInputValues();
            
            // Call prediction API
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ inputs: inputData }),
            });
            
            if (!response.ok) {
                throw new Error('Prediction request failed');
            }
            
            const result = await response.json();
            
            // Store the prediction result
            this.lastPrediction = result;
            
            // Display prediction result
            this.displayPredictionResult(result);
        } catch (error) {
            console.error('Error during prediction:', error);
            
            // Show error in results container
            this.resultsContainer.style.display = 'block';
            this.resultsContainer.innerHTML = `
                <div class="prediction-error">
                    <i class="fas fa-exclamation-circle"></i>
                    <p>Error during prediction. Please try again.</p>
                </div>
            `;
        } finally {
            // Re-enable run button
            this.runButton.disabled = false;
            this.runButton.innerHTML = '<i class="fas fa-play me-2"></i>Run Prediction';
        }
    }
    
    displayPredictionResult(result) {
        if (!this.resultsContainer) return;
        
        // Determine if we're showing classification or regression results
        const isClassification = result.probabilities && Object.keys(result.probabilities).length > 0;
        
        if (isClassification) {
            this.displayClassificationResult(result);
        } else {
            this.displayRegressionResult(result);
        }
        
        // Show results container
        this.resultsContainer.style.display = 'block';
    }
    
    displayClassificationResult(result) {
        // Get probabilities sorted by value (descending)
        const probabilities = Object.entries(result.probabilities)
            .sort((a, b) => b[1] - a[1]);
        
        // Create HTML for classification result
        let html = `
            <div class="prediction-value">${result.prediction}</div>
            <div class="prediction-explanation">Predicted Class</div>
            <div class="prediction-chart-container">
                <canvas id="prediction-chart" height="200"></canvas>
            </div>
        `;
        
        // Update the results container
        this.resultsContainer.innerHTML = html;
        
        // Create chart for probabilities
        const ctx = document.getElementById('prediction-chart').getContext('2d');
        
        // Destroy existing chart if it exists
        if (window.predictionChart) {
            window.predictionChart.destroy();
        }
        
        window.predictionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: probabilities.map(p => p[0]),
                datasets: [{
                    label: 'Class Probability',
                    data: probabilities.map(p => p[1]),
                    backgroundColor: probabilities.map((p, i) => {
                        return p[0] === result.prediction 
                            ? 'rgba(0, 206, 209, 0.7)' 
                            : 'rgba(123, 104, 238, 0.7)';
                    }),
                    borderColor: probabilities.map((p, i) => {
                        return p[0] === result.prediction 
                            ? 'rgb(0, 206, 209)' 
                            : 'rgb(123, 104, 238)';
                    }),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return (context.raw * 100).toFixed(2) + '%';
                            }
                        }
                    }
                }
            }
        });
    }
    
    displayRegressionResult(result) {
        // Create HTML for regression result
        let html = `
            <div class="prediction-value">${parseFloat(result.prediction).toFixed(4)}</div>
            <div class="prediction-explanation">Predicted Value</div>
        `;
        
        // Add additional information if available
        if (result.mse || result.mae || result.r2) {
            html += `
                <div class="prediction-metrics">
                    <div class="prediction-metric">
                        <span class="metric-label">MAE:</span>
                        <span class="metric-value">${result.mae ? result.mae.toFixed(4) : 'N/A'}</span>
                    </div>
                    <div class="prediction-metric">
                        <span class="metric-label">MSE:</span>
                        <span class="metric-value">${result.mse ? result.mse.toFixed(4) : 'N/A'}</span>
                    </div>
                    <div class="prediction-metric">
                        <span class="metric-label">R²:</span>
                        <span class="metric-value">${result.r2 ? result.r2.toFixed(4) : 'N/A'}</span>
                    </div>
                </div>
            `;
        }
        
        // Update the results container
        this.resultsContainer.innerHTML = html;
    }
    
    // Method to generate example inputs based on features
    generateExampleInputs() {
        const exampleInputs = {};
        
        this.config.features.forEach(feature => {
            if (feature.example !== undefined) {
                // Use provided example value
                exampleInputs[feature.name] = feature.example;
            } else if (feature.type === 'number') {
                // Generate a random number within the feature's range
                const min = feature.min !== undefined ? feature.min : 0;
                const max = feature.max !== undefined ? feature.max : 10;
                exampleInputs[feature.name] = Math.random() * (max - min) + min;
                
                // Round to reasonable precision
                if (feature.step) {
                    const step = parseFloat(feature.step);
                    exampleInputs[feature.name] = Math.round(exampleInputs[feature.name] / step) * step;
                } else {
                    exampleInputs[feature.name] = parseFloat(exampleInputs[feature.name].toFixed(2));
                }
            } else {
                // Use empty string for non-numeric fields without examples
                exampleInputs[feature.name] = '';
            }
        });
        
        return exampleInputs;
    }
    
    // Method to fill inputs with example values
    fillExampleInputs() {
        const exampleInputs = this.generateExampleInputs();
        
        // Update input fields
        Object.entries(exampleInputs).forEach(([name, value]) => {
            const input = document.getElementById(`prediction-input-${name}`);
            if (input) {
                input.value = value;
                this.inputValues[name] = value;
            }
        });
    }
}

// Global instance for prediction interface
let predictionInterface;

// Initialize prediction interface
function initPredictionInterface(config = {}) {
    predictionInterface = new PredictionInterface(config);
    return predictionInterface;
}

// Example feature configuration for reference
const exampleFeatures = [
    {
        name: 'feature1',
        label: 'Feature 1',
        type: 'number',
        min: 0,
        max: 10,
        step: 0.1,
        defaultValue: 5,
        example: 7.5
    },
    {
        name: 'feature2',
        label: 'Feature 2',
        type: 'number',
        min: -1,
        max: 1,
        step: 0.01,
        defaultValue: 0
    }
]; 