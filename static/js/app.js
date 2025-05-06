// Neural Network Builder - Main Application
class NeuralNetworkBuilder {
    constructor() {
        // State management
        this.state = {
            currentStep: 1,
            datasetInfo: null,
            currentFile: null,
            targetColumn: '',
            architecture: null,
            modelSummary: '',
            trainingResults: null,
            isTraining: false,
            isPaused: false,
            codeGenerated: false,
            generatedCode: '',
            csvDelimiter: null,
        };
        
        // UI elements
        this.sections = {
            datasetSection: document.getElementById('datasetSection'),
            architectureSection: document.getElementById('architectureSection'),
            modelSummarySection: document.getElementById('modelSummarySection'),
            trainingResultsSection: document.getElementById('trainingResultsSection'),
            codeSection: document.getElementById('codeSection'),
            predictionSection: document.getElementById('predictionSection')
        };
        
        // Step indicators
        this.steps = {
            step1: document.getElementById('step1'),
            step2: document.getElementById('step2'),
            step3: document.getElementById('step3'),
            step4: document.getElementById('step4'),
            step5: document.getElementById('step5')
        };
        
        // Form inputs and buttons
        this.elements = {
            datasetFile: document.getElementById('datasetFile'),
            targetColumn: document.getElementById('targetColumn'),
            fileName: document.getElementById('fileName'),
            fileDropzone: document.getElementById('fileDropzone'),
            analyzeBtn: document.getElementById('analyzeBtn'),
            buildModelBtn: document.getElementById('buildModelBtn'),
            directCodeBtn: document.getElementById('directCodeBtn'),
            startTrainingBtn: document.getElementById('startTrainingBtn'),
            pauseTrainingBtn: document.getElementById('pauseTrainingBtn'),
            downloadBtn: document.getElementById('downloadBtn'),
            generateCodeBtn: document.getElementById('generateCodeBtn'),
            copyCodeBtn: document.getElementById('copyCodeBtn'),
            epochs: document.getElementById('epochs'),
            batchSize: document.getElementById('batchSize'),
            optimizer: document.getElementById('optimizer'),
            learningRate: document.getElementById('learningRate'),
            trainTestSplit: document.getElementById('trainTestSplit'),
            loadingOverlay: document.getElementById('loadingOverlay'),
            loadingText: document.getElementById('loadingText')
        };
        
        // Training metrics storage
        this.trainingMetrics = {
            epochs: [],
            accuracy: [],
            loss: [],
            val_accuracy: [],
            val_loss: []
        };
        
        // Initialize components
        this.init();
    }
    
    init() {
        // Initialize event listeners
        this.initEventListeners();
        
        // Initialize file upload
        this.initFileUpload();
    }
    
    initEventListeners() {
        // Dataset analysis
        if (this.elements.analyzeBtn) {
            this.elements.analyzeBtn.addEventListener('click', () => this.analyzeDataset());
        }
        
        // Build model
        if (this.elements.buildModelBtn) {
            this.elements.buildModelBtn.addEventListener('click', () => this.buildModel());
        }
        
        // Direct code generation without training
        if (this.elements.directCodeBtn) {
            this.elements.directCodeBtn.addEventListener('click', () => this.generateCodeDirectly());
        }
        
        // Back to architecture button
        const backToArchitectureBtn = document.getElementById('backToArchitectureBtn');
        if (backToArchitectureBtn) {
            backToArchitectureBtn.addEventListener('click', () => this.goBackToArchitecture());
        }
        
        // Back to training button
        const backToTrainingBtn = document.getElementById('backToTrainingBtn');
        if (backToTrainingBtn) {
            backToTrainingBtn.addEventListener('click', () => this.goBackToTraining());
        }
        
        // Back to results button
        const backToResultsBtn = document.getElementById('backToResultsBtn');
        if (backToResultsBtn) {
            backToResultsBtn.addEventListener('click', () => this.goBackToResults());
        }
        
        // Start training
        if (this.elements.startTrainingBtn) {
            this.elements.startTrainingBtn.addEventListener('click', () => this.startTraining());
        }
        
        // Pause training
        if (this.elements.pauseTrainingBtn) {
            this.elements.pauseTrainingBtn.addEventListener('click', () => this.pauseTraining());
        }
        
        // Download model
        if (this.elements.downloadBtn) {
            this.elements.downloadBtn.addEventListener('click', () => this.downloadModel());
        }
        
        // Generate code after training
        if (this.elements.generateCodeBtn) {
            this.elements.generateCodeBtn.addEventListener('click', () => this.generateCode());
        }
        
        // Copy code button
        if (this.elements.copyCodeBtn) {
            this.elements.copyCodeBtn.addEventListener('click', () => this.copyGeneratedCode());
        }
        
        // Expand metrics button
        const expandMetricsBtn = document.getElementById('expandMetricsBtn');
        if (expandMetricsBtn) {
            expandMetricsBtn.addEventListener('click', () => this.toggleExpandedMetrics());
        }
    }
    
    initFileUpload() {
        const fileInput = this.elements.datasetFile;
        const fileDropzone = this.elements.fileDropzone;
        
        if (fileInput && fileDropzone) {
            // Handle file selection via input
            fileInput.addEventListener('change', (event) => this.handleFileUpload(event));
            
            // Handle file drag and drop
            fileDropzone.addEventListener('dragover', (e) => {
                e.preventDefault();
                fileDropzone.classList.add('file-active');
            });
            
            fileDropzone.addEventListener('dragleave', () => {
                fileDropzone.classList.remove('file-active');
            });
            
            fileDropzone.addEventListener('drop', (e) => {
                e.preventDefault();
                fileDropzone.classList.remove('file-active');
                
                if (e.dataTransfer.files.length) {
                    fileInput.files = e.dataTransfer.files;
                    this.handleFileUpload({ target: { files: e.dataTransfer.files } });
                }
            });
        }
    }
    
    handleFileUpload(event) {
        console.log("handleFileUpload called", event);
        const file = event.target.files[0];
        if (!file) {
            console.log("No file selected in handleFileUpload");
            return;
        }
        
        console.log("File selected:", file.name, "Size:", file.size, "Type:", file.type);
        
        // Check if file is a CSV
        if (file.type !== 'text/csv' && !file.name.toLowerCase().endsWith('.csv')) {
            this.showError('Please upload a CSV file.');
            return;
        }
        
        // Update UI with file name
        if (this.elements.fileName) {
            this.elements.fileName.textContent = file.name;
        }
        
        // Store file reference
        this.state.currentFile = file;
        
        // Read file to populate target column dropdown
        this.showLoading('Reading file...');
        
        const reader = new FileReader();
        reader.onload = (e) => {
            console.log("File read completed");
            try {
                // Handle different line endings (CRLF, LF, CR)
                const content = e.target.result;
                const lines = content.split(/\r\n|\n|\r/).filter(line => line.trim().length > 0);
                
                if (lines.length === 0) {
                    throw new Error('The file appears to be empty.');
                }
                
                console.log("First line:", lines[0]);
                
                // Try to detect delimiter (comma, semicolon, tab)
                let delimiter = ',';
                const firstLine = lines[0];
                const commaCount = (firstLine.match(/,/g) || []).length;
                const semicolonCount = (firstLine.match(/;/g) || []).length;
                const tabCount = (firstLine.match(/\t/g) || []).length;
                
                if (semicolonCount > commaCount && semicolonCount > tabCount) {
                    delimiter = ';';
                } else if (tabCount > commaCount && tabCount > semicolonCount) {
                    delimiter = '\t';
                }
                
                console.log(`Detected delimiter: "${delimiter}"`);
                
                // Parse headers (support quoted fields)
                const headers = this.parseCSVLine(firstLine, delimiter);
                if (headers.length === 0) {
                    throw new Error('Could not parse headers from the CSV file.');
                }
                
                console.log("Detected columns:", headers);
                
                // Populate the target column dropdown
                const targetColumnSelect = this.elements.targetColumn;
                if (targetColumnSelect) {
                    targetColumnSelect.innerHTML = '<option value="">Select target variable</option>';
                    
                    headers.forEach(col => {
                        if (col && col.trim()) {  // Only add non-empty columns
                            const option = document.createElement('option');
                            option.value = col.trim();
                            option.textContent = col.trim();
                            targetColumnSelect.appendChild(option);
                        }
                    });
                    console.log("Target column dropdown populated with", headers.length, "options");
                } else {
                    console.log("Target column select element not found");
                }
                
                // Store the detected delimiter for later use
                this.state.csvDelimiter = delimiter;
            } catch (error) {
                console.error("Error parsing CSV:", error);
                this.showError(`Error parsing CSV file: ${error.message}`);
            } finally {
                this.hideLoading();
            }
        };
        
        reader.onerror = (error) => {
            console.error("Error reading file:", error);
            this.showError('There was an error reading the file.');
            this.hideLoading();
        };
        
        console.log("Starting file read...");
        reader.readAsText(file);
    }
    
    // Helper method to parse a CSV line handling quoted fields
    parseCSVLine(line, delimiter = ',') {
        const result = [];
        let current = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            
            if (char === '"') {
                // Toggle quote state
                inQuotes = !inQuotes;
            } else if (char === delimiter && !inQuotes) {
                // End of field
                result.push(current);
                current = '';
            } else {
                current += char;
            }
        }
        
        // Add the last field
        result.push(current);
        
        return result.map(field => {
            // Remove surrounding quotes if present
            if (field.startsWith('"') && field.endsWith('"')) {
                return field.slice(1, -1);
            }
            return field;
        });
    }
    
    analyzeDataset() {
        console.log("analyzeDataset function called");
        
        // Validate inputs
        if (!this.state.currentFile) {
            console.log("No file selected");
            this.showError('Please select a CSV file first.');
            return;
        }
        
        if (!this.elements.targetColumn.value) {
            console.log("No target column selected");
            this.showError('Please select a target column.');
            return;
        }
        
        this.state.targetColumn = this.elements.targetColumn.value;
        console.log("Target column:", this.state.targetColumn);
        
        // Prepare form data
        const formData = new FormData();
        formData.append('file', this.state.currentFile);
        formData.append('target_column', this.state.targetColumn);
        
        // Get selected dataset type (classification or regression)
        const datasetType = document.querySelector('input[name="datasetType"]:checked').value;
        formData.append('dataset_type', datasetType);
        console.log("Selected dataset type:", datasetType);
        
        // Add detected delimiter if available
        if (this.state.csvDelimiter) {
            formData.append('delimiter', this.state.csvDelimiter);
            console.log("Using detected delimiter:", this.state.csvDelimiter);
        }
        
        console.log("Sending request to /upload_dataset");
        
        // Send request to server
        this.showLoading('Analyzing dataset...');
        
        fetch('/upload_dataset', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log("Received response:", response);
            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.status} ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Received data:", data);
            
            if (!data || data.error) {
                throw new Error(data.error || 'Invalid response from server');
            }
            
            // Store dataset info and architecture
            this.state.datasetInfo = data.dataset_info;
            this.state.architecture = data.architecture;
            
            // Update network visualization
            if (window.neuralVis) {
                // Initialize with suggested architecture
                neuralVis.updateNetworkStructure({
                    inputShape: this.state.datasetInfo.n_features,
                    layers: this.state.architecture.layers
                });
            } else {
                // Create new visualization
                initNeuralNetworkVis('network-vis', {
                    width: document.getElementById('network-vis').clientWidth || 800,
                    height: 380, // Fixed height that fits well in the card
                    layerSpacing: 180, // Increase spacing between layers
                    onLayerAdd: (layer) => {
                        console.log('Layer added', layer);
                        this.updateArchitectureFromVisualization();
                    },
                    onLayerRemove: (layer) => {
                        console.log('Layer removed', layer);
                        this.updateArchitectureFromVisualization();
                    },
                    onNeuronAdd: (neuron) => {
                        console.log('Neuron added', neuron);
                        this.updateArchitectureFromVisualization();
                    },
                    onNeuronRemove: (neuron) => {
                        console.log('Neuron removed', neuron);
                        this.updateArchitectureFromVisualization();
                    },
                    onActivationChange: (layerId, activation) => {
                        console.log('Activation changed', layerId, activation);
                        this.updateArchitectureFromVisualization();
                    },
                    onRegularizationChange: (layerId, regularization) => {
                        console.log('Regularization changed', layerId, regularization);
                        this.updateArchitectureFromVisualization();
                    },
                    onLayerTypeChange: (layerId, layerType) => {
                        console.log('Layer type changed', layerId, layerType);
                        this.updateArchitectureFromVisualization();
                    }
                });
                
                // Update with suggested architecture
                neuralVis.updateNetworkStructure({
                    inputShape: this.state.datasetInfo.n_features,
                    layers: this.state.architecture.layers
                });
            }
            
            // Update dataset stats display
            this.updateDatasetStats();
            
            // Move to next step
            console.log("Going to step 2");
            this.goToStep(2);
            
            // Ensure architecture section is visible
            if (this.sections.architectureSection) {
                this.sections.architectureSection.style.display = 'block';
                console.log("Set architecture section display to block");
            } else {
                console.error("Architecture section element not found");
            }
        })
        .catch(error => {
            console.error('Error during dataset analysis:', error);
            this.showError(`Error analyzing dataset: ${error.message}`);
        })
        .finally(() => {
            this.hideLoading();
        });
    }
    
    updateArchitectureFromVisualization() {
        if (window.neuralVis) {
            try {
                // Get architecture from visualization including activation functions and regularization
                const architecture = neuralVis.getNetworkArchitecture();
                
                // Log the architecture from visualization for debugging
                console.log("Network architecture from visualization:", architecture);
                
                if (!architecture || !architecture.layers) {
                    console.error("Invalid architecture from visualization");
                    return;
                }
                
                // Deep clone the architecture to avoid reference issues
                const architectureClone = JSON.parse(JSON.stringify(architecture));
                
                // Create a new object with visualization's exact architecture
                this.state.architecture = {
                    input_shape: architectureClone.inputShape,
                    layers: architectureClone.layers.map(layer => {
                        // Convert from visualization format to server format if needed
                        const baseLayer = {
                            type: layer.type || 'Dense', // Default to Dense if not specified
                            regularization: layer.regularization || undefined
                        };
                        
                        // Add type-specific properties
                        if (layer.type === 'Dropout') {
                            return {
                                ...baseLayer,
                                rate: layer.rate || 0.2
                            };
                        } else if (layer.type === 'BatchNormalization') {
                            return baseLayer;
                        } else {
                            // For Dense, LSTM, GRU, etc. that need units and activation
                            return {
                                ...baseLayer,
                                units: layer.units,
                                activation: layer.activation
                            };
                        }
                    })
                };
                
                // Verify each layer has the required properties
                this.state.architecture.layers.forEach((layer, index) => {
                    // Make sure units is a number
                    if (typeof layer.units !== 'number' || isNaN(layer.units)) {
                        console.warn(`Layer ${index} has invalid units, setting default`);
                        layer.units = index === this.state.architecture.layers.length - 1 ? 1 : 32;
                    }
                    
                    // Verify activation function
                    if (!layer.activation) {
                        console.warn(`Layer ${index} missing activation, setting default`);
                        // Default activations based on layer position
                        if (index === this.state.architecture.layers.length - 1) {
                            // Output layer
                            layer.activation = this.isProbablyClassification() ? 'softmax' : 'linear';
                        } else {
                            // Hidden layer
                            layer.activation = 'relu';
                        }
                    }
                });
                
                console.log("Final architecture for server:", this.state.architecture);
            } catch (error) {
                console.error("Error updating architecture from visualization:", error);
            }
        } else {
            console.error("Neural network visualization is not initialized");
        }
    }
    
    // Helper to determine if this is likely a classification task
    isProbablyClassification() {
        return this.state.datasetInfo && 
               this.state.datasetInfo.target_type && 
               this.state.datasetInfo.target_type.toLowerCase().includes('classif');
    }
    
    updateDatasetStats() {
        if (!this.state.datasetInfo) return;
        
        const statSamples = document.getElementById('statSamples');
        const statFeatures = document.getElementById('statFeatures');
        const statTaskType = document.getElementById('statTaskType');
        
        if (statSamples) statSamples.textContent = this.state.datasetInfo.n_samples;
        if (statFeatures) statFeatures.textContent = this.state.datasetInfo.n_features;
        if (statTaskType) {
            const taskType = this.state.datasetInfo.target_type;
            statTaskType.textContent = taskType.charAt(0).toUpperCase() + taskType.slice(1);
        }
    }
    
    buildModel() {
        return new Promise((resolve, reject) => {
            if (!this.state.architecture) {
                const errorMsg = 'Architecture not defined. Please analyze a dataset first.';
                this.showError(errorMsg);
                reject(new Error(errorMsg));
                return;
            }
            
            try {
                // Get the architecture directly from the visualization
                const visArchitecture = neuralVis.getNetworkArchitecture();
                
                // Get training parameters
                const optimizer = this.elements.optimizer?.value || 'adam';
                const learningRate = parseFloat(this.elements.learningRate?.value || '0.001');
                
                // Prepare request with the exact architecture from the visualization
                const modelConfig = {
                    architecture: {
                        input_shape: visArchitecture.inputShape,
                        layers: visArchitecture.layers.map(layer => ({
                            type: layer.type || 'Dense', // Use the layer type from visualization
                            units: layer.units,
                            activation: layer.activation,
                            regularization: layer.regularization,
                            // Add dropout rate for Dropout layers
                            ...(layer.type === 'Dropout' && { rate: layer.rate || 0.2 })
                        }))
                    },
                    // Include dataset type from state
                    dataset_type: this.state.datasetInfo?.target_type || 'classification',
                    optimizer: optimizer,
                    learning_rate: learningRate
                };
                
                console.log("Sending validated model config to server:", modelConfig);
                
                // Send request to server
                this.showLoading('Building model...');
                
                fetch('/build_model', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(modelConfig)
                })
                .then(response => {
                    console.log("Build model response status:", response.status);
                    if (!response.ok) {
                        return response.text().then(text => {
                            throw new Error(`Network response was not ok: ${response.status} ${response.statusText}. Details: ${text}`);
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    console.log("Build model response data:", data);
                    // Store model summary
                    this.state.modelSummary = data.model_summary;
                    
                    // Update model summary display
                    const modelSummary = document.getElementById('modelSummary');
                    if (modelSummary) {
                        modelSummary.textContent = this.state.modelSummary;
                        
                        // Highlight code if highlight.js is available
                        if (window.hljs) {
                            hljs.highlightBlock(modelSummary);
                        }
                    }
                    
                    // Move to next step
                    this.goToStep(3);
                    
                    // Resolve the promise with the data
                    resolve(data);
                })
                .catch(error => {
                    console.error('Error building model:', error);
                    this.showError(`Error building model: ${error.message}`);
                    reject(error);
                })
                .finally(() => {
                    this.hideLoading();
                });
            } catch (error) {
                console.error('Error building model:', error);
                this.showError(`Error building model: ${error.message}`);
                reject(error);
            }
        });
    }
    
    async startTraining() {
        try {
            this.showLoading('Initializing training...');
            
            // Reset training metrics storage
            this.trainingMetrics = {
                epochs: [],
                accuracy: [],
                loss: [],
                val_accuracy: [],
                val_loss: []
            };
            
            // Initialize training monitor if not already initialized
            if (!window.trainingMonitor) {
                window.trainingMonitor = initTrainingMonitor({
                    onEpochEnd: (data) => {
                        console.log('Epoch end callback:', data);
                        // Store metrics for later display
                        this.trainingMetrics.epochs.push(data.epoch);
                        this.trainingMetrics.accuracy.push(data.accuracy);
                        this.trainingMetrics.loss.push(data.loss);
                        this.trainingMetrics.val_accuracy.push(data.val_accuracy);
                        this.trainingMetrics.val_loss.push(data.val_loss);
                    }
                });
            }
            
            // Check if infinite epochs mode is enabled
            const infiniteEpochsEnabled = document.getElementById('infiniteEpochs')?.checked || false;
            const epochs = infiniteEpochsEnabled ? 9999 : parseInt(this.elements.epochs?.value || '10');
            const batchSize = parseInt(this.elements.batchSize?.value || '32');
            const trainTestSplit = parseInt(this.elements.trainTestSplit?.value || '80') / 100; // Convert percentage to decimal
            
            // Create training config
            const trainingConfig = {
                optimizer: this.elements.optimizer.value,
                learning_rate: parseFloat(this.elements.learningRate.value),
                batch_size: batchSize,
                epochs: epochs,
                test_size: 1 - trainTestSplit, // Convert train percentage to test percentage (1 - train)
                infinite_training: infiniteEpochsEnabled
            };
            
            // Send training request to server
            const response = await axios.post('/train', trainingConfig);
            
            if (response.data.success) {
                // Hide loading overlay immediately so charts are visible during training
                this.hideLoading();
                
                // Initialize training monitor
                window.trainingMonitor.resetCharts();
                window.trainingMonitor.startTraining(epochs);
                
                // Update status text to indicate training has started
                const trainingStatusText = document.getElementById('training-status-text');
                if (trainingStatusText) {
                    trainingStatusText.textContent = infiniteEpochsEnabled ? 
                        'Training in infinite mode. Use pause button when satisfied.' : 
                        'Training started. Watch progress in charts below...';
                }
                
                // Show pause button and hide start button during training
                if (this.elements.startTrainingBtn) this.elements.startTrainingBtn.style.display = 'none';
                if (this.elements.pauseTrainingBtn) this.elements.pauseTrainingBtn.style.display = 'block';
                
                // Set training state
                this.state.isTraining = true;
                this.state.isPaused = false;
                
                // Start polling for training updates
                this.trainingInterval = setInterval(() => this.checkTrainingStatus(), 1000);
            } else {
                this.hideLoading();
                this.showError(response.data.error || 'Failed to start training.');
            }
        } catch (error) {
            this.hideLoading();
            this.showError('Error starting training: ' + error.message);
        }
    }
    
    async pauseTraining() {
        try {
            const trainingStatusText = document.getElementById('training-status-text');
            
            // Send pause/resume request to server
            const response = await axios.post('/pause_training');
            
            if (response.data.success) {
                // Update UI based on paused state
                this.state.isPaused = response.data.paused;
                
                if (this.elements.pauseTrainingBtn) {
                    // Update button text and icon based on state
                    if (this.state.isPaused) {
                        this.elements.pauseTrainingBtn.innerHTML = '<i class="fas fa-play me-2"></i>Resume Training';
                        if (trainingStatusText) {
                            trainingStatusText.textContent = 'Training paused. Click Resume to continue.';
                        }
                    } else {
                        this.elements.pauseTrainingBtn.innerHTML = '<i class="fas fa-pause me-2"></i>Pause Training';
                        if (trainingStatusText) {
                            trainingStatusText.textContent = 'Training resumed. Watch progress in charts below...';
                        }
                    }
                }
            } else {
                this.showError(response.data.error || 'Failed to pause/resume training.');
            }
        } catch (error) {
            this.showError('Error pausing/resuming training: ' + error.message);
        }
    }
    
    // Display all training metrics in the results section
    displayAllMetrics() {
        // Check if we have training metrics to display
        if (!this.trainingMetrics || this.trainingMetrics.epochs.length === 0) {
            console.log('No training metrics to display');
            return;
        }
        
        // Get the metrics container
        const metricsContainer = document.getElementById('all-metrics-container');
        if (!metricsContainer) {
            console.error('Metrics container not found');
            // Create the container if it doesn't exist
            const resultsSection = document.querySelector('#trainingResultsSection .card-body');
            if (resultsSection) {
                const metricsSection = document.createElement('div');
                metricsSection.className = 'mt-4';
                metricsSection.innerHTML = `
                    <h6 class="mb-3"><i class="fas fa-chart-line me-2"></i>Training Metrics</h6>
                    <div id="all-metrics-container"></div>
                    <button id="expandMetricsBtn" class="btn btn-sm btn-outline-primary mt-2" style="display: none;">
                        Show All Epochs
                    </button>
                `;
                resultsSection.appendChild(metricsSection);
                
                // Add event listener to the expand button
                const expandBtn = metricsSection.querySelector('#expandMetricsBtn');
                if (expandBtn) {
                    expandBtn.addEventListener('click', () => this.toggleExpandedMetrics());
                }
                
                // Now get the container we just created
                return this.displayAllMetrics();
            }
            return;
        }
        
        // Clear previous content
        metricsContainer.innerHTML = '';
        
        // Create a summary table for the metrics
        const table = document.createElement('table');
        table.className = 'table table-sm table-hover metrics-table';
        
        // Create table header
        const thead = document.createElement('thead');
        thead.innerHTML = `
            <tr>
                <th>Epoch</th>
                <th>Training Loss</th>
                <th>Training Accuracy</th>
                <th>Validation Loss</th>
                <th>Validation Accuracy</th>
            </tr>
        `;
        table.appendChild(thead);
        
        // Create table body
        const tbody = document.createElement('tbody');
        
        // Add rows for each epoch (limited to first 5 in collapsed view)
        const displayCount = this.expandedMetricsView ? this.trainingMetrics.epochs.length : Math.min(5, this.trainingMetrics.epochs.length);
        
        for (let i = 0; i < displayCount; i++) {
            const row = document.createElement('tr');
            
            // Format values to 4 decimal places
            const accuracy = this.trainingMetrics.accuracy[i] * 100;
            const valAccuracy = this.trainingMetrics.val_accuracy[i] * 100;
            
            row.innerHTML = `
                <td>${this.trainingMetrics.epochs[i] + 1}</td>
                <td>${this.trainingMetrics.loss[i].toFixed(4)}</td>
                <td>${accuracy.toFixed(2)}%</td>
                <td>${this.trainingMetrics.val_loss[i].toFixed(4)}</td>
                <td>${valAccuracy.toFixed(2)}%</td>
            `;
            
            tbody.appendChild(row);
        }
        
        table.appendChild(tbody);
        metricsContainer.appendChild(table);
        
        // Show/hide expand button based on number of epochs
        const expandBtn = document.getElementById('expandMetricsBtn');
        if (expandBtn) {
            if (this.trainingMetrics.epochs.length > 5) {
                expandBtn.style.display = 'block';
                expandBtn.textContent = this.expandedMetricsView ? 'Show Less' : 'Show All Epochs';
            } else {
                expandBtn.style.display = 'none';
            }
        }
    }
    
    // Toggle between expanded and collapsed metrics view
    toggleExpandedMetrics() {
        this.expandedMetricsView = !this.expandedMetricsView;
        this.displayAllMetrics();
    }
    
    updateTestMetrics() {
        if (!this.state.trainingResults) return;
        
        const testAccuracy = document.getElementById('testAccuracy');
        const testLoss = document.getElementById('testLoss');
        
        if (testAccuracy && this.state.trainingResults.testAccuracy !== null) {
            // Fix accuracy value - ensure it's a value between 0-1 before converting to percentage
            const accuracy = parseFloat(this.state.trainingResults.testAccuracy);
            
            // For regression tasks, show both transformed accuracy and raw error
            if (this.state.trainingResults.isRegression) {
                // Display the transformed accuracy value (0-1 scale where 1 is best)
                const displayValue = (accuracy * 100).toFixed(2);
                testAccuracy.textContent = displayValue + '%';
                
                // Change label to indicate this is a model quality score
                const label = testAccuracy.closest('.metric-card')?.querySelector('.metric-label');
                if (label) {
                    label.textContent = 'Model Quality';
                }
                
                // Add raw MAE display if available
                if (this.state.trainingResults.test_mae_raw !== undefined) {
                    const rawMae = parseFloat(this.state.trainingResults.test_mae_raw);
                    // Create or update MAE display
                    let maeElement = document.getElementById('testMae');
                    if (!maeElement) {
                        // Create a new metric card for MAE
                        const metricsContainer = testAccuracy.closest('.metrics-container');
                        if (metricsContainer) {
                            const maeCard = document.createElement('div');
                            maeCard.className = 'metric-card';
                            maeCard.innerHTML = `
                                <div class="metric-value" id="testMae">${rawMae.toFixed(4)}</div>
                                <div class="metric-label">Mean Absolute Error</div>
                                <div class="metric-description">Lower values indicate better predictions</div>
                            `;
                            metricsContainer.appendChild(maeCard);
                        }
                    } else {
                        maeElement.textContent = rawMae.toFixed(4);
                    }
                }
            } else {
                // For classification, show as percentage
                const displayValue = accuracy > 1 ? accuracy.toFixed(2) : (accuracy * 100).toFixed(2);
                testAccuracy.textContent = displayValue + '%';
            }
        }
        
        if (testLoss && this.state.trainingResults.testLoss !== null) {
            testLoss.textContent = this.state.trainingResults.testLoss.toFixed(4);
        }
    }
    
    downloadModel() {
        this.showLoading('Preparing download...');
        
        fetch('/download_model', {
            method: 'GET',
            headers: {
                'Accept': 'application/zip',
            },
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.blob();
        })
        .then(blob => {
            // Create download link
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'neural_network_model.zip');
            document.body.appendChild(link);
            link.click();
            link.remove();
            
            // Clean up
            window.URL.revokeObjectURL(url);
        })
        .catch(error => {
            console.error('Error downloading model:', error);
            this.showError(`Error downloading model: ${error.message}`);
        })
        .finally(() => {
            this.hideLoading();
        });
    }
    
    generateCode() {
        if (this.state.codeGenerated && this.state.generatedCode) {
            // Code already generated, just show it
            this.updateCodeDisplay();
            return;
        }
        
        this.showLoading('Generating code...');
        this.generateCodeAndShow();
    }
    
    updateCodeDisplay() {
        const codeContent = document.getElementById('codeContent');
        if (codeContent && this.state.generatedCode) {
            codeContent.textContent = this.state.generatedCode;
            
            // Highlight code if highlight.js is available
            if (window.hljs) {
                hljs.highlightBlock(codeContent);
            }
        }
    }
    
    initializePrediction() {
        if (!this.state.datasetInfo) return;
        
        // Initialize prediction interface if not already
        if (!window.predictionInterface) {
            initPredictionInterface({
                containerId: 'prediction-container',
                inputsContainerId: 'prediction-inputs',
                resultsContainerId: 'prediction-results',
                runButtonId: 'run-prediction-btn'
            });
        }
        
        // Generate feature config from dataset info
        const features = [];
        const n_features = this.state.datasetInfo.n_features;
        for (let i = 0; i < n_features; i++) {
            features.push({
                name: `feature${i+1}`,
                label: `Feature ${i+1}`,
                type: 'number',
                defaultValue: 0
            });
        }
        
        // Set features
        window.predictionInterface.setFeatures(features);
    }
    
    goToStep(stepNumber) {
        // Update state
        this.state.currentStep = stepNumber;
        
        // Update UI
        this.updateStepIndicators();
        this.showSection(stepNumber);
    }
    
    updateStepIndicators() {
        const activeStep = this.state.currentStep;
        
        // Reset all steps
        Object.values(this.steps).forEach(step => {
            if (step) {
                step.classList.remove('active');
                step.classList.remove('completed');
            }
        });
        
        // Set active step
        const currentStepEl = this.steps[`step${activeStep}`];
        if (currentStepEl) {
            currentStepEl.classList.add('active');
        }
        
        // Set previous steps as completed
        for (let i = 1; i < activeStep; i++) {
            const stepEl = this.steps[`step${i}`];
            if (stepEl) {
                stepEl.classList.add('completed');
            }
        }
    }
    
    showSection(stepNumber) {
        console.log(`Showing section for step ${stepNumber}`);
        
        // Hide all sections
        Object.values(this.sections).forEach(section => {
            if (section) {
                section.style.display = 'none';
                console.log(`Hidden section: ${section.id}`);
            } else {
                console.warn("Found a null section in sections object");
            }
        });
        
        // Show the appropriate section
        switch (stepNumber) {
            case 1:
                if (this.sections.datasetSection) {
                    this.sections.datasetSection.style.display = 'block';
                    console.log("Showing datasetSection");
                } else {
                    console.error("datasetSection element not found");
                }
                break;
            case 2:
                if (this.sections.architectureSection) {
                    this.sections.architectureSection.style.display = 'block';
                    console.log("Showing architectureSection");
                } else {
                    console.error("architectureSection element not found");
                }
                break;
            case 3:
                if (this.sections.modelSummarySection) {
                    this.sections.modelSummarySection.style.display = 'block';
                    console.log("Showing modelSummarySection");
                } else {
                    console.error("modelSummarySection element not found");
                }
                break;
            case 4:
                if (this.sections.trainingResultsSection) {
                    this.sections.trainingResultsSection.style.display = 'block';
                    console.log("Showing trainingResultsSection");
                } else {
                    console.error("trainingResultsSection element not found");
                }
                break;
            case 5:
                if (this.sections.codeSection) {
                    this.sections.codeSection.style.display = 'block';
                    console.log("Showing codeSection");
                } else {
                    console.error("codeSection element not found");
                }
                break;
            default:
                console.error(`Invalid step number: ${stepNumber}`);
        }
    }
    
    showLoading(text = 'Processing...') {
        const overlay = this.elements.loadingOverlay;
        const textElement = this.elements.loadingText;
        
        if (textElement) {
            textElement.textContent = text;
        }
        
        if (overlay) {
            overlay.style.display = 'flex';
            overlay.classList.add('active');
        }
    }
    
    hideLoading() {
        const overlay = this.elements.loadingOverlay;
        
        if (overlay) {
            overlay.classList.remove('active');
            setTimeout(() => {
                overlay.style.display = 'none';
            }, 300);
        }
    }
    
    showError(message) {
        console.error("Error:", message);
        
        // Create toast notification
        const toast = document.createElement('div');
        toast.className = 'toast error';
        toast.style.position = 'fixed';
        toast.style.top = '20px';
        toast.style.right = '20px';
        toast.style.background = '#ff3d00';
        toast.style.color = 'white';
        toast.style.padding = '15px';
        toast.style.borderRadius = '5px';
        toast.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
        toast.style.zIndex = '9999';
        toast.style.minWidth = '300px';
        
        toast.innerHTML = `
            <div class="toast-icon">
                <i class="fas fa-exclamation-circle"></i>
            </div>
            <div class="toast-content">
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close" style="background: none; border: none; color: white; cursor: pointer; position: absolute; top: 10px; right: 10px;">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // Add to notifications container or create one
        let notificationsContainer = document.querySelector('.notifications-container');
        if (!notificationsContainer) {
            notificationsContainer = document.createElement('div');
            notificationsContainer.className = 'notifications-container';
            document.body.appendChild(notificationsContainer);
        }
        
        notificationsContainer.appendChild(toast);
        
        // Add close button functionality
        const closeButton = toast.querySelector('.toast-close');
        if (closeButton) {
            closeButton.addEventListener('click', () => {
                toast.classList.add('toast-hidden');
                setTimeout(() => toast.remove(), 300);
            });
        }
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            toast.classList.add('toast-hidden');
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }
    
    checkTrainingStatus() {
        const trainingStatusText = document.getElementById('training-status-text');
        
        if (!this.eventSource) {
            console.log('Setting up SSE connection to /train_stream');
            // Setup Server-Sent Events
            this.eventSource = new EventSource('/train_stream');
            
            this.eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log('Received SSE message:', data);
                
                if (data.status === 'epoch') {
                    // Update training monitor with epoch data
                    if (window.trainingMonitor) {
                        window.trainingMonitor.updateEpoch(data);
                    } else {
                        console.error('TrainingMonitor not initialized');
                    }
                    
                    // Update status text without showing loading overlay
                    if (trainingStatusText) {
                        // Check if infinite epochs mode is enabled
                        const infiniteEpochsEnabled = document.getElementById('infiniteEpochs')?.checked || false;
                        if (infiniteEpochsEnabled) {
                            trainingStatusText.textContent = `Training in progress... Epoch ${data.epoch + 1} (Infinite mode)`;
                        } else {
                            trainingStatusText.textContent = `Training in progress... Epoch ${data.epoch + 1}/${this.elements.epochs.value}`;
                        }
                    }
                    
                    // Make sure loading overlay is hidden during training
                    this.hideLoading();
                }
                else if (data.status === 'completed') {
                    console.log('Training completed:', data);
                    // Training is complete
                    this.hideLoading();
                    if (trainingStatusText) {
                        trainingStatusText.textContent = 'Training completed!';
                    }
                    
                    // Reset UI elements
                    if (this.elements.startTrainingBtn) this.elements.startTrainingBtn.style.display = 'block';
                    if (this.elements.pauseTrainingBtn) {
                        this.elements.pauseTrainingBtn.style.display = 'none';
                        // Reset pause button text
                        this.elements.pauseTrainingBtn.innerHTML = '<i class="fas fa-pause me-2"></i>Pause Training';
                    }
                    
                    // Clear the interval
                    clearInterval(this.trainingInterval);
                    this.trainingInterval = null;
                    
                    // Close the SSE connection
                    if (this.eventSource) {
                        this.eventSource.close();
                        this.eventSource = null;
                    }
                    
                    // Update test metrics
                    const testAccuracy = document.getElementById('testAccuracy');
                    const testLoss = document.getElementById('testLoss');
                    
                    if (testAccuracy) {
                        // Fix accuracy value - ensure it's a value between 0-1 before converting to percentage
                        const accuracy = parseFloat(data.test_accuracy);
                        
                        // For regression tasks, show the raw error value
                        if (data.is_regression === true) {
                            testAccuracy.textContent = accuracy.toFixed(4);
                            // Change label if possible
                            const label = testAccuracy.closest('.metric-card')?.querySelector('.metric-label');
                            if (label) {
                                label.textContent = 'Test Error';
                            }
                        } else {
                            // For classification, show as percentage
                            testAccuracy.textContent = (accuracy * 100).toFixed(2) + '%';
                        }
                    }
                    
                    if (testLoss) {
                        testLoss.textContent = data.test_loss.toFixed(4);
                    }
                    
                    // Store the results
                    this.state.trainingResults = {
                        testAccuracy: data.test_accuracy,
                        testLoss: data.test_loss,
                        isRegression: data.is_regression,
                        test_mae_raw: data.test_mae_raw, // Store the raw MAE value for regression tasks
                        trainingMetrics: this.trainingMetrics // Store all training metrics
                    };
                    
                    // Display all metrics in the results section
                    this.displayAllMetrics();
                    
                    // Reset training state
                    this.state.isTraining = false;
                    this.state.isPaused = false;
                    
                    // Move to results step
                    this.goToStep(4);
                    
                    // Initialize prediction
                    this.initializePrediction();
                }
                else if (data.status === 'error') {
                    console.error('Training error:', data.message);
                    // Error occurred during training
                    this.hideLoading();
                    this.showError(`Error during training: ${data.message}`);
                    
                    // Reset UI elements
                    if (this.elements.startTrainingBtn) this.elements.startTrainingBtn.style.display = 'block';
                    if (this.elements.pauseTrainingBtn) this.elements.pauseTrainingBtn.style.display = 'none';
                    
                    // Reset training state
                    this.state.isTraining = false;
                    this.state.isPaused = false;
                    
                    // Clear the interval
                    clearInterval(this.trainingInterval);
                    this.trainingInterval = null;
                    
                    // Close the SSE connection
                    if (this.eventSource) {
                        this.eventSource.close();
                        this.eventSource = null;
                    }
                }
            };
            
            this.eventSource.onerror = (error) => {
                console.error('EventSource error:', error);
                this.hideLoading();
                this.showError('Connection to training stream lost.');
                
                // Reset UI elements
                if (this.elements.startTrainingBtn) this.elements.startTrainingBtn.style.display = 'block';
                if (this.elements.pauseTrainingBtn) this.elements.pauseTrainingBtn.style.display = 'none';
                
                // Reset training state
                this.state.isTraining = false;
                this.state.isPaused = false;
                
                // Clear the interval
                clearInterval(this.trainingInterval);
                this.trainingInterval = null;
                
                // Close the SSE connection
                if (this.eventSource) {
                    this.eventSource.close();
                    this.eventSource = null;
                }
            };
        }
    }
    
    generateCodeDirectly() {
        if (!this.state.architecture) {
            this.showError('Architecture not defined. Please analyze a dataset and build a model first.');
            return;
        }
        
        this.showLoading('Generating code...');
        
        // First build the model if not already built
        if (!this.state.modelSummary) {
            // Force update architecture from visualization
            this.updateArchitectureFromVisualization();
            
            this.showLoading('Building model for code generation...');
            this.buildModel()
                .then(() => {
                    console.log("Model built successfully, generating code...");
                    this.generateCodeAndShow();
                })
                .catch(error => {
                    console.log("Failed to build model, but still trying to generate code...");
                    // Even if model building fails, still try to generate code
                    // The server will create a temporary model
                    this.generateCodeAndShow();
                });
        } else {
            this.generateCodeAndShow();
        }
    }
    
    generateCodeAndShow() {
        console.log("Requesting code generation from server...");
        
        fetch('/generate_code', {
            method: 'GET'
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(errorData.error || 'Network response was not ok');
                }).catch(e => {
                    throw new Error(`Failed to generate code: ${response.status} ${response.statusText}`);
                });
            }
            return response.json();
        })
        .then(data => {
            console.log("Code generation successful");
            
            // Store generated code
            if (data.code) {
                this.state.generatedCode = data.code;
                this.state.codeGenerated = true;
                
                // Show code section
                this.sections.codeSection.style.display = 'block';
                
                // Make sure other sections don't interfere
                if (this.sections.trainingResultsSection) {
                    this.sections.trainingResultsSection.style.display = 'none';
                }
                
                // Update code display
                this.updateCodeDisplay();
                
                // Update step indicators
                this.goToStep(5);
            } else {
                throw new Error('No code was generated by the server');
            }
        })
        .catch(error => {
            console.error('Error generating code:', error);
            this.showError(`Error generating code: ${error.message}`);
        })
        .finally(() => {
            this.hideLoading();
        });
    }
    
    copyGeneratedCode() {
        if (!this.state.generatedCode) {
            this.showError('No code has been generated yet.');
            return;
        }
        
        const codeElement = document.getElementById('codeContent');
        if (!codeElement) return;
        
        // Create a temporary textarea element to copy from
        const textarea = document.createElement('textarea');
        textarea.value = this.state.generatedCode;
        textarea.setAttribute('readonly', '');
        textarea.style.position = 'absolute';
        textarea.style.left = '-9999px';
        document.body.appendChild(textarea);
        
        // Select and copy the text
        textarea.select();
        let success = false;
        try {
            success = document.execCommand('copy');
        } catch (err) {
            console.error('Failed to copy text:', err);
        }
        
        // Clean up
        document.body.removeChild(textarea);
        
        // Show feedback
        if (success) {
            // Change the copy button text temporarily
            const copyButton = this.elements.copyCodeBtn;
            if (copyButton) {
                const originalHtml = copyButton.innerHTML;
                copyButton.innerHTML = '<i class="fas fa-check me-2"></i>Copied!';
                setTimeout(() => {
                    copyButton.innerHTML = originalHtml;
                }, 2000);
            }
        } else {
            this.showError('Failed to copy code to clipboard.');
        }
    }
    
    // goBackToArchitecture function has been moved and enhanced below
    
    goBackToTraining() {
        // Implement the logic to go back to the training stage
        console.log("Going back to training stage");
        
        this.showLoading('Returning to training stage...');
        
        // Call the backend to reset training state if needed
        axios.post('/go_back')
            .then(response => {
                if (response.data.success) {
                    console.log('Successfully went back to training stage');
                    
                    // Reset any training-related state
                    if (this.trainingInterval) {
                        clearInterval(this.trainingInterval);
                        this.trainingInterval = null;
                    }
                    
                    // Reset training monitor if it exists
                    if (window.trainingMonitor) {
                        window.trainingMonitor.resetCharts();
                    }
                    
                    // Update status text
                    const trainingStatusText = document.getElementById('training-status-text');
                    if (trainingStatusText) {
                        trainingStatusText.textContent = 'Ready to train';
                    }
                    
                    // Navigate back to the training step
                    this.goToStep(3);
                } else {
                    this.showError(response.data.error || 'Failed to go back to training stage.');
                }
            })
            .catch(error => {
                console.error('Error going back to training stage:', error);
                this.showError(`Error going back: ${error.message}`);
            })
            .finally(() => {
                this.hideLoading();
            });
    }
    
    goBackToResults() {
        // Implement the logic to go back to the results stage
        console.log("Going back to results stage");
        
        this.showLoading('Returning to results stage...');
        
        // Hide code section
        this.hideAllSections();
        
        // Check if training has occurred or if we came directly from architecture
        if (this.state.trainingResults) {
            // If we have training results, go back to results section
            console.log("Going back to training results");
            this.sections.trainingResultsSection.style.display = 'block';
            this.updateActiveStep(4);
        } else {
            // If no training results, go back to architecture section
            console.log("Going back to architecture (no training results)");
            this.sections.architectureSection.style.display = 'block';
            this.updateActiveStep(2);
        }
        
        this.hideLoading();
    }
    
    // Modified goBackToArchitecture to work from code section as well
    goBackToArchitecture() {
        // Implement the logic to go back to the architecture stage
        console.log("Going back to architecture stage");
        
        this.showLoading('Returning to architecture stage...');
        
        // Check if we're in the code section
        if (this.sections.codeSection.style.display !== 'none') {
            // If we're in the code section, just go back to architecture directly
            this.hideAllSections();
            this.sections.architectureSection.style.display = 'block';
            this.updateActiveStep(2);
            this.hideLoading();
            return;
        }
        
        // Otherwise, proceed with the original implementation for going back from training
        // Call the backend to reset training state if needed
        axios.post('/go_back')
            .then(response => {
                if (response.data.success) {
                    console.log('Successfully went back to architecture stage');
                    
                    // Reset any training-related state
                    if (this.trainingInterval) {
                        clearInterval(this.trainingInterval);
                        this.trainingInterval = null;
                    }
                    
                    // Reset training monitor if it exists
                    if (window.trainingMonitor) {
                        window.trainingMonitor.resetCharts();
                    }
                    
                    // Update status text
                    const trainingStatusText = document.getElementById('training-status-text');
                    if (trainingStatusText) {
                        trainingStatusText.textContent = 'Ready to train';
                    }
                    
                    // Navigate back to the architecture step
                    this.goToStep(2);
                } else {
                    this.showError(response.data.error || 'Failed to go back to architecture stage.');
                }
            })
            .catch(error => {
                console.error('Error going back to architecture stage:', error);
                this.showError(`Error going back: ${error.message}`);
            })
            .finally(() => {
                this.hideLoading();
            });
    }
}

// Initialize the application when document is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new NeuralNetworkBuilder();
});
