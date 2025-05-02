// Training Monitor using Chart.js
class TrainingMonitor {
    constructor(config) {
        this.config = {
            lossChartId: 'loss-chart',
            accuracyChartId: 'accuracy-chart',
            metricsContainerId: 'training-metrics',
            statusElement: 'training-status-text',
            theme: 'dark',
            colors: {
                loss: '#FF6347',
                valLoss: '#FF7F50',
                accuracy: '#7B68EE',
                valAccuracy: '#9370DB'
            },
            onEpochEnd: null
        };
        
        // Override default config with provided values
        Object.assign(this.config, config);
        
        // Charts
        this.lossChart = null;
        this.accuracyChart = null;
        
        // Training data
        this.epochs = [];
        this.trainLoss = [];
        this.valLoss = [];
        this.trainAccuracy = [];
        this.valAccuracy = [];
        this.testLoss = null;
        this.testAccuracy = null;
        
        // Training status
        this.isTraining = false;
        this.currentEpoch = 0;
        this.totalEpochs = 0;
        this.eventSource = null;
        
        this.initialize();
    }
    
    initialize() {
        // Set up charts with dark theme
        this.setupCharts();
    }
    
    setupCharts() {
        // Chart.js global configuration for dark theme
        Chart.defaults.color = this.config.theme === 'dark' ? '#aaaaaa' : '#666666';
        Chart.defaults.borderColor = this.config.theme === 'dark' ? '#404040' : '#e2e2e2';
        
        // Loss chart
        const lossCtx = document.getElementById(this.config.lossChartId)?.getContext('2d');
        if (lossCtx) {
            this.lossChart = new Chart(lossCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Training Loss',
                            data: [],
                            borderColor: this.config.colors.loss,
                            backgroundColor: this.hexToRgba(this.config.colors.loss, 0.1),
                            borderWidth: 2,
                            pointRadius: 3,
                            pointHoverRadius: 5,
                            tension: 0.2,
                            fill: true
                        },
                        {
                            label: 'Validation Loss',
                            data: [],
                            borderColor: this.config.colors.valLoss,
                            backgroundColor: this.hexToRgba(this.config.colors.valLoss, 0.1),
                            borderWidth: 2,
                            pointRadius: 3,
                            pointHoverRadius: 5,
                            tension: 0.2,
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {
                        duration: 300
                    },
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                usePointStyle: true,
                                padding: 15
                            }
                        },
                        tooltip: {
                            backgroundColor: this.config.theme === 'dark' ? '#2d2d2d' : '#ffffff',
                            titleColor: this.config.theme === 'dark' ? '#e0e0e0' : '#333333',
                            bodyColor: this.config.theme === 'dark' ? '#aaaaaa' : '#666666',
                            borderColor: this.config.theme === 'dark' ? '#404040' : '#e2e2e2',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Epoch'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Loss'
                            }
                        }
                    }
                }
            });
        }
        
        // Accuracy chart
        const accuracyCtx = document.getElementById(this.config.accuracyChartId)?.getContext('2d');
        if (accuracyCtx) {
            this.accuracyChart = new Chart(accuracyCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Training Accuracy',
                            data: [],
                            borderColor: this.config.colors.accuracy,
                            backgroundColor: this.hexToRgba(this.config.colors.accuracy, 0.1),
                            borderWidth: 2,
                            pointRadius: 3,
                            pointHoverRadius: 5,
                            tension: 0.2,
                            fill: true
                        },
                        {
                            label: 'Validation Accuracy',
                            data: [],
                            borderColor: this.config.colors.valAccuracy,
                            backgroundColor: this.hexToRgba(this.config.colors.valAccuracy, 0.1),
                            borderWidth: 2,
                            pointRadius: 3,
                            pointHoverRadius: 5,
                            tension: 0.2,
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {
                        duration: 300
                    },
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                usePointStyle: true,
                                padding: 15
                            }
                        },
                        tooltip: {
                            backgroundColor: this.config.theme === 'dark' ? '#2d2d2d' : '#ffffff',
                            titleColor: this.config.theme === 'dark' ? '#e0e0e0' : '#333333',
                            bodyColor: this.config.theme === 'dark' ? '#aaaaaa' : '#666666',
                            borderColor: this.config.theme === 'dark' ? '#404040' : '#e2e2e2',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Epoch'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            max: 1,
                            title: {
                                display: true,
                                text: 'Accuracy'
                            },
                            ticks: {
                                callback: function(value) {
                                    return (value * 100).toFixed(0) + '%';
                                }
                            }
                        }
                    }
                }
            });
        }
    }
    
    clearData() {
        // Reset all data arrays
        this.epochs = [];
        this.trainLoss = [];
        this.valLoss = [];
        this.trainAccuracy = [];
        this.valAccuracy = [];
        this.testLoss = null;
        this.testAccuracy = null;
        this.currentEpoch = 0;
        
        // Clear charts
        if (this.lossChart) {
            this.lossChart.data.labels = [];
            this.lossChart.data.datasets[0].data = [];
            this.lossChart.data.datasets[1].data = [];
            this.lossChart.update();
        }
        
        if (this.accuracyChart) {
            this.accuracyChart.data.labels = [];
            this.accuracyChart.data.datasets[0].data = [];
            this.accuracyChart.data.datasets[1].data = [];
            this.accuracyChart.update();
        }
        
        // Clear metrics
        this.updateMetrics({
            currentEpoch: 0,
            totalEpochs: 0,
            trainLoss: 0,
            valLoss: 0,
            trainAccuracy: 0,
            valAccuracy: 0
        });
    }
    
    startTraining(totalEpochs) {
        if (this.isTraining) {
            this.stopTraining();
        }
        
        this.clearData();
        this.isTraining = true;
        this.totalEpochs = totalEpochs;
        
        // Update status text
        this.updateStatus('Training in progress...');
        
        // Set up event source for server-sent events
        this.setupEventSource();
    }
    
    stopTraining() {
        this.isTraining = false;
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        this.updateStatus('Training stopped');
    }
    
    setupEventSource() {
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        this.eventSource = new EventSource('/train_stream');
        
        this.eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.status === 'epoch') {
                this.updateEpoch(data);
            } else if (data.status === 'completed') {
                this.handleTrainingCompleted(data);
            } else if (data.status === 'error') {
                this.handleTrainingError(data);
            }
        };
        
        this.eventSource.onerror = (error) => {
            console.error('EventSource failed:', error);
            this.stopTraining();
            this.updateStatus('Error during training');
        };
    }
    
    updateEpoch(data) {
        if (!data) return;
        
        console.log("Training monitor received update:", data);
        
        // Update epoch counter
        this.currentEpoch = data.epoch;
        
        // Store data (ensure the values are numeric)
        this.epochs.push(data.epoch);
        this.trainLoss.push(parseFloat(data.loss));
        this.valLoss.push(parseFloat(data.val_loss));
        this.trainAccuracy.push(parseFloat(data.acc));
        this.valAccuracy.push(parseFloat(data.val_acc));
        
        console.log("Current accuracy arrays:", {
            trainAccuracy: this.trainAccuracy,
            valAccuracy: this.valAccuracy
        });
        
        // Update charts
        this.updateCharts();
        
        // Update metrics display
        this.updateMetrics({
            currentEpoch: this.currentEpoch + 1, // +1 for display (0-based to 1-based)
            totalEpochs: this.totalEpochs,
            trainLoss: data.loss,
            valLoss: data.val_loss,
            trainAccuracy: data.acc,
            valAccuracy: data.val_acc
        });
        
        // Call callback if provided
        if (this.config.onEpochEnd) {
            this.config.onEpochEnd(data);
        }
        
        return this;
    }
    
    handleTrainingCompleted(data) {
        this.isTraining = false;
        this.testLoss = data.test_loss;
        this.testAccuracy = data.test_accuracy;
        
        // Close event source
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        
        // Update status
        this.updateStatus('Training completed');
        
        // Update final metrics
        this.updateMetrics({
            currentEpoch: this.currentEpoch,
            totalEpochs: this.totalEpochs,
            trainLoss: this.trainLoss[this.trainLoss.length - 1],
            valLoss: this.valLoss[this.valLoss.length - 1],
            trainAccuracy: this.trainAccuracy[this.trainAccuracy.length - 1],
            valAccuracy: this.valAccuracy[this.valAccuracy.length - 1],
            testLoss: this.testLoss,
            testAccuracy: this.testAccuracy
        });
    }
    
    handleTrainingError(data) {
        this.isTraining = false;
        
        // Close event source
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        
        // Update status
        this.updateStatus(`Training error: ${data.message}`);
    }
    
    updateCharts() {
        // Update loss chart
        if (this.lossChart) {
            this.lossChart.data.labels = this.epochs;
            this.lossChart.data.datasets[0].data = this.trainLoss;
            this.lossChart.data.datasets[1].data = this.valLoss;
            this.lossChart.update();
        }
        
        // Update accuracy chart
        if (this.accuracyChart) {
            this.accuracyChart.data.labels = this.epochs;
            this.accuracyChart.data.datasets[0].data = this.trainAccuracy;
            this.accuracyChart.data.datasets[1].data = this.valAccuracy;
            this.accuracyChart.update();
        }
    }
    
    updateMetrics(metrics) {
        const metricsContainer = document.getElementById(this.config.metricsContainerId);
        if (!metricsContainer) return;
        
        // Ensure all metric values are valid numbers
        const trainAcc = parseFloat(metrics.trainAccuracy) || 0;
        const valAcc = parseFloat(metrics.valAccuracy) || 0;
        const trainLoss = parseFloat(metrics.trainLoss) || 0;
        const valLoss = parseFloat(metrics.valLoss) || 0;
        
        console.log("Formatting metrics for display:", {
            trainAcc, valAcc, trainLoss, valLoss
        });
        
        // Create metrics HTML
        let html = `
            <div class="metric-card">
                <div class="metric-value">${metrics.currentEpoch} / ${metrics.totalEpochs}</div>
                <div class="metric-label">Epochs</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${trainLoss.toFixed(4)}</div>
                <div class="metric-label">Training Loss</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${valLoss.toFixed(4)}</div>
                <div class="metric-label">Validation Loss</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(trainAcc * 100).toFixed(2)}%</div>
                <div class="metric-label">Training Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(valAcc * 100).toFixed(2)}%</div>
                <div class="metric-label">Validation Accuracy</div>
            </div>`;
            
        // Add test metrics if available
        if (metrics.testLoss !== undefined && metrics.testAccuracy !== undefined) {
            const testLoss = parseFloat(metrics.testLoss) || 0;
            const testAcc = parseFloat(metrics.testAccuracy) || 0;
            
            html += `
                <div class="metric-card">
                    <div class="metric-value">${testLoss.toFixed(4)}</div>
                    <div class="metric-label">Test Loss</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${(testAcc * 100).toFixed(2)}%</div>
                    <div class="metric-label">Test Accuracy</div>
                </div>`;
        }
        
        metricsContainer.innerHTML = html;
    }
    
    updateStatus(status) {
        const statusElement = document.getElementById(this.config.statusElement);
        if (statusElement) {
            statusElement.textContent = status;
            
            // Update icon based on status
            const iconElement = statusElement.closest('.training-status')?.querySelector('i');
            if (iconElement) {
                // Remove all existing classes
                iconElement.className = '';
                
                // Add appropriate icon class
                if (status.includes('progress')) {
                    iconElement.className = 'fas fa-spinner fa-spin';
                } else if (status.includes('completed')) {
                    iconElement.className = 'fas fa-check-circle';
                } else if (status.includes('error') || status.includes('stopped')) {
                    iconElement.className = 'fas fa-exclamation-circle';
                } else {
                    iconElement.className = 'fas fa-info-circle';
                }
            }
        }
    }
    
    // Utility function to convert hex color to rgba
    hexToRgba(hex, alpha = 1) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result
            ? `rgba(${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}, ${alpha})`
            : null;
    }
    
    // Get training data for later use
    getTrainingData() {
        return {
            epochs: [...this.epochs],
            trainLoss: [...this.trainLoss],
            valLoss: [...this.valLoss],
            trainAccuracy: [...this.trainAccuracy],
            valAccuracy: [...this.valAccuracy],
            testLoss: this.testLoss,
            testAccuracy: this.testAccuracy
        };
    }
    
    resetCharts() {
        // Clear all data arrays
        this.epochs = [];
        this.trainLoss = [];
        this.valLoss = [];
        this.trainAccuracy = [];
        this.valAccuracy = [];
        this.testLoss = null;
        this.testAccuracy = null;
        this.currentEpoch = 0;
        
        // Reset training status
        this.isTraining = false;
        
        // Clear charts
        if (this.lossChart) {
            this.lossChart.data.labels = [];
            this.lossChart.data.datasets[0].data = [];
            this.lossChart.data.datasets[1].data = [];
            this.lossChart.update();
        }
        
        if (this.accuracyChart) {
            this.accuracyChart.data.labels = [];
            this.accuracyChart.data.datasets[0].data = [];
            this.accuracyChart.data.datasets[1].data = [];
            this.accuracyChart.update();
        }
        
        // Clear metrics and reset status
        const metricsContainer = document.getElementById(this.config.metricsContainerId);
        if (metricsContainer) {
            metricsContainer.innerHTML = '';
        }
        
        const statusElement = document.getElementById(this.config.statusElement);
        if (statusElement) {
            statusElement.textContent = 'Ready to train';
        }
        
        return this;
    }
}

// Global instance to be used in the application
let trainingMonitor;

// Initialize training monitor
function initTrainingMonitor(config = {}) {
    trainingMonitor = new TrainingMonitor(config);
    return trainingMonitor;
} 