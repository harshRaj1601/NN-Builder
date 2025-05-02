// Neural Network Visualization using D3.js
class NeuralNetworkVis {
    constructor(config) {
        this.config = {
            containerId: 'network-vis',
            width: 900,
            height: 500,
            layerSpacing: 150,
            neuronRadius: 20,
            neuronMargin: 15,
            transitionDuration: 300, // Reduced for better performance
            colors: {
                input: ['#FF7F50', '#FF6347'],
                hidden: ['#9370DB', '#7B68EE'],
                output: ['#40E0D0', '#00CED1'],
                connection: '#555555',
                connectionHighlight: '#7B68EE'
            },
            onNeuronClick: null,
            onLayerAdd: null,
            onLayerRemove: null,
            onNeuronAdd: null,
            onNeuronRemove: null,
            onActivationChange: null,
            onRegularizationChange: null
        };

        // Override default config with provided values
        Object.assign(this.config, config);

        this.svg = null;
        this.layers = [];
        this.connections = [];
        this.tooltip = null;
        this.selectedNeuron = null;
        this.activationFunctions = ['relu', 'sigmoid', 'tanh', 'linear', 'softmax'];
        this.regularizationOptions = ['none', 'l1', 'l2', 'l1_l2'];
        
        // Add zoom properties
        this.zoom = null;
        this.zoomGroup = null;
        this.currentZoom = 1;
        
        // Performance optimization flags
        this.renderPending = false;
        this.useRequestAnimationFrame = true;
        this.connectionOpacity = 0.5; // Reduce default opacity for better readability
        
        this.initialize();
    }

    initialize() {
        // Create SVG container
        this.svg = d3.select(`#${this.config.containerId}`)
            .append('svg')
            .attr('width', this.config.width)
            .attr('height', this.config.height)
            .attr('class', 'neural-network-svg')
            .style('max-width', '100%')
            .style('height', 'auto')
            .attr('viewBox', `0 0 ${this.config.width} ${this.config.height}`);  // Add viewBox for better scaling

        // Create a group to apply zoom transformations
        this.zoomGroup = this.svg.append('g')
            .attr('class', 'zoom-group');
            
        // Create groups for layers, connections, and interactions
        this.connectionsGroup = this.zoomGroup.append('g').attr('class', 'connections-group');
        this.layersGroup = this.zoomGroup.append('g').attr('class', 'layers-group');
        this.interactionGroup = this.zoomGroup.append('g').attr('class', 'interaction-group');

        // Optimize SVG rendering
        this.svg.style('shape-rendering', 'geometricPrecision');
        
        // Create defs for gradients
        this.svg.append('defs');

        // Create tooltip
        this.tooltip = d3.select('body')
            .append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0);
            
        // Initialize zoom behavior
        this.initializeZoom();

        // Initialize empty layers
        this.addDefaultLayers();
        
        // Add window resize handler
        window.addEventListener('resize', this.debounce(this.handleResize.bind(this), 200));
        
        // Add zoom controls to the SVG
        this.addZoomControls();
        
        // Force an initial resize to ensure proper sizing
        setTimeout(() => {
            this.handleResize();
        }, 0);
    }
    
    // Performance optimization: debounce function to limit frequent calls
    debounce(func, wait) {
        let timeout;
        return function(...args) {
            const context = this;
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(context, args), wait);
        };
    }
    
    // Initialize zoom behavior
    initializeZoom() {
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 4])  // Allow zoom from 0.1x to 4x
            .on('zoom', (event) => {
                this.zoomGroup.attr('transform', event.transform);
                this.currentZoom = event.transform.k;
                
                // Update zoom control button states
                this.updateZoomControlState();
            });
            
        this.svg.call(this.zoom);
        
        // Double click to reset zoom
        this.svg.on('dblclick.zoom', () => {
            this.resetZoom();
        });
    }
    
    // Update zoom control button states
    updateZoomControlState() {
        const zoomInButton = this.svg.select('.zoom-in');
        const zoomOutButton = this.svg.select('.zoom-out');
        
        // Disable zoom in button if at max zoom
        zoomInButton.style('opacity', this.currentZoom >= 4 ? 0.5 : 1)
                    .style('pointer-events', this.currentZoom >= 4 ? 'none' : 'all');
                    
        // Disable zoom out button if at min zoom
        zoomOutButton.style('opacity', this.currentZoom <= 0.1 ? 0.5 : 1)
                     .style('pointer-events', this.currentZoom <= 0.1 ? 'none' : 'all');
    }
    
    // Zoom in by a factor
    zoomIn() {
        this.svg.transition().duration(300).call(
            this.zoom.scaleBy, 1.3
        );
    }
    
    // Zoom out by a factor
    zoomOut() {
        this.svg.transition().duration(300).call(
            this.zoom.scaleBy, 0.7
        );
    }
    
    // Reset zoom to default
    resetZoom() {
        this.svg.transition().duration(300).call(
            this.zoom.transform, d3.zoomIdentity
        );
    }
    
    // Handle resize with performance optimizations
    handleResize() {
        const container = document.getElementById(this.config.containerId);
        if (container) {
            // Force the container to be visible
            container.style.display = 'block';
            
            // Update width to match container, keeping aspect ratio
            const newWidth = container.clientWidth || this.config.width;
            const newHeight = container.clientHeight || this.config.height;
            
            // Calculate layers total width to ensure they fit
            const layerCount = this.layers.length;
            const minRequiredWidth = (layerCount + 1) * this.config.layerSpacing;
            
            // If container is too narrow, adjust layer spacing
            if (newWidth < minRequiredWidth) {
                this.config.layerSpacing = (newWidth / (layerCount + 1)) * 0.9;
            } else if (this.config.layerSpacing < 150 && newWidth >= minRequiredWidth) {
                // Restore default spacing if there's enough space now
                this.config.layerSpacing = 150;
            }
            
            this.config.width = newWidth;
            this.config.height = newHeight > 300 ? newHeight : 500; // Ensure minimum height
            
            this.svg
                .attr('width', this.config.width)
                .attr('height', this.config.height)
                .attr('viewBox', `0 0 ${this.config.width} ${this.config.height}`);
                
            // Update zoom controls position
            this.svg.select('.zoom-controls-container')
                .attr('transform', `translate(${this.config.width - 60}, 20)`);
                
            // Request render using RAF for performance
            this.requestRender();
        }
    }
    
    // Queue a render using requestAnimationFrame for better performance
    requestRender() {
        if (this.useRequestAnimationFrame && !this.renderPending) {
            this.renderPending = true;
            window.requestAnimationFrame(() => {
                this.render();
                this.renderPending = false;
            });
        } else if (!this.useRequestAnimationFrame) {
            this.render();
        }
    }

    addDefaultLayers() {
        // Add default input and output layers
        this.layers = [
            {
                id: 'input',
                type: 'input',
                neurons: [
                    { id: 'i1', value: 'X₁' },
                    { id: 'i2', value: 'X₂' }
                ],
                activationFn: null,
                regularization: 'none'
            },
            {
                id: 'hidden1',
                type: 'hidden',
                neurons: [
                    { id: 'h1_1', value: '' },
                    { id: 'h1_2', value: '' }
                ],
                activationFn: 'relu',
                regularization: 'none'
            },
            {
                id: 'output',
                type: 'output',
                neurons: [
                    { id: 'o1', value: 'Y' }
                ],
                activationFn: 'sigmoid',
                regularization: 'none'
            }
        ];

        this.render();
    }

    render() {
        // Clear any existing elements to prevent memory leaks
        this.connectionsGroup.selectAll('*').remove();
        this.layersGroup.selectAll('*').remove();
        this.interactionGroup.selectAll('*').remove();
        
        // Render layers first to calculate neuron positions
        this.renderLayers();
        // Then render connections using the calculated positions
        this.renderConnections();
        // Finally add interaction elements on top
        this.addInteractions();
    }

    renderLayers() {
        const layerSpacing = this.config.layerSpacing;
        const centralY = this.config.height / 2;
        
        // Calculate positions for layers
        this.layers.forEach((layer, i) => {
            const x = (i + 1) * layerSpacing;
            layer.x = x;
            layer.y = centralY;
            
            // Calculate neuron positions within the layer
            const totalNeurons = layer.neurons.length;
            const totalHeight = totalNeurons * (this.config.neuronRadius * 2 + this.config.neuronMargin) - this.config.neuronMargin;
            const startY = centralY - totalHeight / 2;
            
            layer.neurons.forEach((neuron, j) => {
                neuron.x = x;
                neuron.y = startY + j * (this.config.neuronRadius * 2 + this.config.neuronMargin) + this.config.neuronRadius;
                neuron.layer = layer.id;
            });
        });
        
        // Store a reference to this for use in callback functions
        const self = this;
        
        // Clear existing layer groups
        this.layersGroup.selectAll('.layer').remove();
        
        // Create layer groups
        const layerGroups = this.layersGroup
            .selectAll('.layer')
            .data(this.layers, d => d.id)
            .enter()
            .append('g')
            .attr('class', d => `layer layer-${d.type}`)
            .attr('id', d => `layer-${d.id}`);
        
        // Add layer info box
        layerGroups.append('rect')
            .attr('class', 'layer-info-bg')
            .attr('rx', 5)
            .attr('ry', 5)
            .attr('x', d => d.x - 60)
            .attr('y', d => {
                const neuronCount = d.neurons.length;
                const totalHeight = neuronCount * (this.config.neuronRadius * 2 + this.config.neuronMargin);
                return d.y - totalHeight / 2 - 30;
            })
            .attr('width', 120)
            .attr('height', 25);
            
        layerGroups.append('text')
            .attr('class', 'layer-info-text')
            .attr('x', d => d.x)
            .attr('y', d => {
                const neuronCount = d.neurons.length;
                const totalHeight = neuronCount * (this.config.neuronRadius * 2 + this.config.neuronMargin);
                return d.y - totalHeight / 2 - 15;
            })
            .attr('text-anchor', 'middle')
            .text(d => {
                if (d.type === 'input') return 'Input Layer';
                if (d.type === 'output') return 'Output Layer';
                return 'Hidden Layer';
            });
        
        // Add layer controls at the top of each layer (except for input layer)
        layerGroups.filter(d => d.type !== 'input')
            .append('g')
            .attr('class', 'layer-controls')
            .attr('transform', d => {
                const neuronCount = d.neurons.length;
                const totalHeight = neuronCount * (this.config.neuronRadius * 2 + this.config.neuronMargin);
                return `translate(${d.x}, ${d.y - totalHeight / 2 - 50})`;
            })
            .call(g => {
                // Add activation function dropdown for hidden and output layers
                g.filter(d => d.type === 'hidden' || d.type === 'output')
                .append('foreignObject')
                    .attr('x', -60)
                    .attr('y', -45)
                    .attr('width', 120)
                    .attr('height', 30)
                    .append('xhtml:div')
                    .html(d => {
                        let options = '';
                        self.activationFunctions.forEach(fn => {
                            const selected = fn === d.activationFn ? 'selected' : '';
                            options += `<option value="${fn}" ${selected}>${fn}</option>`;
                        });
                        return `<select class="layer-activation form-select form-select-sm" data-layer="${d.id}">
                                ${options}
                            </select>`;
                    })
                    .on('change', function(event) {
                        const layerId = this.querySelector('select').getAttribute('data-layer');
                        const activationFn = this.querySelector('select').value;
                        self.updateLayerActivation(layerId, activationFn);
                    });
                        
                // Add regularization dropdown    
                g.filter(d => d.type === 'hidden' || d.type === 'output')
                .append('foreignObject')
                    .attr('x', -60)
                    .attr('y', -15)
                    .attr('width', 120)
                    .attr('height', 30)
                    .append('xhtml:div')
                    .html(d => {
                        let options = '';
                        self.regularizationOptions.forEach(option => {
                            const selected = option === d.regularization ? 'selected' : '';
                            options += `<option value="${option}" ${selected}>${option}</option>`;
                        });
                        return `<select class="layer-regularization form-select form-select-sm" data-layer="${d.id}">
                                ${options}
                            </select>`;
                    })
                    .on('change', function(event) {
                        const layerId = this.querySelector('select').getAttribute('data-layer');
                        const regularization = this.querySelector('select').value;
                        self.updateLayerRegularization(layerId, regularization);
                    });
            });
        
        // Clear any existing gradients to prevent duplicates
        this.svg.select('defs').selectAll('*').remove();
        
        // Add neurons for each layer
        this.layers.forEach(layer => {
            const layerGroup = this.layersGroup.select(`#layer-${layer.id}`);
            
            // Create neurons
            const neurons = layerGroup.selectAll('.neuron')
                .data(layer.neurons)
                .enter()
                .append('g')
                .attr('class', 'neuron')
                .attr('id', d => `neuron-${d.id}`)
                .attr('transform', d => `translate(${d.x}, ${d.y})`)
                .style('cursor', 'pointer');
                
            // Create gradient for each neuron
            layer.neurons.forEach(neuron => {
                const gradientId = `gradient-${neuron.id}`;
                
                // Define gradient colors based on layer type
                let colors;
                if (layer.type === 'input') {
                    colors = self.config.colors.input;
                } else if (layer.type === 'output') {
                    colors = self.config.colors.output;
                } else {
                    colors = self.config.colors.hidden;
                }
                
                // Create gradient definition
                const gradient = self.svg.select('defs')
                    .append('radialGradient')
                    .attr('id', gradientId)
                    .attr('cx', '30%')
                    .attr('cy', '30%')
                    .attr('r', '70%');
                    
                gradient.append('stop')
                    .attr('offset', '0%')
                    .attr('stop-color', colors[0]);
                    
                gradient.append('stop')
                    .attr('offset', '100%')
                    .attr('stop-color', colors[1]);
            });
            
            // Add neuron circle
            neurons.append('circle')
                .attr('class', 'neuron-circle')
                .attr('r', self.config.neuronRadius)
                .attr('cx', 0)
                .attr('cy', 0)
                .attr('fill', d => `url(#gradient-${d.id})`)
                .attr('stroke', '#ffffff')
                .attr('stroke-width', 1.5);
                
            // Add neuron text
            neurons.append('text')
                .attr('class', 'neuron-text')
                .attr('x', 0)
                .attr('y', 0)
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'central')
                .attr('fill', 'white')
                .text(d => d.value);
                
            // Add neuron event handlers
            neurons
                .on('mouseover', function(event, d) {
                    d3.select(this).select('.neuron-circle')
                        .transition()
                        .duration(150)
                        .attr('r', self.config.neuronRadius * 1.1);
                    
                    // Highlight connections
                    self.connections.forEach(conn => {
                        if (conn.source.id === d.id || conn.target.id === d.id) {
                            d3.select(`#connection-${conn.source.id}-${conn.target.id}`)
                                .style('stroke', self.config.colors.connectionHighlight)
                                .style('stroke-width', Math.max(2, conn.value * 4))
                                .style('opacity', 0.8);
                        }
                    });
                    
                    // Show tooltip
                    self.tooltip
                        .html(`<strong>${d.layer}</strong><br>Neuron ${d.id}`)
                        .style('opacity', 0.9)
                        .style('left', `${event.pageX + 10}px`)
                        .style('top', `${event.pageY - 20}px`);
                })
                .on('mouseout', function(event, d) {
                    if (d.id !== (self.selectedNeuron && self.selectedNeuron.id)) {
                        d3.select(this).select('.neuron-circle')
                            .transition()
                            .duration(150)
                            .attr('r', self.config.neuronRadius);
                    }
                    
                    // Reset connections unless selected
                    self.connections.forEach(conn => {
                        if ((conn.source.id === d.id || conn.target.id === d.id) && 
                            (!self.selectedNeuron || (conn.source.id !== self.selectedNeuron.id && conn.target.id !== self.selectedNeuron.id))) {
                            d3.select(`#connection-${conn.source.id}-${conn.target.id}`)
                                .style('stroke', self.config.colors.connection)
                                .style('stroke-width', Math.max(1, conn.value * 3))
                                .style('opacity', Math.max(0.2, conn.value));
                        }
                    });
                    
                    // Hide tooltip
                    self.tooltip.style('opacity', 0);
                })
                .on('click', function(event, d) {
                    event.stopPropagation();
                    self.selectNeuron(d);
                });
        });
        
        // Add layer add buttons between layers
        if (this.layers.length < 10) { // Limit to 10 layers for performance
            this.interactionGroup.selectAll('.add-layer-group').remove();
            this.interactionGroup.selectAll('.remove-layer-group').remove();
            
            // Create groups for add/remove layer buttons
            const layerButtonGroups = this.interactionGroup.selectAll('.layer-button-group')
                .data(this.layers.slice(0, -1)) // Exclude the last layer
                .enter()
                .append('g')
                .attr('class', 'layer-button-group')
                .attr('transform', (d, i) => {
                    const x = (d.x + this.layers[i + 1].x) / 2;
                    return `translate(${x}, ${centralY})`; // Center vertically
                });
                
            // Filter out button groups where the next layer is the output layer
            layerButtonGroups.each(function(d, i) {
                const nextLayer = self.layers[i + 1];
                if (nextLayer.type === 'output') {
                    // Skip creating buttons if the next layer is output
                    d3.select(this).remove();
                    return;
                }
                
                // Add layer add button (centered)
                const buttonGroup = d3.select(this);
                
                buttonGroup.append('g')
                    .attr('class', 'add-layer-group')
                    .attr('transform', 'translate(0, 0)') // Centered position
                    .call(g => {
                        g.append('circle')
                            .attr('class', 'add-layer-btn')
                            .attr('r', 15)
                            .attr('fill', 'rgba(45, 45, 55, 0.9)')
                            .attr('stroke', '#aaa')
                            .attr('stroke-width', 1.5)
                            .style('cursor', 'pointer');
                            
                        g.append('text')
                            .attr('class', 'add-layer-icon')
                            .attr('text-anchor', 'middle')
                            .attr('dominant-baseline', 'central')
                            .attr('fill', 'white')
                            .attr('font-size', '20px')
                            .style('pointer-events', 'none')
                            .text('+');
                            
                        g.on('click', function(event) {
                            event.stopPropagation();
                            self.addLayer(d.id);
                        });
                    });
            });
        }
        
        // Add neuron add/remove buttons for each layer
        this.interactionGroup.selectAll('.add-neuron-group').remove();
        this.interactionGroup.selectAll('.remove-neuron-group').remove();
        this.interactionGroup.selectAll('.remove-layer-btn-bottom').remove();
        
        this.layers.forEach(layer => {
            if (layer.type !== 'input' && layer.type !== 'output') { // Don't add controls for input or output layer
                const lastNeuron = layer.neurons[layer.neurons.length - 1];
                const y = lastNeuron.y + this.config.neuronRadius * 2 + this.config.neuronMargin;
                
                // Create a control group for triangle layout
                const controlGroup = this.interactionGroup
                    .append('g')
                    .attr('class', 'neuron-control-group')
                    .attr('transform', `translate(${layer.x}, ${y})`);
                
                // Only add neuron controls if under the limit and not input layer
                if (layer.neurons.length < 10) {
                    // Add neuron plus button (left position at top)
                    controlGroup.append('g')
                        .attr('class', 'add-neuron-group')
                        .attr('transform', 'translate(-15, 0)')
                        .call(g => {
                            g.append('circle')
                                .attr('class', 'add-neuron-btn')
                                .attr('r', 12)
                                .attr('fill', 'rgba(45, 45, 55, 0.9)')
                                .attr('stroke', '#aaa')
                                .attr('stroke-width', 1.5)
                                .style('cursor', 'pointer');
                                
                            g.append('text')
                                .attr('text-anchor', 'middle')
                                .attr('dominant-baseline', 'central')
                                .attr('fill', 'white')
                                .attr('font-size', '16px')
                                .style('pointer-events', 'none')
                                .text('+');
                                
                            g.on('click', function(event) {
                                event.stopPropagation();
                                self.addNeuron(layer.id);
                            });
                        });
                    
                    // Add neuron minus button (right position at top)
                    if (layer.neurons.length > 1) { // Only allow removing if more than 1 neuron
                        controlGroup.append('g')
                            .attr('class', 'remove-neuron-group')
                            .attr('transform', 'translate(15, 0)') // Right neighbor to plus button
                            .call(g => {
                                g.append('circle')
                                    .attr('class', 'remove-neuron-btn')
                                    .attr('r', 12)
                                    .attr('fill', 'rgba(45, 45, 55, 0.9)')
                                    .attr('stroke', '#aaa')
                                    .attr('stroke-width', 1.5)
                                    .style('cursor', 'pointer');
                                    
                                g.append('text')
                                    .attr('text-anchor', 'middle')
                                    .attr('dominant-baseline', 'central')
                                    .attr('fill', 'white')
                                    .attr('font-size', '16px')
                                    .style('pointer-events', 'none')
                                    .text('−');
                                    
                                g.on('click', function(event) {
                                    event.stopPropagation();
                                    self.removeNeuron(layer.neurons[layer.neurons.length - 1].id); // Remove last neuron
                                });
                            });
                    }
                }
                
                // Add layer remove button (bottom position - below the plus and minus)
                if (layer.type === 'hidden') {
                    controlGroup.append('g')
                        .attr('class', 'remove-layer-btn-bottom')
                        .attr('transform', 'translate(0, 25)') // Below the plus and minus buttons
                        .call(g => {
                            g.append('circle')
                                .attr('class', 'remove-layer-btn')
                                .attr('r', 12)
                                .attr('fill', 'rgba(224, 83, 63, 0.7)') // Reddish color for delete
                                .attr('stroke', '#aaa')
                                .attr('stroke-width', 1.5)
                                .style('cursor', 'pointer');
                                
                            g.append('text')
                                .attr('text-anchor', 'middle')
                                .attr('dominant-baseline', 'central')
                                .attr('fill', 'white')
                                .attr('font-size', '16px')
                                .style('pointer-events', 'none')
                                .text('×');
                                
                            g.on('click', function(event) {
                                event.stopPropagation();
                                self.removeLayer(layer.id);
                            });
                        });
                }
            }
        });
        
        // Empty click handler to clear selection
        this.svg.on('click', () => this.clearSelection());
    }
    
    updateLayerActivation(layerId, activationFn) {
        const layer = this.layers.find(l => l.id === layerId);
        if (layer) {
            layer.activationFn = activationFn;
            this.renderLayers();
            
            if (this.config.onActivationChange) {
                this.config.onActivationChange({ layerId, activationFn });
            }
        }
    }
    
    updateLayerRegularization(layerId, regularization) {
        const layer = this.layers.find(l => l.id === layerId);
        if (layer) {
            layer.regularization = regularization;
            
            if (this.config.onRegularizationChange) {
                this.config.onRegularizationChange({ layerId, regularization });
            }
        }
    }
    
    renderConnections() {
        // Pre-calculate all connections for better performance
        this.connections = [];
        
        // Clean up existing connections to prevent duplicates
        this.connectionsGroup.selectAll('*').remove();
        
        for (let i = 0; i < this.layers.length - 1; i++) {
            const sourceLayer = this.layers[i];
            const targetLayer = this.layers[i + 1];
            
            for (let j = 0; j < sourceLayer.neurons.length; j++) {
                for (let k = 0; k < targetLayer.neurons.length; k++) {
                    this.connections.push({
                        source: sourceLayer.neurons[j],
                        target: targetLayer.neurons[k],
                        value: 0.5  // Default weight value
                    });
                }
            }
        }
        
        // Batch render connections for better performance
        const connectionPaths = this.connectionsGroup
            .selectAll('.connection')
            .data(this.connections, d => `${d.source.id}-${d.target.id}`);
        
        connectionPaths.enter()
            .append('path')
            .attr('class', 'connection')
            .attr('id', d => `connection-${d.source.id}-${d.target.id}`)
            .attr('d', d => {
                // Start and end at the exact centers of neurons
                const x1 = d.source.x;
                const y1 = d.source.y;
                const x2 = d.target.x;
                const y2 = d.target.y;
                
                // Use a simple cubic bezier curve with control points positioned 
                // to create a gentle curve that looks natural
                return `M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}`;
            })
            .style('fill', 'none')
            .style('stroke', this.config.colors.connection)
            .style('stroke-width', d => Math.max(1, d.value * 3))
            .style('opacity', d => Math.max(0.2, Math.min(0.8, d.value)))
            .style('stroke-linecap', 'round');
    }
    
    addInteractions() {
        // Clear selection when clicking on empty space
        this.svg.on('click', () => {
            this.clearSelection();
        });
        
        // Add event listeners for layer controls
        this.layersGroup.selectAll('.layer-delete-btn')
            .on('click', (event, d) => {
                event.stopPropagation();
                const layerId = d3.select(event.target.parentNode.parentNode.parentNode).datum().id;
                this.removeLayer(layerId);
            });
    }
    
    selectNeuron(neuron) {
        this.clearSelection();
        this.selectedNeuron = neuron;
        
        d3.select(`#neuron-${neuron.id}`)
            .classed('selected', true)
            .select('.neuron-circle')
            .attr('stroke', '#7B68EE')
            .attr('stroke-width', 3);
            
        // Highlight connections
        this.connections.forEach(conn => {
            if (conn.source.id === neuron.id || conn.target.id === neuron.id) {
                d3.select(`#connection-${conn.source.id}-${conn.target.id}`)
                    .style('stroke', this.config.colors.connectionHighlight)
                    .style('stroke-width', Math.max(2, conn.value * 5))
                    .style('opacity', 1);
            }
        });
    }
    
    clearSelection() {
        if (this.selectedNeuron) {
            d3.select(`#neuron-${this.selectedNeuron.id}`)
                .classed('selected', false)
                .select('.neuron-circle')
                .attr('stroke', '#ffffff')
                .attr('stroke-width', 1.5);
            
            // Reset connection styles
            this.connections.forEach(conn => {
                d3.select(`#connection-${conn.source.id}-${conn.target.id}`)
                    .style('stroke', this.config.colors.connection)
                    .style('stroke-width', Math.max(1, conn.value * 3))
                    .style('opacity', Math.max(0.2, conn.value));
            });
            
            this.selectedNeuron = null;
        }
    }
    
    addLayer(afterLayerId) {
        const layerIndex = this.layers.findIndex(l => l.id === afterLayerId);
        if (layerIndex === -1) return;
        
        // Generate new layer ID
        const newLayerId = `hidden${Date.now().toString().substr(-4)}`;
        
        // Create new hidden layer with default neurons
        const newLayer = {
            id: newLayerId,
            type: 'hidden',
            neurons: [
                { id: `${newLayerId}_n1`, value: '' },
                { id: `${newLayerId}_n2`, value: '' }
            ],
            activationFn: 'relu',
            regularization: 'none'
        };
        
        // Insert new layer after the specified layer
        this.layers.splice(layerIndex + 1, 0, newLayer);
        
        // Re-render the network
        this.render();
        
        // Call callback if provided
        if (this.config.onLayerAdd) {
            this.config.onLayerAdd(newLayer);
        }
    }
    
    removeLayer(layerId) {
        const layerIndex = this.layers.findIndex(l => l.id === layerId);
        if (layerIndex === -1 || this.layers[layerIndex].type === 'input' || this.layers[layerIndex].type === 'output') {
            return; // Can't remove input or output layers
        }
        
        // Remove the layer
        const removedLayer = this.layers.splice(layerIndex, 1)[0];
        
        // Re-render the network
        this.render();
        
        // Call callback if provided
        if (this.config.onLayerRemove) {
            this.config.onLayerRemove(removedLayer);
        }
    }
    
    addNeuron(layerId) {
        const layerIndex = this.layers.findIndex(l => l.id === layerId);
        if (layerIndex === -1) return;
        
        const layer = this.layers[layerIndex];
        
        // Generate new neuron ID
        const newNeuronId = `${layer.id}_n${layer.neurons.length + 1}`;
        
        // Add new neuron to the layer
        layer.neurons.push({
            id: newNeuronId,
            value: '',
            layer: layer.id
        });
        
        // Re-render the network
        this.render();
        
        // Call callback if provided
        if (this.config.onNeuronAdd) {
            this.config.onNeuronAdd({ layerId, neuronId: newNeuronId });
        }
    }
    
    removeNeuron(neuronId) {
        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];
            const neuronIndex = layer.neurons.findIndex(n => n.id === neuronId);
            
            if (neuronIndex !== -1) {
                // Don't allow removing the last neuron in a layer
                if (layer.neurons.length <= 1) return;
                
                // Remove the neuron
                const removedNeuron = layer.neurons.splice(neuronIndex, 1)[0];
                
                // Re-render the network
                this.render();
                
                // Call callback if provided
                if (this.config.onNeuronRemove) {
                    this.config.onNeuronRemove({ layerId: layer.id, neuronId });
                }
                
                break;
            }
        }
    }
    
    updateNetworkStructure(architecture) {
        // Convert architecture to our internal format
        const newLayers = [];
        
        // Input layer
        const inputNeurons = [];
        for (let i = 0; i < (architecture.inputShape || 2); i++) {
            inputNeurons.push({
                id: `input_n${i+1}`,
                value: `X${i+1}`
            });
        }
        
        newLayers.push({
            id: 'input',
            type: 'input',
            neurons: inputNeurons,
            activationFn: null,
            regularization: 'none'
        });
        
        // Hidden layers
        if (architecture.layers && architecture.layers.length > 0) {
            // Count how many hidden layers we have (all but the last one which is output)
            const hiddenLayerCount = architecture.layers.length - 1;
            
            for (let i = 0; i < hiddenLayerCount; i++) {
                const layer = architecture.layers[i];
                const neurons = [];
                
                for (let j = 0; j < (layer.units || 2); j++) {
                    neurons.push({
                        id: `hidden${i+1}_n${j+1}`,
                        value: ''
                    });
                }
                
                newLayers.push({
                    id: `hidden${i+1}`,
                    type: 'hidden',
                    neurons: neurons,
                    activationFn: layer.activation || 'relu',
                    regularization: layer.regularization || 'none'
                });
            }
            
            // Output layer
            const outputLayer = architecture.layers[architecture.layers.length - 1];
            const outputNeurons = [];
            
            for (let i = 0; i < (outputLayer.units || 1); i++) {
                outputNeurons.push({
                    id: `output_n${i+1}`,
                    value: outputLayer.units > 1 ? `Y${i+1}` : 'Y'
                });
            }
            
            newLayers.push({
                id: 'output',
                type: 'output',
                neurons: outputNeurons,
                activationFn: outputLayer.activation || 'softmax',
                regularization: outputLayer.regularization || 'none'
            });
        } else {
            // If no layers provided, create a default output layer
            newLayers.push({
                id: 'output',
                type: 'output',
                neurons: [{
                    id: 'output_n1',
                    value: 'Y'
                }],
                activationFn: 'softmax',
                regularization: 'none'
            });
        }
        
        this.layers = newLayers;
        this.render();
        
        // Adjust spacing after rendering to ensure layers fit
        this.handleResize();
    }
    
    updateWeights(weights) {
        // Update connection weights if provided
        if (!weights || !weights.length) return;
        
        let connectionIndex = 0;
        
        for (let i = 0; i < weights.length; i += 2) {
            const layerWeights = weights[i];
            // Iterate through all source and target neurons to update weights
            for (let sourceIdx = 0; sourceIdx < this.layers[i].neurons.length; sourceIdx++) {
                for (let targetIdx = 0; targetIdx < this.layers[i+1].neurons.length; targetIdx++) {
                    // Find the connection
                    const connection = this.connections[connectionIndex++];
                    if (connection) {
                        // Normalize weight value for visualization (between 0 and 1)
                        const absWeight = Math.abs(layerWeights[sourceIdx][targetIdx]);
                        const normalizedWeight = Math.min(1, absWeight / 2);
                        connection.value = normalizedWeight;
                    }
                }
            }
        }
        
        // Re-render the connections with updated weights
        this.renderConnections();
    }
    
    isProbablyRegression() {
        // Try to guess if this is a regression task based on output layer
        const outputLayer = this.layers[this.layers.length - 1];
        return outputLayer && outputLayer.neurons.length === 1 && 
               (outputLayer.activationFn === 'linear' || !outputLayer.activationFn);
    }
    
    getNetworkArchitecture() {
        // Convert our internal format to a Keras-like architecture format that matches exactly what's shown in the UI
        const architecture = {
            inputShape: this.layers[0].neurons.length,
            layers: []
        };
        
        console.log("Internal layers structure:", JSON.stringify(this.layers));
        
        // Add all layers except input
        for (let i = 1; i < this.layers.length; i++) {
            const layer = this.layers[i];
            
            // Get the exact neuron count for this layer
            const neuronCount = layer.neurons ? layer.neurons.length : 0;
            
            // Get the activation function for this layer
            // Use the explicitly set activation function, or a sensible default based on layer type
            const activation = layer.activationFn || 
                             (layer.type === 'output' ? 'linear' : 'relu');
            
            const layerConfig = {
                type: 'Dense',
                units: neuronCount,
                activation: activation
            };
            
            // Add regularization if it's not 'none'
            if (layer.regularization && layer.regularization !== 'none') {
                layerConfig.regularization = layer.regularization;
            }
            
            architecture.layers.push(layerConfig);
            
            console.log(`Layer ${i} (${layer.type}): ${neuronCount} neurons, activation: ${activation}`);
        }
        
        // Debug the final architecture
        console.log("Final architecture from visualization:", JSON.stringify(architecture));
        
        return architecture;
    }
    
    resize(width, height) {
        if (width) this.config.width = width;
        if (height) this.config.height = height;
        
        this.svg
            .attr('width', this.config.width)
            .attr('height', this.config.height);
            
        this.render();
    }

    // Add zoom control buttons
    addZoomControls() {
        // Remove any existing controls first to prevent duplication
        this.svg.selectAll('.zoom-controls-container').remove();
        
        // Create a fixed container positioned at the top-right
        const controlsContainer = this.svg.append('g')
            .attr('class', 'zoom-controls-container')
            .attr('transform', `translate(${this.config.width - 60}, 20)`)
            .style('pointer-events', 'none');
        
        const controlGroup = controlsContainer.append('g')
            .attr('class', 'zoom-controls')
            .style('pointer-events', 'all');
        
        // Create the zoom in button
        const zoomInButton = controlGroup.append('g')
            .attr('class', 'zoom-button')
            .attr('transform', 'translate(0, 0)')
            .style('cursor', 'pointer');
            
        zoomInButton.append('circle')
            .attr('class', 'zoom-button-bg')
            .attr('r', 15)
            .attr('cx', 0)
            .attr('cy', 0)
            .attr('fill', 'rgba(45, 45, 55, 0.95)')
            .attr('stroke', '#aaa')
            .attr('stroke-width', 1.5);
            
        zoomInButton.append('text')
            .attr('class', 'zoom-button-icon')
            .attr('x', 0)
            .attr('y', 0)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .attr('fill', 'white')
            .attr('font-size', '18px')
            .attr('font-weight', 'bold')
            .attr('pointer-events', 'none')
            .text('+');
            
        zoomInButton.on('click', (event) => {
            event.stopPropagation();
            this.zoomIn();
        });
        
        // Create the zoom out button
        const zoomOutButton = controlGroup.append('g')
            .attr('class', 'zoom-button')
            .attr('transform', 'translate(0, 40)')
            .style('cursor', 'pointer');
            
        zoomOutButton.append('circle')
            .attr('class', 'zoom-button-bg')
            .attr('r', 15)
            .attr('cx', 0)
            .attr('cy', 0)
            .attr('fill', 'rgba(45, 45, 55, 0.95)')
            .attr('stroke', '#aaa')
            .attr('stroke-width', 1.5);
            
        zoomOutButton.append('text')
            .attr('class', 'zoom-button-icon')
            .attr('x', 0)
            .attr('y', 0)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .attr('fill', 'white')
            .attr('font-size', '18px')
            .attr('font-weight', 'bold')
            .attr('pointer-events', 'none')
            .text('−');
            
        zoomOutButton.on('click', (event) => {
            event.stopPropagation();
            this.zoomOut();
        });
        
        // Create the reset zoom button
        const resetButton = controlGroup.append('g')
            .attr('class', 'zoom-button')
            .attr('transform', 'translate(0, 80)')
            .style('cursor', 'pointer');
            
        resetButton.append('circle')
            .attr('class', 'zoom-button-bg')
            .attr('r', 15)
            .attr('cx', 0)
            .attr('cy', 0)
            .attr('fill', 'rgba(45, 45, 55, 0.95)')
            .attr('stroke', '#aaa')
            .attr('stroke-width', 1.5);
            
        resetButton.append('text')
            .attr('class', 'zoom-button-icon')
            .attr('x', 0)
            .attr('y', 0)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .attr('fill', 'white')
            .attr('font-size', '16px')
            .attr('font-weight', 'bold')
            .attr('pointer-events', 'none')
            .text('⟲');
            
        resetButton.on('click', (event) => {
            event.stopPropagation();
            this.resetZoom();
        });
    }
}

// Global instance to be used in the application
let neuralVis;

// Initialize neural network visualization
function initNeuralNetworkVis(containerId, config = {}) {
    // Clean up any existing visualization first
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '';
        
        // Force container to have a minimum size if not already set
        if (!container.style.minHeight) {
            container.style.minHeight = '400px';
        }
        
        // Make sure the container is visible
        container.style.display = 'block';
    }
    
    neuralVis = new NeuralNetworkVis({
        containerId,
        ...config
    });
    
    // Force a resize after a short delay to make sure all DOM measurements are accurate
    setTimeout(() => {
        if (neuralVis) {
            neuralVis.handleResize();
        }
    }, 50);
    
    return neuralVis;
}

// Update visualization when window is resized
window.addEventListener('resize', () => {
    if (neuralVis) {
        neuralVis.handleResize();
    }
}); 