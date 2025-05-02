// UI Effects and Animations
class UIEffects {
    constructor() {
        this.animationCanvas = null;
        this.animationContext = null;
        this.animationParticles = [];
        this.animationFrameId = null;
        this.tooltips = [];
        
        this.init();
    }
    
    init() {
        // Initialize header animation
        this.initHeaderAnimation();
        
        // Initialize code highlighting
        this.initCodeHighlighting();
        
        // Initialize tooltips
        this.initTooltips();
        
        // Initialize code copy functionality
        this.initCodeCopy();
        
        // Add listeners for dynamic content
        this.addListeners();
    }
    
    initHeaderAnimation() {
        // Create neural background canvas
        const headerContainer = document.querySelector('.page-header');
        if (!headerContainer) return;
        
        const canvas = document.createElement('canvas');
        canvas.className = 'neural-bg';
        headerContainer.appendChild(canvas);
        
        this.animationCanvas = canvas;
        this.animationContext = canvas.getContext('2d');
        
        // Set canvas dimensions
        this.resizeCanvas();
        
        // Initialize particles
        this.createParticles();
        
        // Start animation loop
        this.animateHeader();
        
        // Handle resize
        window.addEventListener('resize', () => {
            this.resizeCanvas();
            this.createParticles();
        });
    }
    
    resizeCanvas() {
        if (!this.animationCanvas) return;
        
        const header = this.animationCanvas.parentElement;
        this.animationCanvas.width = header.offsetWidth;
        this.animationCanvas.height = header.offsetHeight;
    }
    
    createParticles() {
        if (!this.animationCanvas) return;
        
        // Clear existing particles
        this.animationParticles = [];
        
        // Create new particles
        const particleCount = Math.floor(this.animationCanvas.width * this.animationCanvas.height / 10000);
        for (let i = 0; i < particleCount; i++) {
            this.animationParticles.push({
                x: Math.random() * this.animationCanvas.width,
                y: Math.random() * this.animationCanvas.height,
                radius: Math.random() * 3 + 1,
                color: this.getRandomColor(),
                speedX: Math.random() * 1 - 0.5,
                speedY: Math.random() * 1 - 0.5,
                connections: []
            });
        }
        
        // Calculate initial connections
        this.updateConnections();
    }
    
    updateConnections() {
        // For each particle, find nearby particles to connect
        this.animationParticles.forEach(particle => {
            particle.connections = [];
            
            this.animationParticles.forEach(otherParticle => {
                if (particle === otherParticle) return;
                
                const dx = particle.x - otherParticle.x;
                const dy = particle.y - otherParticle.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                // Only connect particles within a certain distance
                const maxDistance = 100;
                if (distance < maxDistance) {
                    particle.connections.push({
                        particle: otherParticle,
                        distance
                    });
                }
            });
        });
    }
    
    animateHeader() {
        if (!this.animationCanvas || !this.animationContext) return;
        
        // Clear canvas
        this.animationContext.clearRect(0, 0, this.animationCanvas.width, this.animationCanvas.height);
        
        // Update and draw particles
        this.animationParticles.forEach(particle => {
            // Update position
            particle.x += particle.speedX;
            particle.y += particle.speedY;
            
            // Bounce off edges
            if (particle.x < 0 || particle.x > this.animationCanvas.width) {
                particle.speedX *= -1;
            }
            
            if (particle.y < 0 || particle.y > this.animationCanvas.height) {
                particle.speedY *= -1;
            }
            
            // Draw particle
            this.animationContext.beginPath();
            this.animationContext.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
            this.animationContext.fillStyle = particle.color;
            this.animationContext.fill();
            
            // Draw connections
            particle.connections.forEach(conn => {
                this.animationContext.beginPath();
                this.animationContext.moveTo(particle.x, particle.y);
                this.animationContext.lineTo(conn.particle.x, conn.particle.y);
                
                // Calculate opacity based on distance
                const opacity = 1 - conn.distance / 100;
                this.animationContext.strokeStyle = `rgba(100, 100, 150, ${opacity * 0.2})`;
                this.animationContext.lineWidth = 1;
                this.animationContext.stroke();
            });
        });
        
        // Update connections periodically
        if (Math.random() < 0.05) {
            this.updateConnections();
        }
        
        // Continue animation loop
        this.animationFrameId = requestAnimationFrame(() => this.animateHeader());
    }
    
    getRandomColor() {
        // Return a random color from our theme palette
        const colors = [
            'rgba(123, 104, 238, 0.6)', // Primary - Medium Slate Blue
            'rgba(0, 206, 209, 0.6)',   // Secondary - Dark Turquoise
            'rgba(255, 99, 71, 0.6)'    // Accent - Tomato
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }
    
    initCodeHighlighting() {
        // If highlight.js is available, initialize it
        if (window.hljs) {
            // Apply highlighting to all code blocks
            document.querySelectorAll('pre code').forEach(block => {
                hljs.highlightBlock(block);
            });
        }
    }
    
    initTooltips() {
        // Create tooltip container
        const tooltipContainer = document.createElement('div');
        tooltipContainer.className = 'tooltip';
        tooltipContainer.style.opacity = '0';
        document.body.appendChild(tooltipContainer);
        
        // Initialize tooltips for elements with data-tooltip attribute
        this.initializeTooltipElements();
    }
    
    initializeTooltipElements() {
        document.querySelectorAll('[data-tooltip]').forEach(element => {
            if (!element._hasTooltipListeners) {
                element._hasTooltipListeners = true;
                
                element.addEventListener('mouseenter', (e) => {
                    const tooltip = document.querySelector('.tooltip');
                    tooltip.textContent = e.target.getAttribute('data-tooltip');
                    tooltip.style.left = `${e.pageX + 10}px`;
                    tooltip.style.top = `${e.pageY - 20}px`;
                    tooltip.style.opacity = '1';
                });
                
                element.addEventListener('mousemove', (e) => {
                    const tooltip = document.querySelector('.tooltip');
                    tooltip.style.left = `${e.pageX + 10}px`;
                    tooltip.style.top = `${e.pageY - 20}px`;
                });
                
                element.addEventListener('mouseleave', () => {
                    const tooltip = document.querySelector('.tooltip');
                    tooltip.style.opacity = '0';
                });
            }
        });
    }
    
    initCodeCopy() {
        // Add copy buttons to all code blocks
        document.querySelectorAll('.code-content').forEach(codeBlock => {
            // Skip if copy button already exists
            if (codeBlock.querySelector('.code-copy-btn')) return;
            
            const copyButton = document.createElement('button');
            copyButton.className = 'btn btn-sm btn-icon code-copy-btn';
            copyButton.setAttribute('data-tooltip', 'Copy to clipboard');
            copyButton.innerHTML = '<i class="fas fa-copy"></i>';
            copyButton.style.position = 'absolute';
            copyButton.style.top = '10px';
            copyButton.style.right = '10px';
            
            copyButton.addEventListener('click', () => {
                const codeText = codeBlock.textContent;
                navigator.clipboard.writeText(codeText).then(() => {
                    // Show success feedback
                    copyButton.innerHTML = '<i class="fas fa-check"></i>';
                    setTimeout(() => {
                        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                    }, 2000);
                }).catch(err => {
                    console.error('Could not copy text: ', err);
                    copyButton.innerHTML = '<i class="fas fa-times"></i>';
                    setTimeout(() => {
                        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                    }, 2000);
                });
            });
            
            // Add the copy button to the code block's parent
            const codeContainer = codeBlock.parentElement;
            if (codeContainer.style.position !== 'relative') {
                codeContainer.style.position = 'relative';
            }
            codeContainer.appendChild(copyButton);
        });
    }
    
    addListeners() {
        // Use mutation observer to detect new elements
        const observer = new MutationObserver((mutations) => {
            let shouldRefreshTooltips = false;
            let shouldRefreshCodeBlocks = false;
            
            mutations.forEach(mutation => {
                // Check if new nodes were added
                if (mutation.addedNodes.length) {
                    // Check if we have new tooltip elements
                    const hasTooltipElements = Array.from(mutation.addedNodes).some(node => {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            return node.hasAttribute('data-tooltip') || 
                                   node.querySelector('[data-tooltip]');
                        }
                        return false;
                    });
                    
                    // Check if we have new code blocks
                    const hasCodeBlocks = Array.from(mutation.addedNodes).some(node => {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            return node.classList?.contains('code-content') || 
                                   node.querySelector('.code-content');
                        }
                        return false;
                    });
                    
                    if (hasTooltipElements) shouldRefreshTooltips = true;
                    if (hasCodeBlocks) shouldRefreshCodeBlocks = true;
                }
            });
            
            // Refresh tooltips if new tooltip elements were added
            if (shouldRefreshTooltips) {
                this.initializeTooltipElements();
            }
            
            // Refresh code blocks if new code blocks were added
            if (shouldRefreshCodeBlocks) {
                this.initCodeHighlighting();
                this.initCodeCopy();
            }
        });
        
        // Observe changes to the DOM
        observer.observe(document.body, { 
            childList: true, 
            subtree: true 
        });
    }
    
    // Method to create a typed text animation
    typedTextAnimation(element, text, speed = 50, delay = 0) {
        if (!element) return;
        
        const typingEffect = () => {
            let i = 0;
            element.textContent = '';
            
            const typing = () => {
                if (i < text.length) {
                    element.textContent += text.charAt(i);
                    i++;
                    setTimeout(typing, speed);
                }
            };
            
            typing();
        };
        
        setTimeout(typingEffect, delay);
    }
    
    // Create a particle burst effect (for button clicks, etc.)
    createParticleBurst(x, y, color, count = 20) {
        // Create a canvas for the burst
        const burstCanvas = document.createElement('canvas');
        burstCanvas.className = 'particle-burst';
        burstCanvas.style.position = 'fixed';
        burstCanvas.style.pointerEvents = 'none';
        burstCanvas.style.zIndex = '9999';
        burstCanvas.style.left = '0';
        burstCanvas.style.top = '0';
        burstCanvas.width = window.innerWidth;
        burstCanvas.height = window.innerHeight;
        document.body.appendChild(burstCanvas);
        
        const ctx = burstCanvas.getContext('2d');
        const particles = [];
        
        // Create particles
        for (let i = 0; i < count; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = Math.random() * 5 + 2;
            particles.push({
                x,
                y,
                radius: Math.random() * 4 + 1,
                color: color || this.getRandomColor(),
                speedX: Math.cos(angle) * speed,
                speedY: Math.sin(angle) * speed,
                lifetime: Math.random() * 30 + 20,
                life: 0
            });
        }
        
        // Animate particles
        const animate = () => {
            ctx.clearRect(0, 0, burstCanvas.width, burstCanvas.height);
            
            let isAlive = false;
            
            particles.forEach(particle => {
                if (particle.life < particle.lifetime) {
                    isAlive = true;
                    
                    // Update position
                    particle.x += particle.speedX;
                    particle.y += particle.speedY;
                    
                    // Add gravity
                    particle.speedY += 0.1;
                    
                    // Update life
                    particle.life++;
                    
                    // Calculate opacity based on life
                    const opacity = 1 - particle.life / particle.lifetime;
                    
                    // Draw particle
                    ctx.beginPath();
                    ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
                    ctx.fillStyle = particle.color.replace(/[\d\.]+\)$/g, `${opacity})`);
                    ctx.fill();
                }
            });
            
            if (isAlive) {
                requestAnimationFrame(animate);
            } else {
                // Remove canvas when all particles are dead
                burstCanvas.remove();
            }
        };
        
        animate();
    }
    
    // Add click event to create burst effect on buttons
    addBurstEffectToButtons() {
        document.querySelectorAll('.btn').forEach(button => {
            if (!button._hasBurstEffect) {
                button._hasBurstEffect = true;
                
                button.addEventListener('click', (e) => {
                    // Get button color for the burst
                    const computedStyle = window.getComputedStyle(button);
                    const bgColor = computedStyle.backgroundColor;
                    
                    // Create burst at click position
                    this.createParticleBurst(e.clientX, e.clientY, bgColor);
                });
            }
        });
    }
}

// Initialize UI effects when document is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.uiEffects = new UIEffects();
    
    // Add burst effect to buttons
    window.uiEffects.addBurstEffectToButtons();
    
    // Example of typed text animation for the header title
    const titleElement = document.querySelector('.title-animation h1');
    if (titleElement) {
        window.uiEffects.typedTextAnimation(titleElement, 'Neural Network Builder', 70);
    }
}); 