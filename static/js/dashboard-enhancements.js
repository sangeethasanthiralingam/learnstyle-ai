// Dashboard Enhancements - Vanilla JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize dashboard enhancements
    initProgressBars();
    initCharts();
    initRealTimeUpdates();
    initContentRecommendations();
});

// Animated Progress Bars
function initProgressBars() {
    const progressBars = document.querySelectorAll('.progress-fill');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const progressBar = entry.target;
                const percentage = progressBar.dataset.percentage || 0;
                
                setTimeout(() => {
                    progressBar.style.width = percentage + '%';
                }, 200);
            }
        });
    });
    
    progressBars.forEach(bar => observer.observe(bar));
}

// Interactive Charts using Chart.js
function initCharts() {
    // Check if Chart.js is loaded
    if (typeof Chart === 'undefined') {
        loadChartJS().then(() => initCharts());
        return;
    }
    
    // Learning Style Distribution Chart
    const styleChartCanvas = document.getElementById('style-chart');
    if (styleChartCanvas) {
        const ctx = styleChartCanvas.getContext('2d');
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Visual', 'Auditory', 'Kinesthetic'],
                datasets: [{
                    data: [
                        styleChartCanvas.dataset.visual || 0,
                        styleChartCanvas.dataset.auditory || 0,
                        styleChartCanvas.dataset.kinesthetic || 0
                    ],
                    backgroundColor: [
                        '#3498db',
                        '#e74c3c',
                        '#f39c12'
                    ],
                    borderWidth: 3,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed + '%';
                            }
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    duration: 2000
                }
            }
        });
    }
    
    // Learning Progress Chart
    const progressChartCanvas = document.getElementById('progress-chart');
    if (progressChartCanvas) {
        const ctx = progressChartCanvas.getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'],
                datasets: [{
                    label: 'Learning Progress',
                    data: [20, 35, 45, 60, 75, 85],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }
}

// Load Chart.js dynamically
function loadChartJS() {
    return new Promise((resolve, reject) => {
        if (typeof Chart !== 'undefined') {
            resolve();
            return;
        }
        
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

// Real-time Updates
function initRealTimeUpdates() {
    // Update stats every 30 seconds
    setInterval(updateDashboardStats, 30000);
    
    // Update content recommendations
    setInterval(updateContentRecommendations, 60000);
}

async function updateDashboardStats() {
    try {
        const response = await fetch('/api/dashboard/statistics');
        if (response.ok) {
            const data = await response.json();
            updateStatsDisplay(data);
        }
    } catch (error) {
        console.log('Stats update failed:', error);
    }
}

function updateStatsDisplay(data) {
    // Update user count
    const userCount = document.getElementById('user-count');
    if (userCount) {
        animateNumber(userCount, data.total_users || 0);
    }
    
    // Update learning sessions
    const sessionCount = document.getElementById('session-count');
    if (sessionCount) {
        animateNumber(sessionCount, data.learning_sessions || 0);
    }
    
    // Update AI predictions
    const predictionCount = document.getElementById('prediction-count');
    if (predictionCount) {
        animateNumber(predictionCount, data.ai_predictions || 0);
    }
}

function animateNumber(element, targetValue) {
    const currentValue = parseInt(element.textContent) || 0;
    const increment = (targetValue - currentValue) / 20;
    let current = currentValue;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= targetValue) || 
            (increment < 0 && current <= targetValue)) {
            current = targetValue;
            clearInterval(timer);
        }
        element.textContent = Math.floor(current);
    }, 50);
}

// Content Recommendations
function initContentRecommendations() {
    loadContentRecommendations();
}

async function loadContentRecommendations() {
    try {
        const response = await fetch('/api/content');
        if (response.ok) {
            const data = await response.json();
            displayContentRecommendations(data.content || []);
        }
    } catch (error) {
        console.log('Content loading failed:', error);
    }
}

function displayContentRecommendations(content) {
    const container = document.getElementById('content-recommendations');
    if (!container) return;
    
    container.innerHTML = '';
    
    content.slice(0, 6).forEach(item => {
        const card = createContentCard(item);
        container.appendChild(card);
    });
}

function createContentCard(content) {
    const card = document.createElement('div');
    card.className = 'content-card';
    card.innerHTML = `
        <div class="content-header">
            <div class="content-type">
                <i class="fas fa-${getContentIcon(content.content_type)}"></i>
                <span>${content.content_type}</span>
            </div>
            <div class="content-difficulty">${content.difficulty_level}</div>
        </div>
        <h4 class="content-title">${content.title}</h4>
        <p class="content-description">${content.description}</p>
        <div class="content-tags">
            ${(content.style_tags || '').split(',').map(tag => 
                `<span class="style-tag">${tag.trim()}</span>`
            ).join('')}
        </div>
        <div class="content-actions">
            <button class="btn btn-primary btn-sm" onclick="startContent('${content.id}')">
                <i class="fas fa-play"></i> Start
            </button>
            <button class="btn btn-outline-secondary btn-sm" onclick="saveContent('${content.id}')">
                <i class="fas fa-bookmark"></i> Save
            </button>
        </div>
    `;
    
    return card;
}

function getContentIcon(type) {
    const icons = {
        'video': 'play-circle',
        'audio': 'headphones',
        'text': 'file-text',
        'interactive': 'mouse-pointer',
        'quiz': 'question-circle'
    };
    return icons[type] || 'file';
}

// Content Actions
function startContent(contentId) {
    // Navigate to content player or open in modal
    window.location.href = `/content/${contentId}`;
}

function saveContent(contentId) {
    // Save content to user's library
    fetch('/api/save-content', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content_id: contentId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('Content saved to your library!', 'success');
        }
    })
    .catch(error => {
        showNotification('Failed to save content', 'error');
    });
}

// Notification System
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${type === 'success' ? 'check-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Add notification styles
const notificationStyles = `
<style>
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    padding: 1rem;
    z-index: 1000;
    display: flex;
    align-items: center;
    gap: 1rem;
    animation: slideInRight 0.3s ease;
}

.notification-success {
    border-left: 4px solid #28a745;
}

.notification-error {
    border-left: 4px solid #dc3545;
}

.notification-info {
    border-left: 4px solid #17a2b8;
}

.notification-content {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.notification-close {
    background: none;
    border: none;
    cursor: pointer;
    color: #666;
}

@keyframes slideInRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}
</style>
`;

document.head.insertAdjacentHTML('beforeend', notificationStyles);
