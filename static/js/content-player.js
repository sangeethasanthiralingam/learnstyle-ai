// Content Player - Vanilla JavaScript
class ContentPlayer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.mediaElement = null;
        this.isPlaying = false;
        this.currentTime = 0;
        this.duration = 0;
        this.volume = 1;
        this.playbackRate = 1;
        this.progressInterval = null;
        this.timeSpent = 0;
        this.startTime = null;
        
        this.init();
    }
    
    init() {
        this.createPlayer();
        this.bindEvents();
        this.startTimeTracking();
    }
    
    createPlayer() {
        this.container.innerHTML = `
            <div class="content-player">
                <div class="player-header">
                    <div class="content-info">
                        <h3 class="content-title" id="content-title">Loading...</h3>
                        <div class="content-meta">
                            <span class="content-type" id="content-type">
                                <i class="fas fa-file"></i> <span>Loading...</span>
                            </span>
                            <span class="content-difficulty" id="content-difficulty">-</span>
                            <span class="content-duration" id="content-duration">00:00</span>
                        </div>
                    </div>
                    <div class="player-actions">
                        <button class="btn btn-outline-secondary btn-sm" id="bookmark-btn">
                            <i class="fas fa-bookmark"></i>
                        </button>
                        <button class="btn btn-outline-secondary btn-sm" id="share-btn">
                            <i class="fas fa-share"></i>
                        </button>
                    </div>
                </div>
                
                <div class="player-content">
                    <div class="media-container" id="media-container">
                        <div class="loading-spinner">
                            <i class="fas fa-spinner fa-spin"></i>
                            <p>Loading content...</p>
                        </div>
                    </div>
                    
                    <div class="player-controls">
                        <button class="control-btn play-pause" id="play-pause-btn">
                            <i class="fas fa-play"></i>
                        </button>
                        
                        <div class="progress-container">
                            <div class="progress-bar" id="progress-bar">
                                <div class="progress-fill" id="progress-fill"></div>
                                <div class="progress-handle" id="progress-handle"></div>
                            </div>
                            <div class="time-display">
                                <span id="current-time">00:00</span>
                                <span id="total-time">00:00</span>
                            </div>
                        </div>
                        
                        <div class="volume-control">
                            <i class="fas fa-volume-up"></i>
                            <input type="range" id="volume-slider" min="0" max="1" step="0.1" value="1">
                        </div>
                        
                        <div class="playback-controls">
                            <button class="control-btn" id="speed-btn" title="Playback Speed">1x</button>
                            <button class="control-btn" id="fullscreen-btn" title="Fullscreen">
                                <i class="fas fa-expand"></i>
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="player-footer">
                    <div class="time-spent">
                        <i class="fas fa-clock"></i>
                        <span>Time spent: <span id="time-spent">00:00</span></span>
                    </div>
                    <div class="completion-status">
                        <span id="completion-status">Not started</span>
                    </div>
                </div>
            </div>
        `;
        
        this.loadStyles();
    }
    
    loadStyles() {
        const styles = `
            <style>
            .content-player {
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }
            
            .player-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                padding: 1.5rem;
                border-bottom: 1px solid #e9ecef;
            }
            
            .content-title {
                font-size: 1.5rem;
                font-weight: 600;
                color: #333;
                margin-bottom: 0.5rem;
            }
            
            .content-meta {
                display: flex;
                gap: 1rem;
                flex-wrap: wrap;
            }
            
            .content-meta span {
                display: flex;
                align-items: center;
                gap: 0.25rem;
                font-size: 0.9rem;
                color: #666;
                background: #f8f9fa;
                padding: 0.25rem 0.75rem;
                border-radius: 12px;
            }
            
            .player-actions {
                display: flex;
                gap: 0.5rem;
            }
            
            .player-content {
                position: relative;
                background: #000;
                min-height: 300px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .media-container {
                width: 100%;
                height: 100%;
                position: relative;
            }
            
            .loading-spinner {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 1.2rem;
            }
            
            .loading-spinner i {
                font-size: 2rem;
                margin-bottom: 1rem;
            }
            
            .player-controls {
                display: flex;
                align-items: center;
                gap: 1rem;
                padding: 1rem 1.5rem;
                background: #f8f9fa;
            }
            
            .control-btn {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                border: none;
                background: #667eea;
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
            }
            
            .control-btn:hover {
                background: #5a6fd8;
                transform: scale(1.05);
            }
            
            .progress-container {
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }
            
            .progress-bar {
                width: 100%;
                height: 6px;
                background: #e9ecef;
                border-radius: 3px;
                cursor: pointer;
                position: relative;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 3px;
                transition: width 0.1s ease;
            }
            
            .progress-handle {
                position: absolute;
                top: 50%;
                width: 16px;
                height: 16px;
                background: white;
                border: 2px solid #667eea;
                border-radius: 50%;
                transform: translate(-50%, -50%);
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            
            .progress-bar:hover .progress-handle {
                opacity: 1;
            }
            
            .time-display {
                display: flex;
                justify-content: space-between;
                font-size: 0.9rem;
                color: #666;
            }
            
            .volume-control {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .volume-control input {
                width: 80px;
            }
            
            .playback-controls {
                display: flex;
                gap: 0.5rem;
            }
            
            .player-footer {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1rem 1.5rem;
                background: #f8f9fa;
                border-top: 1px solid #e9ecef;
                font-size: 0.9rem;
                color: #666;
            }
            
            .time-spent {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .completion-status {
                font-weight: 600;
            }
            
            .completion-status.completed {
                color: #28a745;
            }
            
            .completion-status.in-progress {
                color: #ffc107;
            }
            
            @media (max-width: 768px) {
                .player-header {
                    flex-direction: column;
                    gap: 1rem;
                }
                
                .player-controls {
                    flex-wrap: wrap;
                    gap: 0.5rem;
                }
                
                .volume-control {
                    order: 3;
                    width: 100%;
                }
                
                .volume-control input {
                    width: 100%;
                }
            }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
    
    bindEvents() {
        // Play/Pause button
        document.getElementById('play-pause-btn').addEventListener('click', () => {
            this.togglePlayPause();
        });
        
        // Progress bar
        document.getElementById('progress-bar').addEventListener('click', (e) => {
            this.seekTo(e);
        });
        
        // Volume control
        document.getElementById('volume-slider').addEventListener('input', (e) => {
            this.setVolume(e.target.value);
        });
        
        // Speed control
        document.getElementById('speed-btn').addEventListener('click', () => {
            this.toggleSpeed();
        });
        
        // Fullscreen
        document.getElementById('fullscreen-btn').addEventListener('click', () => {
            this.toggleFullscreen();
        });
        
        // Bookmark
        document.getElementById('bookmark-btn').addEventListener('click', () => {
            this.toggleBookmark();
        });
        
        // Share
        document.getElementById('share-btn').addEventListener('click', () => {
            this.shareContent();
        });
    }
    
    loadContent(contentData) {
        this.contentData = contentData;
        
        // Update header
        document.getElementById('content-title').textContent = contentData.title;
        document.getElementById('content-type').innerHTML = `
            <i class="fas fa-${this.getContentIcon(contentData.content_type)}"></i>
            <span>${contentData.content_type}</span>
        `;
        document.getElementById('content-difficulty').textContent = contentData.difficulty_level;
        
        // Create media element
        this.createMediaElement(contentData);
        
        // Update duration
        if (contentData.duration) {
            document.getElementById('content-duration').textContent = this.formatTime(contentData.duration);
        }
    }
    
    createMediaElement(contentData) {
        const container = document.getElementById('media-container');
        container.innerHTML = '';
        
        let mediaElement;
        
        switch (contentData.content_type) {
            case 'video':
                mediaElement = document.createElement('video');
                mediaElement.src = contentData.url || '#';
                mediaElement.controls = false;
                mediaElement.style.width = '100%';
                mediaElement.style.height = 'auto';
                mediaElement.style.maxHeight = '500px';
                break;
                
            case 'audio':
                mediaElement = document.createElement('audio');
                mediaElement.src = contentData.url || '#';
                mediaElement.controls = false;
                break;
                
            case 'interactive':
                mediaElement = document.createElement('iframe');
                mediaElement.src = contentData.url || '#';
                mediaElement.style.width = '100%';
                mediaElement.style.height = '500px';
                mediaElement.style.border = 'none';
                break;
                
            default:
                // Text content
                mediaElement = document.createElement('div');
                mediaElement.className = 'text-content';
                mediaElement.style.padding = '2rem';
                mediaElement.style.background = '#f8f9fa';
                mediaElement.innerHTML = `
                    <div style="font-size: 1.1rem; line-height: 1.6; color: #333;">
                        ${contentData.content || contentData.description || 'No content available'}
                    </div>
                `;
                break;
        }
        
        if (mediaElement) {
            container.appendChild(mediaElement);
            this.mediaElement = mediaElement;
            this.setupMediaEvents();
        }
    }
    
    setupMediaEvents() {
        if (!this.mediaElement) return;
        
        this.mediaElement.addEventListener('loadedmetadata', () => {
            this.duration = this.mediaElement.duration || 0;
            document.getElementById('total-time').textContent = this.formatTime(this.duration);
        });
        
        this.mediaElement.addEventListener('timeupdate', () => {
            this.currentTime = this.mediaElement.currentTime || 0;
            this.updateProgress();
        });
        
        this.mediaElement.addEventListener('ended', () => {
            this.onContentComplete();
        });
        
        this.mediaElement.addEventListener('play', () => {
            this.isPlaying = true;
            this.updatePlayButton();
        });
        
        this.mediaElement.addEventListener('pause', () => {
            this.isPlaying = false;
            this.updatePlayButton();
        });
    }
    
    togglePlayPause() {
        if (!this.mediaElement) return;
        
        if (this.isPlaying) {
            this.mediaElement.pause();
        } else {
            this.mediaElement.play();
        }
    }
    
    updatePlayButton() {
        const btn = document.getElementById('play-pause-btn');
        const icon = btn.querySelector('i');
        
        if (this.isPlaying) {
            icon.className = 'fas fa-pause';
        } else {
            icon.className = 'fas fa-play';
        }
    }
    
    seekTo(event) {
        if (!this.mediaElement) return;
        
        const progressBar = event.currentTarget;
        const rect = progressBar.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const percentage = clickX / rect.width;
        const newTime = percentage * this.duration;
        
        this.mediaElement.currentTime = newTime;
    }
    
    updateProgress() {
        const percentage = this.duration > 0 ? (this.currentTime / this.duration) * 100 : 0;
        document.getElementById('progress-fill').style.width = percentage + '%';
        document.getElementById('current-time').textContent = this.formatTime(this.currentTime);
        
        // Update completion status
        if (percentage > 90) {
            document.getElementById('completion-status').textContent = 'Almost complete';
            document.getElementById('completion-status').className = 'completion-status in-progress';
        }
    }
    
    setVolume(value) {
        this.volume = parseFloat(value);
        if (this.mediaElement) {
            this.mediaElement.volume = this.volume;
        }
    }
    
    toggleSpeed() {
        const speeds = [0.5, 0.75, 1, 1.25, 1.5, 2];
        const currentIndex = speeds.indexOf(this.playbackRate);
        const nextIndex = (currentIndex + 1) % speeds.length;
        
        this.playbackRate = speeds[nextIndex];
        
        if (this.mediaElement) {
            this.mediaElement.playbackRate = this.playbackRate;
        }
        
        document.getElementById('speed-btn').textContent = this.playbackRate + 'x';
    }
    
    toggleFullscreen() {
        if (!document.fullscreenElement) {
            this.container.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }
    
    toggleBookmark() {
        // Toggle bookmark state
        const btn = document.getElementById('bookmark-btn');
        const icon = btn.querySelector('i');
        
        if (icon.classList.contains('fas')) {
            icon.className = 'far fa-bookmark';
            btn.classList.add('bookmarked');
            this.showNotification('Content bookmarked!', 'success');
        } else {
            icon.className = 'fas fa-bookmark';
            btn.classList.remove('bookmarked');
            this.showNotification('Bookmark removed', 'info');
        }
    }
    
    shareContent() {
        if (navigator.share) {
            navigator.share({
                title: this.contentData.title,
                text: 'Check out this learning content',
                url: window.location.href
            });
        } else {
            // Fallback: copy to clipboard
            navigator.clipboard.writeText(window.location.href).then(() => {
                this.showNotification('Link copied to clipboard!', 'success');
            });
        }
    }
    
    startTimeTracking() {
        this.startTime = Date.now();
        
        setInterval(() => {
            if (this.isPlaying) {
                this.timeSpent++;
                document.getElementById('time-spent').textContent = this.formatTime(this.timeSpent);
            }
        }, 1000);
    }
    
    onContentComplete() {
        this.isPlaying = false;
        this.updatePlayButton();
        
        document.getElementById('completion-status').textContent = 'Completed!';
        document.getElementById('completion-status').className = 'completion-status completed';
        
        // Show rating modal
        this.showRatingModal();
        
        // Save progress
        this.saveProgress();
    }
    
    showRatingModal() {
        const modal = document.createElement('div');
        modal.className = 'rating-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <h3>Rate this content</h3>
                <p>How would you rate your learning experience?</p>
                <div class="star-rating">
                    ${[1, 2, 3, 4, 5].map(star => `
                        <button class="star" data-rating="${star}">
                            <i class="fas fa-star"></i>
                        </button>
                    `).join('')}
                </div>
                <div class="modal-actions">
                    <button class="btn btn-outline-secondary" onclick="this.closest('.rating-modal').remove()">Skip</button>
                    <button class="btn btn-primary" id="submit-rating" disabled>Submit Rating</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Handle star rating
        modal.querySelectorAll('.star').forEach(star => {
            star.addEventListener('click', () => {
                const rating = parseInt(star.dataset.rating);
                this.setRating(rating);
                
                modal.querySelectorAll('.star').forEach((s, index) => {
                    if (index < rating) {
                        s.classList.add('active');
                    } else {
                        s.classList.remove('active');
                    }
                });
                
                document.getElementById('submit-rating').disabled = false;
            });
        });
        
        // Submit rating
        document.getElementById('submit-rating').addEventListener('click', () => {
            this.submitRating();
            modal.remove();
        });
    }
    
    setRating(rating) {
        this.rating = rating;
    }
    
    submitRating() {
        if (!this.rating) return;
        
        fetch('/api/progress', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                content_id: this.contentData.id,
                completion_status: 'completed',
                time_spent: this.timeSpent,
                score: this.rating * 20, // Convert 1-5 rating to 0-100 score
                engagement_rating: this.rating
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                this.showNotification('Rating submitted successfully!', 'success');
            }
        })
        .catch(error => {
            this.showNotification('Failed to submit rating', 'error');
        });
    }
    
    saveProgress() {
        fetch('/api/progress', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                content_id: this.contentData.id,
                completion_status: 'completed',
                time_spent: this.timeSpent,
                progress: 100
            })
        })
        .catch(error => {
            console.log('Progress save failed:', error);
        });
    }
    
    getContentIcon(type) {
        const icons = {
            'video': 'play-circle',
            'audio': 'headphones',
            'text': 'file-text',
            'interactive': 'mouse-pointer',
            'quiz': 'question-circle'
        };
        return icons[type] || 'file';
    }
    
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    showNotification(message, type = 'info') {
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
        
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }
}

// Initialize content player when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    const playerContainer = document.getElementById('content-player-container');
    if (playerContainer) {
        window.contentPlayer = new ContentPlayer('content-player-container');
    }
});
