/**
 * Manual Learning Site Tracking
 * Allows users to manually track their learning activities
 */

class ManualLearningTracker {
    constructor() {
        this.currentSession = null;
        this.init();
    }

    init() {
        this.createTrackingInterface();
        this.loadRecentSites();
    }

    createTrackingInterface() {
        // Create floating tracking button
        const trackerButton = document.createElement('div');
        trackerButton.id = 'manual-tracker-button';
        trackerButton.innerHTML = `
            <div class="tracker-button" onclick="toggleTracker()">
                <i class="fas fa-plus"></i>
            </div>
        `;
        
        trackerButton.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 10000;
        `;

        document.body.appendChild(trackerButton);

        // Add CSS styles
        const style = document.createElement('style');
        style.textContent = `
            .tracker-button {
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 24px;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                transition: all 0.3s ease;
            }
            
            .tracker-button:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 16px rgba(0,0,0,0.4);
            }
            
            .tracker-modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.5);
                display: none;
                align-items: center;
                justify-content: center;
                z-index: 10001;
            }
            
            .tracker-content {
                background: white;
                border-radius: 15px;
                padding: 2rem;
                max-width: 500px;
                width: 90%;
                max-height: 80vh;
                overflow-y: auto;
            }
            
            .tracker-form {
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }
            
            .form-group {
                display: flex;
                flex-direction: column;
            }
            
            .form-group label {
                font-weight: 600;
                margin-bottom: 0.5rem;
                color: #2c3e50;
            }
            
            .form-group input, .form-group select, .form-group textarea {
                padding: 0.75rem;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                font-size: 1rem;
                transition: border-color 0.3s ease;
            }
            
            .form-group input:focus, .form-group select:focus, .form-group textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            
            .btn-track {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .btn-track:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }
            
            .form-actions {
                display: flex;
                gap: 1rem;
                justify-content: flex-end;
                margin-top: 1rem;
            }
            
            .btn-cancel {
                background: #6c757d;
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .btn-cancel:hover {
                background: #5a6268;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(108, 117, 125, 0.3);
            }
            
            .recent-sites {
                margin-top: 2rem;
            }
            
            .site-item {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 1rem;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                margin-bottom: 0.5rem;
            }
            
            .site-info {
                display: flex;
                align-items: center;
                gap: 1rem;
            }
            
            .site-icon {
                width: 40px;
                height: 40px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
            }
            
            .close-modal {
                position: absolute;
                top: 1rem;
                right: 1rem;
                background: none;
                border: none;
                font-size: 1.5rem;
                cursor: pointer;
                color: #6c757d;
            }
        `;

        document.head.appendChild(style);

        // Create modal
        const modal = document.createElement('div');
        modal.id = 'tracker-modal';
        modal.className = 'tracker-modal';
        modal.innerHTML = `
            <div class="tracker-content">
                <button class="close-modal" onclick="closeTracker()">&times;</button>
                <h3><i class="fas fa-plus-circle"></i> Track Learning Activity</h3>
                <form class="tracker-form" onsubmit="submitTracking(event)">
                    <div class="form-group">
                        <label for="site-url">Website URL</label>
                        <input type="url" id="site-url" placeholder="https://youtube.com/watch?v=..." required>
                    </div>
                    <div class="form-group">
                        <label for="site-name">Site Name</label>
                        <input type="text" id="site-name" placeholder="YouTube" required>
                    </div>
                    <div class="form-group">
                        <label for="activity-type">Activity Type</label>
                        <select id="activity-type" required>
                            <option value="visit">Visit</option>
                            <option value="study">Study Session</option>
                            <option value="video">Video Learning</option>
                            <option value="quiz">Quiz/Assessment</option>
                            <option value="research">Research</option>
                            <option value="coding">Coding Practice</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="time-spent">Time Spent (minutes)</label>
                        <input type="number" id="time-spent" placeholder="30" min="1" required>
                    </div>
                    <div class="form-group">
                        <label for="content-type">Content Type</label>
                        <select id="content-type" required>
                            <option value="video">Video</option>
                            <option value="article">Article</option>
                            <option value="interactive">Interactive</option>
                            <option value="quiz">Quiz</option>
                            <option value="coding">Coding</option>
                            <option value="research">Research</option>
                            <option value="general">General</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="notes">Notes (Optional)</label>
                        <textarea id="notes" placeholder="Add any notes about what you learned, key takeaways, or thoughts..."></textarea>
                    </div>
                    <div class="form-actions">
                        <button type="button" class="btn-cancel" onclick="closeTracker()">
                            <i class="fas fa-times"></i> Cancel
                        </button>
                        <button type="submit" class="btn-track">
                            <i class="fas fa-save"></i> Track Activity
                        </button>
                    </div>
                </form>
                <div class="recent-sites">
                    <h4>Recent Sites</h4>
                    <div id="recent-sites-list"></div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
    }

    async loadRecentSites() {
        try {
            const response = await fetch('/api/learning-sites?per_page=5');
            if (response.ok) {
                const data = await response.json();
                this.displayRecentSites(data.activities);
            }
        } catch (error) {
            console.error('Error loading recent sites:', error);
        }
    }

    displayRecentSites(sites) {
        const container = document.getElementById('recent-sites-list');
        if (!container) return;

        if (sites.length === 0) {
            container.innerHTML = '<p>No recent learning activities found.</p>';
            return;
        }

        container.innerHTML = sites.map(site => `
            <div class="site-item">
                <div class="site-info">
                    <div class="site-icon" style="background: ${this.getSiteColor(site.site_url)};">
                        ${this.getSiteIcon(site.site_url)}
                    </div>
                    <div style="flex: 1;">
                        <div><strong>${site.site_name}</strong></div>
                        <div style="font-size: 0.9rem; color: #6c757d;">
                            ${site.activity_type} • ${Math.floor(site.time_spent / 60)} min
                        </div>
                        ${site.notes ? `<div style="font-size: 0.8rem; color: #495057; margin-top: 0.25rem; font-style: italic;">
                            "${site.notes.substring(0, 50)}${site.notes.length > 50 ? '...' : ''}"
                        </div>` : ''}
                    </div>
                </div>
                <button onclick="quickTrack('${site.site_url}', '${site.site_name}', '${site.notes || ''}')" 
                        style="background: none; border: none; color: #667eea; cursor: pointer; margin-left: 1rem;">
                    <i class="fas fa-plus"></i>
                </button>
            </div>
        `).join('');
    }

    getSiteColor(url) {
        if (url.includes('youtube.com')) return '#ff0000';
        if (url.includes('coursera.org')) return '#0056b3';
        if (url.includes('khanacademy.org')) return '#14bf96';
        if (url.includes('github.com')) return '#333';
        if (url.includes('stackoverflow.com')) return '#f48024';
        return '#667eea';
    }

    getSiteIcon(url) {
        if (url.includes('youtube.com')) return '📺';
        if (url.includes('coursera.org')) return '🎓';
        if (url.includes('khanacademy.org')) return '🏫';
        if (url.includes('github.com')) return '💻';
        if (url.includes('stackoverflow.com')) return '❓';
        return '🌐';
    }
}

// Global functions for the tracker
function toggleTracker() {
    const modal = document.getElementById('tracker-modal');
    modal.style.display = 'flex';
}

function closeTracker() {
    const modal = document.getElementById('tracker-modal');
    modal.style.display = 'none';
}

async function submitTracking(event) {
    event.preventDefault();
    
    const formData = {
        url: document.getElementById('site-url').value,
        name: document.getElementById('site-name').value,
        activity_type: document.getElementById('activity-type').value,
        time_spent: parseInt(document.getElementById('time-spent').value) * 60, // Convert to seconds
        content_type: document.getElementById('content-type').value,
        notes: document.getElementById('notes').value
    };

    try {
        const response = await fetch('/api/learning-sites', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        if (response.ok) {
            showNotification('Learning activity tracked successfully!', 'success');
            closeTracker();
            // Reload recent sites
            if (window.manualTracker) {
                window.manualTracker.loadRecentSites();
            }
        } else {
            const error = await response.json();
            showNotification(error.error || 'Failed to track activity', 'error');
        }
    } catch (error) {
        showNotification('Error tracking activity', 'error');
    }
}

function quickTrack(url, name, notes = '') {
    document.getElementById('site-url').value = url;
    document.getElementById('site-name').value = name;
    document.getElementById('notes').value = notes;
    
    // Auto-detect content type
    if (url.includes('youtube.com')) {
        document.getElementById('content-type').value = 'video';
        document.getElementById('activity-type').value = 'video';
    } else if (url.includes('github.com')) {
        document.getElementById('content-type').value = 'coding';
        document.getElementById('activity-type').value = 'coding';
    }
}

function showNotification(message, type) {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'success' ? 'success' : 'danger'} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Initialize the manual tracker
document.addEventListener('DOMContentLoaded', () => {
    window.manualTracker = new ManualLearningTracker();
});
