/**
 * LearnStyle AI - Learning Tracker
 * Tracks learning activities on educational websites
 */

class LearnStyleTracker {
    constructor() {
        this.isTracking = false;
        this.startTime = null;
        this.currentUrl = window.location.href;
        this.currentTitle = document.title;
        this.apiEndpoint = 'http://localhost:5000/api/learning-sites';
        this.sessionId = this.generateSessionId();
        
        this.init();
    }

    init() {
        // Check if this is a learning site
        if (!this.isLearningSite()) {
            console.log('LearnStyle AI: Not a learning site, skipping tracking');
            return;
        }

        // Listen for tracking messages
        window.addEventListener('message', (event) => {
            if (event.data.action === 'startLearnStyleTracking') {
                this.startTracking();
            } else if (event.data.action === 'stopLearnStyleTracking') {
                this.stopTracking();
            } else if (event.data.action === 'pauseTracking') {
                this.pauseTracking();
            } else if (event.data.action === 'resumeTracking') {
                this.resumeTracking();
            }
        });

        // Auto-start tracking for YouTube videos
        if (this.isYouTubeVideo()) {
            this.startTracking();
        }

        // Track page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseTracking();
            } else {
                this.resumeTracking();
            }
        });

        // Track before page unload
        window.addEventListener('beforeunload', () => {
            this.stopTracking();
        });

        console.log('LearnStyle AI: Tracker initialized on', window.location.hostname);
    }

    isLearningSite() {
        const learningSites = [
            'coursera.org', 'khanacademy.org', 'edx.org', 'udemy.com',
            'github.com', 'stackoverflow.com', 'youtube.com', 'scholar.google.com',
            'researchgate.net', 'academia.edu', 'arxiv.org', 'ieee.org'
        ];
        
        return learningSites.some(site => window.location.hostname.includes(site));
    }

    isYouTubeVideo() {
        return window.location.hostname.includes('youtube.com') && 
               window.location.pathname.includes('/watch');
    }

    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    startTracking() {
        if (this.isTracking) return;
        
        this.isTracking = true;
        this.startTime = Date.now();
        
        console.log('LearnStyle AI: Started tracking on', this.currentUrl);
        
        // Notify that tracking started
        window.postMessage({ action: 'trackingStarted' }, '*');
        
        // Send initial tracking data
        this.sendTrackingData('visit');
    }

    stopTracking() {
        if (!this.isTracking) return;
        
        const timeSpent = Date.now() - this.startTime;
        
        console.log('LearnStyle AI: Stopped tracking, time spent:', timeSpent + 'ms');
        
        // Send final tracking data
        this.sendTrackingData('leave', timeSpent);
        
        this.isTracking = false;
        this.startTime = null;
        
        // Notify that tracking stopped
        window.postMessage({ action: 'trackingStopped' }, '*');
    }

    pauseTracking() {
        if (!this.isTracking) return;
        
        console.log('LearnStyle AI: Paused tracking');
        // Could send pause event here if needed
    }

    resumeTracking() {
        if (!this.isTracking) return;
        
        console.log('LearnStyle AI: Resumed tracking');
        // Could send resume event here if needed
    }

    async sendTrackingData(activityType, timeSpent = 0) {
        try {
            // Get user token from storage
            const userToken = await this.getUserToken();
            
            const data = {
                url: this.currentUrl,
                name: this.getSiteName(),
                activity_type: activityType,
                time_spent: Math.floor(timeSpent / 1000), // Convert to seconds
                content_type: this.getContentType(),
                user_token: userToken
            };

            console.log('LearnStyle AI: Sending tracking data:', data);

            const response = await fetch(this.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (response.ok) {
                const result = await response.json();
                console.log('LearnStyle AI: Activity tracked successfully', result);
            } else {
                console.error('LearnStyle AI: Failed to track activity', response.status, await response.text());
            }
        } catch (error) {
            console.error('LearnStyle AI: Error tracking activity', error);
        }
    }

    async getUserToken() {
        try {
            // Try to get token from extension storage
            if (typeof chrome !== 'undefined' && chrome.storage) {
                return new Promise((resolve) => {
                    chrome.storage.local.get(['userToken'], (result) => {
                        resolve(result.userToken || null);
                    });
                });
            }
            return null;
        } catch (error) {
            console.error('LearnStyle AI: Error getting user token', error);
            return null;
        }
    }

    async getAuthData() {
        try {
            // Try to get auth data from extension storage
            if (typeof chrome !== 'undefined' && chrome.storage) {
                return new Promise((resolve) => {
                    chrome.storage.local.get(['authData'], (result) => {
                        resolve(result.authData);
                    });
                });
            }
            return null;
        } catch (error) {
            console.error('LearnStyle AI: Error getting auth data', error);
            return null;
        }
    }

    getSiteName() {
        const hostname = window.location.hostname;
        
        if (hostname.includes('youtube.com')) {
            return 'YouTube';
        } else if (hostname.includes('coursera.org')) {
            return 'Coursera';
        } else if (hostname.includes('khanacademy.org')) {
            return 'Khan Academy';
        } else if (hostname.includes('edx.org')) {
            return 'edX';
        } else if (hostname.includes('udemy.com')) {
            return 'Udemy';
        } else if (hostname.includes('github.com')) {
            return 'GitHub';
        } else if (hostname.includes('stackoverflow.com')) {
            return 'Stack Overflow';
        } else if (hostname.includes('scholar.google.com')) {
            return 'Google Scholar';
        } else if (hostname.includes('researchgate.net')) {
            return 'ResearchGate';
        }
        
        return hostname;
    }

    getContentType() {
        if (this.isYouTubeVideo()) {
            return 'video';
        } else if (window.location.hostname.includes('github.com')) {
            return 'coding';
        } else if (window.location.hostname.includes('scholar.google.com') || 
                   window.location.hostname.includes('researchgate.net')) {
            return 'research';
        } else if (window.location.hostname.includes('coursera.org') || 
                   window.location.hostname.includes('khanacademy.org') ||
                   window.location.hostname.includes('edx.org')) {
            return 'education';
        }
        
        return 'general';
    }
}

// Initialize the tracker
const tracker = new LearnStyleTracker();
