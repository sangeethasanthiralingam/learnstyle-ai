/**
 * LearnStyle AI - Learning Site Tracker
 * Browser extension script for tracking learning activities
 */

class LearningTracker {
    constructor() {
        this.userId = null;
        this.permissions = {};
        this.currentSite = null;
        this.startTime = Date.now();
        this.isActive = false;
        
        this.init();
    }
    
    async init() {
        // Get user ID from localStorage or prompt for login
        this.userId = localStorage.getItem('learnstyle_user_id');
        if (!this.userId) {
            this.showLoginPrompt();
            return;
        }
        
        // Load user permissions
        await this.loadPermissions();
        
        // Start tracking if permissions allow
        if (this.hasTrackingPermission()) {
            this.startTracking();
        }
    }
    
    async loadPermissions() {
        try {
            const response = await fetch('/api/permissions', {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('learnstyle_token')}`
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.permissions = data.permissions;
            }
        } catch (error) {
            console.error('Error loading permissions:', error);
        }
    }
    
    hasTrackingPermission() {
        const hostname = window.location.hostname.toLowerCase();
        
        // Check educational sites
        if (this.isEducationalSite(hostname)) {
            return this.permissions.eduSites;
        }
        
        // Check coding sites
        if (this.isCodingSite(hostname)) {
            return this.permissions.coding;
        }
        
        // Check video sites
        if (this.isVideoSite(hostname)) {
            return this.permissions.video;
        }
        
        // Check research sites
        if (this.isResearchSite(hostname)) {
            return this.permissions.research;
        }
        
        return false;
    }
    
    isEducationalSite(hostname) {
        const eduSites = [
            'coursera.org', 'khanacademy.org', 'edx.org', 'udemy.com',
            'udacity.com', 'coursera.com', 'khanacademy.com', 'edx.com',
            'skillshare.com', 'pluralsight.com', 'lynda.com', 'linkedin.com'
        ];
        return eduSites.some(site => hostname.includes(site));
    }
    
    isCodingSite(hostname) {
        const codingSites = [
            'github.com', 'stackoverflow.com', 'leetcode.com', 'hackerrank.com',
            'codewars.com', 'freecodecamp.org', 'codecademy.com', 'w3schools.com',
            'geeksforgeeks.org', 'programiz.com', 'tutorialspoint.com'
        ];
        return codingSites.some(site => hostname.includes(site));
    }
    
    isVideoSite(hostname) {
        const videoSites = [
            'youtube.com', 'vimeo.com', 'ted.com', 'khanacademy.org',
            'coursera.org', 'edx.org', 'udemy.com'
        ];
        return videoSites.some(site => hostname.includes(site));
    }
    
    isResearchSite(hostname) {
        const researchSites = [
            'scholar.google.com', 'researchgate.net', 'academia.edu',
            'arxiv.org', 'ieee.org', 'acm.org', 'springer.com',
            'elsevier.com', 'nature.com', 'science.org'
        ];
        return researchSites.some(site => hostname.includes(site));
    }
    
    startTracking() {
        this.isActive = true;
        this.currentSite = {
            url: window.location.href,
            name: this.getSiteName(),
            startTime: Date.now()
        };
        
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
        
        // Track scroll and interaction
        this.trackInteractions();
        
        console.log('LearnStyle AI: Started tracking learning activity');
    }
    
    pauseTracking() {
        if (this.isActive && this.currentSite) {
            this.currentSite.pausedTime = Date.now();
        }
    }
    
    resumeTracking() {
        if (this.isActive && this.currentSite && this.currentSite.pausedTime) {
            const pauseDuration = Date.now() - this.currentSite.pausedTime;
            this.currentSite.startTime += pauseDuration;
            delete this.currentSite.pausedTime;
        }
    }
    
    stopTracking() {
        if (this.isActive && this.currentSite) {
            const timeSpent = Math.floor((Date.now() - this.currentSite.startTime) / 1000);
            
            this.trackActivity({
                url: this.currentSite.url,
                name: this.currentSite.name,
                activity_type: this.detectActivityType(),
                time_spent: timeSpent,
                content_type: this.detectContentType()
            });
            
            this.isActive = false;
            this.currentSite = null;
        }
    }
    
    detectActivityType() {
        const path = window.location.pathname.toLowerCase();
        
        if (path.includes('video') || path.includes('watch')) {
            return 'video';
        } else if (path.includes('quiz') || path.includes('test') || path.includes('exam')) {
            return 'quiz';
        } else if (path.includes('assignment') || path.includes('homework')) {
            return 'assignment';
        } else if (path.includes('discussion') || path.includes('forum')) {
            return 'discussion';
        } else if (path.includes('lecture') || path.includes('lesson')) {
            return 'lecture';
        } else {
            return 'study';
        }
    }
    
    detectContentType() {
        const hostname = window.location.hostname.toLowerCase();
        
        if (this.isEducationalSite(hostname)) {
            return 'educational';
        } else if (this.isCodingSite(hostname)) {
            return 'programming';
        } else if (this.isVideoSite(hostname)) {
            return 'video';
        } else if (this.isResearchSite(hostname)) {
            return 'research';
        } else {
            return 'general';
        }
    }
    
    getSiteName() {
        const hostname = window.location.hostname;
        const title = document.title;
        
        // Extract site name from hostname
        const siteNames = {
            'coursera.org': 'Coursera',
            'khanacademy.org': 'Khan Academy',
            'edx.org': 'edX',
            'udemy.com': 'Udemy',
            'github.com': 'GitHub',
            'stackoverflow.com': 'Stack Overflow',
            'youtube.com': 'YouTube',
            'scholar.google.com': 'Google Scholar'
        };
        
        return siteNames[hostname] || hostname.replace('www.', '');
    }
    
    trackInteractions() {
        let interactionCount = 0;
        let scrollDistance = 0;
        let lastScrollTop = 0;
        
        // Track clicks
        document.addEventListener('click', () => {
            interactionCount++;
        });
        
        // Track scrolling
        window.addEventListener('scroll', () => {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            scrollDistance += Math.abs(scrollTop - lastScrollTop);
            lastScrollTop = scrollTop;
        });
        
        // Track keyboard interactions
        document.addEventListener('keydown', () => {
            interactionCount++;
        });
        
        // Send interaction data periodically
        setInterval(() => {
            if (this.isActive) {
                this.trackEngagement({
                    interactions: interactionCount,
                    scroll_distance: scrollDistance,
                    time_on_page: Math.floor((Date.now() - this.currentSite.startTime) / 1000)
                });
                
                // Reset counters
                interactionCount = 0;
                scrollDistance = 0;
            }
        }, 30000); // Every 30 seconds
    }
    
    async trackActivity(activityData) {
        try {
            const response = await fetch('/api/learning-sites', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('learnstyle_token')}`
                },
                body: JSON.stringify(activityData)
            });
            
            if (response.ok) {
                console.log('LearnStyle AI: Activity tracked successfully');
            } else {
                console.error('LearnStyle AI: Failed to track activity');
            }
        } catch (error) {
            console.error('LearnStyle AI: Error tracking activity:', error);
        }
    }
    
    async trackEngagement(engagementData) {
        try {
            const response = await fetch('/api/engagement', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('learnstyle_token')}`
                },
                body: JSON.stringify({
                    content_id: 1, // Default for external sites
                    interaction_time: engagementData.time_on_page,
                    completion_rate: this.calculateCompletionRate(),
                    click_frequency: engagementData.interactions / (engagementData.time_on_page / 60),
                    scroll_velocity: engagementData.scroll_distance / engagementData.time_on_page,
                    pause_duration: 0,
                    context: this.detectContentType()
                })
            });
            
            if (response.ok) {
                console.log('LearnStyle AI: Engagement tracked successfully');
            }
        } catch (error) {
            console.error('LearnStyle AI: Error tracking engagement:', error);
        }
    }
    
    calculateCompletionRate() {
        // Simple completion rate based on scroll position
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const documentHeight = document.documentElement.scrollHeight - window.innerHeight;
        
        if (documentHeight <= 0) return 1.0;
        
        return Math.min(scrollTop / documentHeight, 1.0);
    }
    
    showLoginPrompt() {
        // Create a simple login prompt
        const prompt = document.createElement('div');
        prompt.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #007bff;
            color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10000;
            font-family: Arial, sans-serif;
            max-width: 300px;
        `;
        
        prompt.innerHTML = `
            <h4 style="margin: 0 0 10px 0;">LearnStyle AI</h4>
            <p style="margin: 0 0 15px 0;">Track your learning activities for personalized insights!</p>
            <button onclick="window.open('http://localhost:5000/login', '_blank')" 
                    style="background: white; color: #007bff; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                Login to LearnStyle AI
            </button>
            <button onclick="this.parentElement.remove()" 
                    style="background: transparent; color: white; border: 1px solid white; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-left: 10px;">
                Dismiss
            </button>
        `;
        
        document.body.appendChild(prompt);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (prompt.parentElement) {
                prompt.remove();
            }
        }, 10000);
    }
}

// Initialize the tracker when the page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new LearningTracker();
    });
} else {
    new LearningTracker();
}

// Export for use in browser extensions
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LearningTracker;
}
