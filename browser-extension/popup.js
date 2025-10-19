/**
 * LearnStyle AI - Browser Extension Popup Script
 */

let isTracking = false;
let startTime = null;
let timeInterval = null;

// Initialize popup
document.addEventListener('DOMContentLoaded', function() {
    loadPageInfo();
    loadUserStats();
    setupEventListeners();
    
    // Update time display every second
    timeInterval = setInterval(updateTimeDisplay, 1000);
});

// Load current page information
async function loadPageInfo() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        if (tab) {
            const response = await chrome.tabs.sendMessage(tab.id, { action: 'getPageInfo' });
            
            if (response) {
                document.getElementById('current-site').textContent = response.title || 'Unknown Site';
                
                // Check if it's a learning site
                if (response.isLearningSite) {
                    document.getElementById('current-site').classList.add('status-active');
                } else {
                    document.getElementById('current-site').classList.add('status-inactive');
                }
            }
        }
    } catch (error) {
        console.error('Error loading page info:', error);
        document.getElementById('current-site').textContent = 'Error loading';
    }
    
    // Hide loading and show content
    document.getElementById('loading').style.display = 'none';
    document.getElementById('content').style.display = 'block';
}

// Load user statistics
async function loadUserStats() {
    try {
        // Get user token
        const userToken = await getUserToken();
        const url = userToken 
            ? `http://localhost:5000/api/learning-sites?per_page=5&user_token=${userToken}`
            : 'http://localhost:5000/api/learning-sites?per_page=5';
        
        const response = await fetch(url);
        if (response.ok) {
            const data = await response.json();
            const activities = data.activities || [];
            
            // Calculate stats
            const sitesVisited = activities.length;
            const totalTime = activities.reduce((sum, activity) => sum + (activity.time_spent || 0), 0);
            const learningScore = Math.min(100, Math.floor((totalTime / 60) * 2)); // Simple scoring
            
            document.getElementById('sites-visited').textContent = sitesVisited;
            document.getElementById('total-time').textContent = formatTime(totalTime);
            document.getElementById('learning-score').textContent = learningScore + '%';
        } else {
            // Fallback to stored stats
            const result = await chrome.storage.local.get(['dailyStats']);
            const stats = result.dailyStats || {
                sitesVisited: 0,
                totalTime: 0,
                learningScore: 0
            };
            
            document.getElementById('sites-visited').textContent = stats.sitesVisited;
            document.getElementById('total-time').textContent = formatTime(stats.totalTime);
            document.getElementById('learning-score').textContent = stats.learningScore + '%';
        }
    } catch (error) {
        console.error('Error loading user stats:', error);
        // Show default values
        document.getElementById('sites-visited').textContent = '0';
        document.getElementById('total-time').textContent = '0:00';
        document.getElementById('learning-score').textContent = '0%';
    }
}

// Get user token from storage
async function getUserToken() {
    try {
        const result = await chrome.storage.local.get(['userToken']);
        return result.userToken || null;
    } catch (error) {
        console.error('Error getting user token:', error);
        return null;
    }
}

// Set user token in storage
async function setUserToken(token) {
    try {
        await chrome.storage.local.set({ userToken: token });
    } catch (error) {
        console.error('Error setting user token:', error);
    }
}

// Authenticate user with LearnStyle AI
async function authenticateUser() {
    try {
        // Open LearnStyle AI in a new tab for login
        const tab = await chrome.tabs.create({ 
            url: 'http://localhost:5000/login',
            active: true 
        });
        
        // Listen for messages from the tab
        const messageListener = (request, sender, sendResponse) => {
            if (request.action === 'userAuthenticated' && sender.tab.id === tab.id) {
                // User is authenticated, get token
                getExtensionToken();
                chrome.runtime.onMessage.removeListener(messageListener);
            }
        };
        
        chrome.runtime.onMessage.addListener(messageListener);
        
        // Also listen for tab updates
        const tabUpdateListener = (tabId, changeInfo, updatedTab) => {
            if (tabId === tab.id && changeInfo.url && changeInfo.url.includes('dashboard')) {
                // User reached dashboard, try to get token
                getExtensionToken();
                chrome.tabs.onUpdated.removeListener(tabUpdateListener);
            }
        };
        
        chrome.tabs.onUpdated.addListener(tabUpdateListener);
        
    } catch (error) {
        console.error('Error authenticating user:', error);
    }
}

// Get extension token from LearnStyle AI
async function getExtensionToken() {
    try {
        const response = await fetch('http://localhost:5000/api/extension-token', {
            method: 'POST',
            credentials: 'include', // Include cookies for session
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            await setUserToken(data.token);
            console.log('Extension token obtained:', data.token);
            
            // Reload stats with new token
            loadUserStats();
        } else {
            console.error('Failed to get extension token:', response.status);
        }
    } catch (error) {
        console.error('Error getting extension token:', error);
    }
}

// Setup event listeners
function setupEventListeners() {
    // Toggle tracking button
    document.getElementById('toggle-tracking').addEventListener('click', toggleTracking);
    
    // Open dashboard button
    document.getElementById('open-dashboard').addEventListener('click', () => {
        chrome.tabs.create({ url: 'http://localhost:5000/dashboard' });
    });
    
    // Open permissions button
    document.getElementById('open-permissions').addEventListener('click', () => {
        chrome.tabs.create({ url: 'http://localhost:5000/permissions' });
    });
    
    // Open help button
    document.getElementById('open-help').addEventListener('click', () => {
        chrome.tabs.create({ url: 'http://localhost:5000/docs' });
    });
    
    // Add authentication button if not already present
    const authButton = document.getElementById('auth-button');
    if (authButton) {
        authButton.addEventListener('click', authenticateUser);
    }
}

// Toggle tracking
async function toggleTracking() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        if (tab) {
            if (isTracking) {
                // Stop tracking
                await chrome.tabs.sendMessage(tab.id, { action: 'stopTracking' });
                isTracking = false;
                startTime = null;
                
                document.getElementById('toggle-tracking').textContent = 'Start Tracking';
                document.getElementById('tracking-status').textContent = 'Inactive';
                document.getElementById('tracking-status').className = 'status-value status-inactive';
            } else {
                // Start tracking
                await chrome.tabs.sendMessage(tab.id, { action: 'startTracking' });
                isTracking = true;
                startTime = Date.now();
                
                document.getElementById('toggle-tracking').textContent = 'Stop Tracking';
                document.getElementById('tracking-status').textContent = 'Active';
                document.getElementById('tracking-status').className = 'status-value status-active';
            }
        }
    } catch (error) {
        console.error('Error toggling tracking:', error);
    }
}

// Update time display
function updateTimeDisplay() {
    if (isTracking && startTime) {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        document.getElementById('time-on-page').textContent = formatTime(elapsed);
    } else {
        document.getElementById('time-on-page').textContent = '0:00';
    }
}

// Format time in MM:SS format
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// Listen for messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'trackingStarted') {
        isTracking = true;
        startTime = Date.now();
        document.getElementById('toggle-tracking').textContent = 'Stop Tracking';
        document.getElementById('tracking-status').textContent = 'Active';
        document.getElementById('tracking-status').className = 'status-value status-active';
    } else if (request.action === 'trackingStopped') {
        isTracking = false;
        startTime = null;
        document.getElementById('toggle-tracking').textContent = 'Start Tracking';
        document.getElementById('tracking-status').textContent = 'Inactive';
        document.getElementById('tracking-status').className = 'status-value status-inactive';
    } else if (request.action === 'statsUpdated') {
        // Update stats when they change
        loadUserStats();
    }
});

// Clean up interval when popup closes
window.addEventListener('beforeunload', () => {
    if (timeInterval) {
        clearInterval(timeInterval);
    }
});
