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
        // Get stored stats from chrome storage
        const result = await chrome.storage.local.get(['dailyStats']);
        const stats = result.dailyStats || {
            sitesVisited: 0,
            totalTime: 0,
            learningScore: 0
        };
        
        document.getElementById('sites-visited').textContent = stats.sitesVisited;
        document.getElementById('total-time').textContent = formatTime(stats.totalTime);
        document.getElementById('learning-score').textContent = stats.learningScore + '%';
    } catch (error) {
        console.error('Error loading user stats:', error);
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
