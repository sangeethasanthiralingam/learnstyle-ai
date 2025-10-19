/**
 * LearnStyle AI - Browser Extension Background Script
 */

// Extension installation/update handler
chrome.runtime.onInstalled.addListener((details) => {
    if (details.reason === 'install') {
        // First time installation
        console.log('LearnStyle AI extension installed');
        
        // Set default settings
        chrome.storage.local.set({
            trackingEnabled: true,
            dailyStats: {
                sitesVisited: 0,
                totalTime: 0,
                learningScore: 0
            }
        });
        
        // Open welcome page
        chrome.tabs.create({ url: 'http://localhost:5000/onboarding' });
    } else if (details.reason === 'update') {
        console.log('LearnStyle AI extension updated');
    }
});

// Track tab changes
chrome.tabs.onActivated.addListener((activeInfo) => {
    // Update tracking when user switches tabs
    updateTrackingState();
});

// Track tab updates
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url) {
        // Page loaded, check if it's a learning site
        if (isLearningSite(tab.url)) {
            // Inject content script if needed
            chrome.scripting.executeScript({
                target: { tabId: tabId },
                files: ['content.js']
            }).catch(() => {
                // Script already injected or error
            });
        }
    }
});

// Track tab removal
chrome.tabs.onRemoved.addListener((tabId, removeInfo) => {
    // Clean up any tracking data for this tab
    chrome.storage.local.remove([`tab_${tabId}`]);
});

// Handle messages from content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'trackActivity') {
        handleActivityTracking(request.data, sender.tab);
    } else if (request.action === 'updateStats') {
        updateDailyStats(request.data);
    } else if (request.action === 'getUserToken') {
        // Get user authentication token
        chrome.storage.local.get(['userToken'], (result) => {
            sendResponse({ token: result.userToken });
        });
        return true; // Keep message channel open for async response
    }
});

// Handle activity tracking
async function handleActivityTracking(data, tab) {
    try {
        // Get user token
        const result = await chrome.storage.local.get(['userToken']);
        const token = result.userToken;
        
        if (!token) {
            console.log('No user token found, skipping activity tracking');
            return;
        }
        
        // Send activity data to LearnStyle AI API
        const response = await fetch('http://localhost:5000/api/learning-sites', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                url: data.url,
                name: data.name,
                activity_type: data.activity_type,
                time_spent: data.time_spent,
                content_type: data.content_type
            })
        });
        
        if (response.ok) {
            console.log('Activity tracked successfully');
            
            // Update local stats
            updateDailyStats({
                sitesVisited: 1,
                totalTime: data.time_spent,
                learningScore: calculateLearningScore(data)
            });
        } else {
            console.error('Failed to track activity:', response.statusText);
        }
    } catch (error) {
        console.error('Error tracking activity:', error);
    }
}

// Update daily statistics
async function updateDailyStats(newStats) {
    try {
        const result = await chrome.storage.local.get(['dailyStats']);
        const currentStats = result.dailyStats || {
            sitesVisited: 0,
            totalTime: 0,
            learningScore: 0
        };
        
        // Update stats
        const updatedStats = {
            sitesVisited: currentStats.sitesVisited + (newStats.sitesVisited || 0),
            totalTime: currentStats.totalTime + (newStats.totalTime || 0),
            learningScore: Math.max(currentStats.learningScore, newStats.learningScore || 0)
        };
        
        // Save updated stats
        await chrome.storage.local.set({ dailyStats: updatedStats });
        
        // Notify popup if it's open
        chrome.runtime.sendMessage({
            action: 'statsUpdated',
            stats: updatedStats
        }).catch(() => {
            // Popup not open, ignore error
        });
    } catch (error) {
        console.error('Error updating daily stats:', error);
    }
}

// Calculate learning score based on activity
function calculateLearningScore(activityData) {
    let score = 0;
    
    // Base score for visiting learning sites
    if (isLearningSite(activityData.url)) {
        score += 20;
    }
    
    // Time-based scoring
    const timeSpent = activityData.time_spent || 0;
    if (timeSpent > 300) { // 5 minutes
        score += 30;
    } else if (timeSpent > 60) { // 1 minute
        score += 15;
    }
    
    // Activity type scoring
    const activityType = activityData.activity_type || 'visit';
    switch (activityType) {
        case 'quiz':
        case 'assignment':
            score += 40;
            break;
        case 'video':
        case 'lecture':
            score += 25;
            break;
        case 'study':
            score += 20;
            break;
        default:
            score += 10;
    }
    
    return Math.min(score, 100); // Cap at 100%
}

// Check if URL is a learning site
function isLearningSite(url) {
    const learningSites = [
        'coursera.org', 'khanacademy.org', 'edx.org', 'udemy.com',
        'github.com', 'stackoverflow.com', 'youtube.com', 'scholar.google.com',
        'researchgate.net', 'academia.edu', 'arxiv.org', 'ieee.org'
    ];
    
    return learningSites.some(site => url.includes(site));
}

// Update tracking state
function updateTrackingState() {
    // This function can be used to manage global tracking state
    chrome.storage.local.get(['trackingEnabled'], (result) => {
        const trackingEnabled = result.trackingEnabled !== false; // Default to true
        
        // Update badge
        chrome.action.setBadgeText({
            text: trackingEnabled ? 'ON' : 'OFF'
        });
        
        chrome.action.setBadgeBackgroundColor({
            color: trackingEnabled ? '#28a745' : '#6c757d'
        });
    });
}

// Initialize tracking state
updateTrackingState();

// Reset daily stats at midnight
chrome.alarms.create('resetDailyStats', {
    when: getNextMidnight(),
    periodInMinutes: 24 * 60 // 24 hours
});

chrome.alarms.onAlarm.addListener((alarm) => {
    if (alarm.name === 'resetDailyStats') {
        chrome.storage.local.set({
            dailyStats: {
                sitesVisited: 0,
                totalTime: 0,
                learningScore: 0
            }
        });
    }
});

// Get next midnight timestamp
function getNextMidnight() {
    const now = new Date();
    const midnight = new Date(now);
    midnight.setHours(24, 0, 0, 0);
    return midnight.getTime();
}
