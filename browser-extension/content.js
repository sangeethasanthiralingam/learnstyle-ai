/**
 * LearnStyle AI - Browser Extension Content Script
 * Injects learning tracker into educational websites
 */

// Inject the learning tracker script
const script = document.createElement('script');
script.src = chrome.runtime.getURL('learning-tracker.js');
script.onload = function() {
    this.remove();
};
(document.head || document.documentElement).appendChild(script);

// Listen for messages from the extension popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'getPageInfo') {
        sendResponse({
            url: window.location.href,
            title: document.title,
            hostname: window.location.hostname,
            isLearningSite: isLearningSite(window.location.hostname)
        });
    } else if (request.action === 'startTracking') {
        // Start tracking on this page
        window.postMessage({ action: 'startLearnStyleTracking' }, '*');
        sendResponse({ success: true });
    } else if (request.action === 'stopTracking') {
        // Stop tracking on this page
        window.postMessage({ action: 'stopLearnStyleTracking' }, '*');
        sendResponse({ success: true });
    }
});

// Check if current site is a learning site
function isLearningSite(hostname) {
    const learningSites = [
        'coursera.org', 'khanacademy.org', 'edx.org', 'udemy.com',
        'github.com', 'stackoverflow.com', 'youtube.com', 'scholar.google.com',
        'researchgate.net', 'academia.edu', 'arxiv.org', 'ieee.org'
    ];
    
    return learningSites.some(site => hostname.includes(site));
}

// Add visual indicator when tracking is active
function addTrackingIndicator() {
    const indicator = document.createElement('div');
    indicator.id = 'learnstyle-indicator';
    indicator.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #28a745;
        color: white;
        padding: 8px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-family: Arial, sans-serif;
        z-index: 10000;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        display: none;
    `;
    indicator.innerHTML = '📚 LearnStyle AI Tracking';
    document.body.appendChild(indicator);
}

// Show/hide tracking indicator
function toggleTrackingIndicator(show) {
    const indicator = document.getElementById('learnstyle-indicator');
    if (indicator) {
        indicator.style.display = show ? 'block' : 'none';
    }
}

// Listen for tracking state changes
window.addEventListener('message', (event) => {
    if (event.data.action === 'trackingStarted') {
        toggleTrackingIndicator(true);
    } else if (event.data.action === 'trackingStopped') {
        toggleTrackingIndicator(false);
    }
});

// Initialize indicator
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', addTrackingIndicator);
} else {
    addTrackingIndicator();
}

// Track page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, pause tracking
        window.postMessage({ action: 'pauseTracking' }, '*');
    } else {
        // Page is visible, resume tracking
        window.postMessage({ action: 'resumeTracking' }, '*');
    }
});

// Track before page unload
window.addEventListener('beforeunload', () => {
    window.postMessage({ action: 'stopTracking' }, '*');
});

console.log('LearnStyle AI: Content script loaded on', window.location.hostname);
