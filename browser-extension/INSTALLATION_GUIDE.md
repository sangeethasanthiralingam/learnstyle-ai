# 🚀 LearnStyle AI Browser Extension - Installation Guide

## 📋 Prerequisites
- Chrome, Edge, or Firefox browser
- LearnStyle AI application running on `http://localhost:5000`
- User account logged in to LearnStyle AI

## 🔧 Installation Steps

### Step 1: Download the Extension
1. Navigate to the `browser-extension` folder in your project
2. Make sure all files are present:
   - `manifest.json`
   - `background.js`
   - `content.js`
   - `learning-tracker.js`
   - `popup.html`
   - `popup.js`

### Step 2: Install in Chrome/Edge
1. Open Chrome or Edge browser
2. Go to `chrome://extensions/` (or `edge://extensions/`)
3. Enable "Developer mode" (toggle in top-right corner)
4. Click "Load unpacked"
5. Select the `browser-extension` folder
6. The extension should appear in your extensions list

### Step 3: Install in Firefox
1. Open Firefox browser
2. Go to `about:debugging`
3. Click "This Firefox"
4. Click "Load Temporary Add-on"
5. Select the `manifest.json` file from the `browser-extension` folder

### Step 4: Pin the Extension
1. Click the puzzle piece icon in your browser toolbar
2. Find "LearnStyle AI - Learning Tracker"
3. Click the pin icon to keep it visible

## 🔐 Authentication Setup

### Step 1: Login to LearnStyle AI
1. Go to `http://localhost:5000`
2. Login to your account
3. Keep the tab open

### Step 2: Authorize the Extension
1. Click the LearnStyle AI extension icon in your browser
2. Click "Authorize Extension"
3. This will open a new tab to authorize the extension
4. Click "Authorize" to grant permissions

## 🎯 Testing YouTube Tracking

### Step 1: Visit YouTube
1. Go to `https://youtube.com`
2. Search for any educational video
3. Click on a video to watch it

### Step 2: Check Tracking
1. Look for the green "📚 LearnStyle AI Tracking" indicator in the top-right corner
2. Watch the video for at least 30 seconds
3. Go back to your LearnStyle AI dashboard
4. Check the "Learning Sites Activity" section

### Step 3: Verify Data
1. The video should appear in "Recent Learning Sites"
2. Time spent should be recorded
3. Activity type should show as "video"

## 🛠️ Troubleshooting

### Extension Not Working?
1. **Check if extension is enabled:**
   - Go to `chrome://extensions/`
   - Make sure LearnStyle AI extension is enabled

2. **Check console for errors:**
   - Press F12 on YouTube
   - Look for "LearnStyle AI" messages in console

3. **Check authentication:**
   - Make sure you're logged in to LearnStyle AI
   - Try re-authorizing the extension

### No Data Showing?
1. **Check API connection:**
   - Make sure LearnStyle AI is running on `http://localhost:5000`
   - Check if the API endpoint is accessible

2. **Check CORS settings:**
   - The extension needs CORS to be enabled
   - Make sure the Flask app allows requests from the extension

3. **Check browser permissions:**
   - Make sure the extension has permission to access YouTube
   - Check if popup blockers are interfering

### Still Not Working?
1. **Refresh the extension:**
   - Go to `chrome://extensions/`
   - Click the refresh icon on the LearnStyle AI extension

2. **Clear extension data:**
   - Right-click the extension icon
   - Select "Inspect popup"
   - Go to Application tab > Storage > Clear storage

3. **Reinstall the extension:**
   - Remove the extension
   - Reload the extension folder

## 📊 What Gets Tracked?

- **YouTube Videos:** Educational content, tutorials, lectures
- **Coursera:** Courses, videos, quizzes
- **Khan Academy:** Lessons, exercises, videos
- **edX:** Courses, videos, assignments
- **Udemy:** Courses, lectures, quizzes
- **GitHub:** Code repositories, documentation
- **Stack Overflow:** Questions, answers, discussions
- **Google Scholar:** Research papers, articles
- **ResearchGate:** Academic papers, discussions

## 🔒 Privacy & Security

- **Data Storage:** All data is stored locally on your LearnStyle AI server
- **No Third-Party Tracking:** We don't share your data with external services
- **Secure Communication:** All data is transmitted over HTTPS
- **User Control:** You can disable tracking anytime through the extension popup

## 🆘 Support

If you're still having issues:
1. Check the browser console for error messages
2. Make sure all files are in the correct locations
3. Verify that LearnStyle AI is running and accessible
4. Try the troubleshooting steps above

---

**Happy Learning! 🎓**
