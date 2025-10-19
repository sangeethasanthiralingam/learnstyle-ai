#!/usr/bin/env python3
"""
Setup script for YouTube tracking functionality
"""

import os
import sys
import subprocess
import time
import requests
import json

def check_flask_running():
    """Check if Flask app is running"""
    try:
        response = requests.get('http://localhost:5000', timeout=5)
        return response.status_code == 200
    except:
        return False

def start_flask_app():
    """Start the Flask application"""
    print("🚀 Starting Flask application...")
    try:
        # Start Flask app in background
        process = subprocess.Popen([
            sys.executable, 'app.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for Flask to start
        time.sleep(3)
        
        if check_flask_running():
            print("✅ Flask app is running on http://localhost:5000")
            return process
        else:
            print("❌ Failed to start Flask app")
            return None
    except Exception as e:
        print(f"❌ Error starting Flask app: {e}")
        return None

def test_api_endpoint():
    """Test the learning sites API endpoint"""
    print("\n🧪 Testing API endpoint...")
    
    test_data = {
        "url": "https://www.youtube.com/watch?v=test123",
        "name": "YouTube",
        "activity_type": "visit",
        "time_spent": 60,
        "content_type": "video"
    }
    
    try:
        response = requests.post(
            'http://localhost:5000/api/learning-sites',
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ API endpoint is working!")
            result = response.json()
            print(f"📊 Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ API endpoint failed: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def test_get_activities():
    """Test getting learning activities"""
    print("\n🔍 Testing GET activities...")
    
    try:
        response = requests.get('http://localhost:5000/api/learning-sites', timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            activities = result.get('activities', [])
            print(f"✅ Found {len(activities)} activities")
            if activities:
                print("📊 Recent activities:")
                for activity in activities[:3]:  # Show first 3
                    print(f"  - {activity.get('site_name', 'Unknown')}: {activity.get('time_spent', 0)}s")
            return True
        else:
            print(f"❌ Failed to get activities: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting activities: {e}")
        return False

def create_demo_data():
    """Create some demo learning activities"""
    print("\n📝 Creating demo learning activities...")
    
    demo_activities = [
        {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "name": "YouTube",
            "activity_type": "visit",
            "time_spent": 180,
            "content_type": "video"
        },
        {
            "url": "https://www.khanacademy.org/math/algebra",
            "name": "Khan Academy",
            "activity_type": "visit",
            "time_spent": 300,
            "content_type": "education"
        },
        {
            "url": "https://www.coursera.org/learn/machine-learning",
            "name": "Coursera",
            "activity_type": "visit",
            "time_spent": 600,
            "content_type": "education"
        }
    ]
    
    success_count = 0
    for activity in demo_activities:
        try:
            response = requests.post(
                'http://localhost:5000/api/learning-sites',
                json=activity,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            if response.status_code == 200:
                success_count += 1
                print(f"  ✅ Created: {activity['name']}")
            else:
                print(f"  ❌ Failed: {activity['name']}")
        except Exception as e:
            print(f"  ❌ Error: {activity['name']} - {e}")
    
    print(f"📊 Created {success_count}/{len(demo_activities)} demo activities")
    return success_count > 0

def print_instructions():
    """Print setup instructions"""
    print("\n" + "="*60)
    print("🎯 YOUTUBE TRACKING SETUP COMPLETE!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Install the browser extension:")
    print("   - Open Chrome/Edge")
    print("   - Go to chrome://extensions/")
    print("   - Enable 'Developer mode'")
    print("   - Click 'Load unpacked'")
    print("   - Select the 'browser-extension' folder")
    print("\n2. Test YouTube tracking:")
    print("   - Go to YouTube.com")
    print("   - Click the LearnStyle AI extension icon")
    print("   - Click 'Start Tracking'")
    print("   - Watch a video for a few minutes")
    print("   - Check your dashboard at http://localhost:5000")
    print("\n3. Check the dashboard:")
    print("   - Go to http://localhost:5000")
    print("   - Look for 'Learning Sites Activity' section")
    print("   - You should see your YouTube activity!")
    print("\n🔧 Troubleshooting:")
    print("   - Make sure Flask app is running")
    print("   - Check browser console for errors")
    print("   - Verify extension is enabled")
    print("   - Check CORS settings if needed")
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print("🚀 LearnStyle AI - YouTube Tracking Setup")
    print("="*50)
    
    # Check if Flask is already running
    if check_flask_running():
        print("✅ Flask app is already running!")
    else:
        # Start Flask app
        flask_process = start_flask_app()
        if not flask_process:
            print("❌ Cannot start Flask app. Please run 'python app.py' manually.")
            return
    
    # Wait for Flask to be ready
    print("⏳ Waiting for Flask to be ready...")
    for i in range(10):
        if check_flask_running():
            break
        time.sleep(1)
        print(f"   Waiting... ({i+1}/10)")
    
    if not check_flask_running():
        print("❌ Flask app is not responding. Please check the logs.")
        return
    
    # Test API endpoints
    api_working = test_api_endpoint()
    if not api_working:
        print("❌ API is not working. Please check the Flask app.")
        return
    
    # Create demo data
    create_demo_data()
    
    # Test getting activities
    test_get_activities()
    
    # Print instructions
    print_instructions()

if __name__ == "__main__":
    main()
