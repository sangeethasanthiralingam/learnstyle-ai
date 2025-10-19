#!/usr/bin/env python3
"""
Test script for YouTube tracking functionality
"""

import requests
import json
import time

def test_youtube_tracking():
    """Test the YouTube tracking API endpoint"""
    
    # Test data
    test_data = {
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "name": "YouTube",
        "activity_type": "visit",
        "time_spent": 120,  # 2 minutes
        "content_type": "video",
        "notes": "Test video tracking"
    }
    
    # API endpoint
    api_url = "http://localhost:5000/api/learning-sites"
    
    print("🧪 Testing YouTube Tracking API...")
    print(f"📡 Sending request to: {api_url}")
    print(f"📊 Test data: {json.dumps(test_data, indent=2)}")
    
    try:
        # Send POST request
        response = requests.post(
            api_url,
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        print(f"\n📈 Response Status: {response.status_code}")
        print(f"📄 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Error! Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the Flask app is running on localhost:5000")
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: Request took too long")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

def test_get_activities():
    """Test getting learning activities"""
    
    api_url = "http://localhost:5000/api/learning-sites"
    
    print("\n🔍 Testing GET Learning Activities...")
    
    try:
        response = requests.get(api_url, timeout=10)
        
        print(f"📈 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Found {len(result.get('activities', []))} activities")
            print(f"📊 Activities: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Error! Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the Flask app is running on localhost:5000")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

def main():
    """Main test function"""
    print("🚀 LearnStyle AI - YouTube Tracking Test")
    print("=" * 50)
    
    # Test POST (track activity)
    test_youtube_tracking()
    
    # Wait a moment
    time.sleep(1)
    
    # Test GET (retrieve activities)
    test_get_activities()
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")

if __name__ == "__main__":
    main()
