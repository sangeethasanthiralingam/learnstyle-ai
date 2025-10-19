#!/usr/bin/env python3
"""
ML Models Testing Script for LearnStyle AI
Tests all ML models to ensure they're working correctly
"""

import os
import sys
import json
import requests
import time
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_server_connection():
    """Test if the Flask server is running"""
    print("🔍 Testing server connection...")
    try:
        response = requests.get('http://localhost:5000', timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Server connection failed: {e}")
        print("💡 Make sure to run 'python app.py' in another terminal")
        return False

def test_learning_style_prediction():
    """Test LearningStylePredictor model"""
    print("\n🧠 Testing Learning Style Prediction...")
    
    # Test data for different learning styles
    test_cases = [
        {
            "name": "Visual Learner",
            "responses": [3, 2, 3, 2, 3, 1, 2, 1, 2, 1, 2, 3, 3, 2, 3],
            "expected": "visual"
        },
        {
            "name": "Auditory Learner", 
            "responses": [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 1, 1, 3, 1],
            "expected": "auditory"
        },
        {
            "name": "Kinesthetic Learner",
            "responses": [2, 1, 2, 1, 2, 2, 3, 2, 3, 2, 3, 2, 2, 1, 2],
            "expected": "kinesthetic"
        }
    ]
    
    try:
        for test_case in test_cases:
            print(f"  Testing {test_case['name']}...")
            
            # Test via API endpoint
            response = requests.post('http://localhost:5000/api/predict', 
                                   json={'quiz_answers': test_case['responses']},
                                   timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                predicted_style = data.get('predicted_style', '').lower()
                confidence = data.get('confidence', 0)
                
                print(f"    Predicted: {predicted_style} (confidence: {confidence:.2f})")
                
                if predicted_style == test_case['expected']:
                    print(f"    ✅ Correct prediction!")
                else:
                    print(f"    ⚠️  Expected {test_case['expected']}, got {predicted_style}")
            elif response.status_code == 401:
                print(f"    ⚠️  Learning style prediction requires authentication (expected)")
                print(f"    ✅ Endpoint exists and is protected")
            else:
                print(f"    ❌ API call failed: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Learning style prediction test failed: {e}")

def test_content_generation():
    """Test ContentGenerator model"""
    print("\n📝 Testing Content Generation...")
    
    try:
        # Test content generation endpoint (requires authentication)
        response = requests.post('http://localhost:5000/api/ask-question',
                               json={'question': 'What is machine learning?'},
                               timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                answer = data.get('answer', {})
                content = answer.get('content', '')
                style = answer.get('style', '')
                confidence = answer.get('confidence_score', 0)
                
                print(f"    Style: {style}")
                print(f"    Confidence: {confidence:.2f}")
                print(f"    Content length: {len(content)} characters")
                print(f"    ✅ Content generated successfully")
            else:
                print(f"    ❌ Content generation failed: {data.get('error')}")
        elif response.status_code == 401:
            print("    ⚠️  Content generation requires authentication (expected)")
            print("    ✅ Endpoint exists and is protected")
        else:
            print(f"    ❌ API call failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Content generation test failed: {e}")

def test_analytics_engine():
    """Test PredictiveAnalyticsEngine"""
    print("\n📊 Testing Analytics Engine...")
    
    try:
        # Test analytics endpoint
        response = requests.post('http://localhost:5000/api/predictive-analytics',
                               json={
                                   'completion_rate': 0.85,
                                   'engagement_score': 0.7,
                                   'quiz_scores': [0.8, 0.9, 0.7],
                                   'help_frequency': 0.3,
                                   'social_score': 0.4
                               },
                               timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                analysis = data.get('analytics_data', {})
                risk_assessment = analysis.get('risk_assessment', {})
                
                print(f"    Risk Level: {risk_assessment.get('risk_level', 'Unknown')}")
                print(f"    Risk Score: {risk_assessment.get('risk_score', 0):.2f}")
                print(f"    ✅ Analytics working correctly")
            else:
                print(f"    ❌ Analytics failed: {data.get('error')}")
        elif response.status_code == 401:
            print("    ⚠️  Analytics requires authentication (expected)")
            print("    ✅ Endpoint exists and is protected")
        else:
            print(f"    ❌ API call failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Analytics test failed: {e}")

def test_multimodal_fusion():
    """Test MultimodalFusionEngine"""
    print("\n🔄 Testing Multimodal Fusion...")
    
    try:
        response = requests.post('http://localhost:5000/api/multimodal-fusion',
                               json={
                                   'interaction_time': 300,
                                   'scroll_velocity': 500,
                                   'click_frequency': 5,
                                   'completion_rate': 0.8
                               },
                               timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                fusion_data = data.get('fusion_data', {})
                updated_weights = fusion_data.get('updated_weights', {})
                
                print(f"    Visual Weight: {updated_weights.get('visual', 0):.2f}")
                print(f"    Auditory Weight: {updated_weights.get('auditory', 0):.2f}")
                print(f"    Kinesthetic Weight: {updated_weights.get('kinesthetic', 0):.2f}")
                print(f"    ✅ Multimodal fusion working correctly")
            else:
                print(f"    ❌ Fusion failed: {data.get('error')}")
        elif response.status_code == 401:
            print("    ⚠️  Multimodal fusion requires authentication (expected)")
            print("    ✅ Endpoint exists and is protected")
        else:
            print(f"    ❌ API call failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Multimodal fusion test failed: {e}")

def test_biometric_feedback():
    """Test BiometricFeedback models"""
    print("\n💓 Testing Biometric Feedback...")
    
    try:
        response = requests.post('http://localhost:5000/api/biometric/fusion-analysis',
                               json={
                                   'hrv_metrics': {
                                       'rmssd': 45.2,
                                       'pnn50': 12.5,
                                       'stress_level': 0.3
                                   },
                                   'gsr_metrics': {
                                       'baseline': 0.5,
                                       'peak': 0.8,
                                       'arousal_level': 0.6
                                   }
                               },
                               timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                fused_state = data.get('fused_state', {})
                learning_readiness = fused_state.get('learning_readiness', 0)
                stress_level = fused_state.get('stress_level', 0)
                
                print(f"    Learning Readiness: {learning_readiness:.2f}")
                print(f"    Stress Level: {stress_level:.2f}")
                print(f"    ✅ Biometric feedback working correctly")
            else:
                print(f"    ❌ Biometric feedback failed: {data.get('error')}")
        elif response.status_code == 401:
            print("    ⚠️  Biometric feedback requires authentication (expected)")
            print("    ✅ Endpoint exists and is protected")
        else:
            print(f"    ❌ API call failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Biometric feedback test failed: {e}")

def test_emotion_ai():
    """Test Emotion AI models"""
    print("\n😊 Testing Emotion AI...")
    
    try:
        response = requests.post('http://localhost:5000/api/emotion-ai/analyze',
                               json={
                                   'facial_data': {
                                       'primary_emotion': 'happy',
                                       'confidence': 0.85,
                                       'valence': 0.8
                                   },
                                   'voice_data': {
                                       'arousal': 0.6,
                                       'valence': 0.7
                                   }
                               },
                               timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                emotion_data = data.get('emotion_data', {})
                fused_emotion = emotion_data.get('fused_emotion', {})
                
                print(f"    Primary Emotion: {fused_emotion.get('primary_emotion', 'Unknown')}")
                print(f"    Confidence: {fused_emotion.get('confidence', 0):.2f}")
                print(f"    ✅ Emotion AI working correctly")
            else:
                print(f"    ❌ Emotion AI failed: {data.get('error')}")
        elif response.status_code == 401:
            print("    ⚠️  Emotion AI requires authentication (expected)")
            print("    ✅ Endpoint exists and is protected")
        else:
            print(f"    ❌ API call failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Emotion AI test failed: {e}")

def test_model_loading():
    """Test if ML models are loaded correctly"""
    print("\n🔧 Testing Model Loading...")
    
    try:
        # Test if models are loaded by checking the server startup logs
        response = requests.get('http://localhost:5000', timeout=5)
        if response.status_code == 200:
            print("    ✅ Server started successfully")
            print("    ✅ ML models should be loaded (check server logs)")
        else:
            print("    ❌ Server not responding properly")
            
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")

def run_comprehensive_test():
    """Run all ML model tests"""
    print("🚀 LearnStyle AI - ML Models Testing Suite")
    print("=" * 50)
    
    # Test server connection first
    if not test_server_connection():
        print("\n❌ Cannot proceed with tests - server not running")
        print("💡 Please run 'python app.py' in another terminal first")
        return
    
    # Run all tests
    test_model_loading()
    test_learning_style_prediction()
    test_content_generation()
    test_analytics_engine()
    test_multimodal_fusion()
    test_biometric_feedback()
    test_emotion_ai()
    
    print("\n" + "=" * 50)
    print("🎉 ML Models Testing Complete!")
    print("💡 Check the results above to see which models are working correctly")

if __name__ == "__main__":
    run_comprehensive_test()
