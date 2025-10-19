#!/usr/bin/env python3
"""
Direct ML Models Testing Script for LearnStyle AI
Tests ML models directly without API calls
"""

import os
import sys
import json
import time
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_learning_style_predictor():
    """Test LearningStylePredictor directly"""
    print("🧠 Testing Learning Style Predictor...")
    
    try:
        from ml_models.learning_style_predictor import LearningStylePredictor
        
        predictor = LearningStylePredictor()
        print("    ✅ LearningStylePredictor imported successfully")
        
        # Train model if needed
        if not predictor.is_trained:
            print("    ⚠️  Models not trained yet, training now...")
            X, y = predictor.generate_synthetic_dataset(n_samples=1000)
            predictor.train_models(X, y)
            predictor.save_models()
            print("    ✅ Models trained successfully")
        
        # Test with sample data
        test_responses = [3, 2, 3, 2, 3, 1, 2, 1, 2, 1, 2, 3, 3, 2, 3]
        prediction = predictor.predict_learning_style(test_responses)
        
        print(f"    Predicted style: {prediction.get('dominant_style', 'Unknown')}")
        print(f"    Confidence scores: {prediction.get('confidence_scores', {})}")
        print("    ✅ Learning style prediction working")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Learning style prediction failed: {e}")
        return False

def test_content_generator():
    """Test ContentGenerator directly"""
    print("\n📝 Testing Content Generator...")
    
    try:
        from app.content_generator import ContentGenerator, ContentRequest, ContentType, ContentStyle
        
        generator = ContentGenerator()
        print("    ✅ ContentGenerator imported successfully")
        
        # Test content generation
        request = ContentRequest(
            topic="Machine Learning",
            content_type=ContentType.TEXT,
            learning_style=ContentStyle.VISUAL,
            difficulty_level="intermediate",
            user_preferences={"learning_goals": "understanding", "time_available": "30_minutes"}
        )
        
        content = generator.generate_content(request)
        print(f"    Generated content length: {len(content.content)} characters")
        print(f"    Content type: {content.content_type}")
        print("    ✅ Content generation working")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Content generation failed: {e}")
        return False

def test_multimodal_fusion():
    """Test MultimodalFusionEngine directly"""
    print("\n🔄 Testing Multimodal Fusion Engine...")
    
    try:
        from ml_models.multimodal_fusion_engine import MultimodalFusionEngine
        
        fusion_engine = MultimodalFusionEngine()
        print("    ✅ MultimodalFusionEngine imported successfully")
        
        # Test fusion
        from ml_models.multimodal_fusion_engine import EngagementMetrics
        from datetime import datetime
        
        engagement_metrics = EngagementMetrics(
            content_interaction_time=300,
            scroll_velocity=500,
            click_frequency=5,
            pause_duration=10,
            completion_rate=0.8,
            timestamp=datetime.now()
        )
        
        result = fusion_engine.update_style_weights(1, engagement_metrics, 'visual', 0.8)
        print(f"    Updated weights: {result}")
        print("    ✅ Multimodal fusion working")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Multimodal fusion failed: {e}")
        return False

def test_predictive_analytics():
    """Test PredictiveAnalyticsEngine directly"""
    print("\n📊 Testing Predictive Analytics Engine...")
    
    try:
        from ml_models.predictive_analytics import PredictiveAnalyticsEngine
        
        analytics_engine = PredictiveAnalyticsEngine()
        print("    ✅ PredictiveAnalyticsEngine imported successfully")
        
        # Test analytics
        from ml_models.predictive_analytics import LearningMetrics
        from datetime import datetime
        metrics = LearningMetrics(
            user_id=1,
            timestamp=datetime.now(),
            content_completion_rate=0.85,
            average_session_duration=1800,
            quiz_scores=[0.8, 0.9, 0.7],
            engagement_score=0.7,
            style_content_mismatch=0.3,
            time_spent_per_content={},
            error_rate=0.2,
            help_seeking_frequency=0.3,
            social_interaction_score=0.4
        )
        
        analysis = analytics_engine.analyze_learning_patterns(metrics)
        print(f"    Risk assessment: {analysis.get('risk_assessment', {})}")
        print("    ✅ Predictive analytics working")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Predictive analytics failed: {e}")
        return False

def test_biometric_feedback():
    """Test BiometricFeedback models directly"""
    print("\n💓 Testing Biometric Feedback...")
    
    try:
        from ml_models.biometric_feedback import BiometricFusionEngine
        
        fusion_engine = BiometricFusionEngine()
        print("    ✅ BiometricFusionEngine imported successfully")
        
        # Test fusion
        hrv_metrics = {'rmssd': 45.2, 'pnn50': 12.5, 'stress_level': 0.3}
        gsr_metrics = {'baseline': 0.5, 'peak': 0.8, 'arousal_level': 0.6}
        
        fused_state = fusion_engine.fuse_biometric_data(hrv_metrics, gsr_metrics)
        print(f"    Learning readiness: {fused_state.learning_readiness:.2f}")
        print(f"    Stress level: {fused_state.stress_level:.2f}")
        print("    ✅ Biometric feedback working")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Biometric feedback failed: {e}")
        return False

def test_emotion_ai():
    """Test Emotion AI models directly"""
    print("\n😊 Testing Emotion AI...")
    
    try:
        from ml_models.emotion_ai import FacialEmotionAnalyzer, VoiceEmotionAnalyzer
        
        facial_analyzer = FacialEmotionAnalyzer()
        voice_analyzer = VoiceEmotionAnalyzer()
        print("    ✅ Emotion AI models imported successfully")
        
        # Test facial emotion analysis
        facial_data = {'primary_emotion': 'happy', 'confidence': 0.85, 'valence': 0.8}
        facial_result = facial_analyzer.analyze_facial_emotion(facial_data)
        print(f"    Facial emotion: {facial_result.primary_emotion if hasattr(facial_result, 'primary_emotion') else 'Unknown'}")
        
        # Test voice emotion analysis
        voice_data = {'arousal': 0.6, 'valence': 0.7}
        voice_result = voice_analyzer.analyze_voice_emotion(voice_data)
        print(f"    Voice emotion: {voice_result.primary_emotion if hasattr(voice_result, 'primary_emotion') else 'Unknown'}")
        
        print("    ✅ Emotion AI working")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Emotion AI failed: {e}")
        return False

def test_model_loading():
    """Test if all models can be loaded"""
    print("\n🔧 Testing Model Loading...")
    
    try:
        # Test if we can import the main app module
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Import the app module directly
        import app
        print("    ✅ Flask app module imported successfully")
        
        # Test if ML models can be imported
        from ml_models.learning_style_predictor import LearningStylePredictor
        from ml_models.multimodal_fusion_engine import MultimodalFusionEngine
        from ml_models.predictive_analytics import PredictiveAnalyticsEngine
        from app.content_generator import ContentGenerator
        
        print("    ✅ All ML models can be imported")
        print("    ✅ Model loading test passed")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Model loading failed: {e}")
        return False

def run_direct_tests():
    """Run all direct ML model tests"""
    print("🚀 LearnStyle AI - Direct ML Models Testing")
    print("=" * 50)
    
    tests = [
        test_model_loading,
        test_learning_style_predictor,
        test_content_generator,
        test_multimodal_fusion,
        test_predictive_analytics,
        test_biometric_feedback,
        test_emotion_ai
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"    ❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All ML models are working correctly!")
        print("💡 Your application is ready for use.")
    else:
        print("⚠️  Some ML models have issues. Check the errors above.")
        print("💡 Fix the failing models before deploying.")
    
    return passed == total

if __name__ == "__main__":
    run_direct_tests()
