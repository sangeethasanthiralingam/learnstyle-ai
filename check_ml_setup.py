#!/usr/bin/env python3
"""
ML Models Setup Checker
Quick diagnostic script to verify ML models setup
"""

import os
import sys
import importlib

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_dependencies():
    """Check required ML dependencies"""
    print("\n📦 Checking ML dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn',
        'plotly', 'opencv-python', 'Pillow', 'librosa', 'networkx',
        'scipy', 'flask', 'flask_sqlalchemy', 'flask_login', 'pymysql'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                importlib.import_module('cv2')
            elif package == 'Pillow':
                importlib.import_module('PIL')
            elif package == 'scikit-learn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   💡 Install missing packages: pip install {' '.join(missing_packages)}")
        return False
    return True

def check_model_files():
    """Check if ML model files exist"""
    print("\n🤖 Checking ML model files...")
    
    model_dir = "ml_models/saved_models"
    required_files = [
        "random_forest_model.pkl",
        "decision_tree_model.pkl", 
        "feature_importance.pkl"
    ]
    
    if not os.path.exists(model_dir):
        print(f"   ❌ Model directory not found: {model_dir}")
        return False
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (missing)")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n   💡 Generate models: python scripts/train_models.py")
        return False
    return True

def check_database_setup():
    """Check database configuration"""
    print("\n🗄️ Checking database setup...")
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("   ✅ .env file found")
    else:
        print("   ⚠️  .env file not found (using defaults)")
    
    # Check if database models can be imported
    try:
        from app.models import db, User, LearningProfile
        print("   ✅ Database models importable")
        return True
    except ImportError as e:
        print(f"   ❌ Database models import failed: {e}")
        return False

def check_ml_models_import():
    """Check if ML models can be imported"""
    print("\n🧠 Checking ML models import...")
    
    try:
        from ml_models.learning_style_predictor import LearningStylePredictor
        from ml_models.multimodal_fusion_engine import MultimodalFusionEngine
        from ml_models.predictive_analytics import PredictiveAnalyticsEngine
        from app.content_generator import ContentGenerator
        print("   ✅ ML models importable")
        return True
    except ImportError as e:
        print(f"   ❌ ML models import failed: {e}")
        return False

def test_model_initialization():
    """Test if models can be initialized"""
    print("\n🔧 Testing model initialization...")
    
    try:
        from ml_models.learning_style_predictor import LearningStylePredictor
        
        predictor = LearningStylePredictor()
        print("   ✅ LearningStylePredictor initialized")
        
        # Test if models can be loaded
        if hasattr(predictor, 'load_models'):
            try:
                predictor.load_models()
                print("   ✅ Models loaded successfully")
                return True
            except Exception as e:
                print(f"   ⚠️  Model loading failed: {e}")
                return False
        else:
            print("   ⚠️  load_models method not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Model initialization failed: {e}")
        return False

def run_diagnostic():
    """Run complete diagnostic"""
    print("🔍 LearnStyle AI - ML Setup Diagnostic")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_model_files(),
        check_database_setup(),
        check_ml_models_import(),
        test_model_initialization()
    ]
    
    passed = sum(checks)
    total = len(checks)
    
    print("\n" + "=" * 50)
    print(f"📊 Diagnostic Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Your ML setup is ready.")
        print("💡 You can now run: python app.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("💡 Run the suggested commands to resolve issues.")
    
    return passed == total

if __name__ == "__main__":
    run_diagnostic()
