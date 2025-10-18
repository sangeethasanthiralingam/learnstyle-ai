"""
Model Training Script for LearnStyle AI
Trains and saves the machine learning models
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.learning_style_predictor import LearningStylePredictor
import pandas as pd

def train_models():
    """Train and save the ML models"""
    print("Starting model training...")
    
    # Initialize predictor
    predictor = LearningStylePredictor()
    
    # Generate synthetic training data
    print("Generating synthetic training data...")
    X, y = predictor.generate_synthetic_dataset(n_samples=2000)
    
    print(f"Generated {len(X)} training samples")
    print(f"Class distribution:")
    print(y.value_counts())
    
    # Train models
    print("Training models...")
    results = predictor.train_models(X, y)
    
    # Print results
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    print(f"Random Forest Accuracy: {results['rf_accuracy']:.3f}")
    print(f"Decision Tree Accuracy: {results['dt_accuracy']:.3f}")
    print(f"RF Cross-validation: {results['rf_cv_mean']:.3f} (+/- {results['rf_cv_std']*2:.3f})")
    print(f"DT Cross-validation: {results['dt_cv_mean']:.3f} (+/- {results['dt_cv_std']*2:.3f})")
    
    print("\nFeature Importance (Top 10):")
    feature_importance = results['feature_importance']
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:10]:
        print(f"  {feature}: {importance:.3f}")
    
    # Save models
    print("\nSaving models...")
    predictor.save_models()
    
    # Test prediction
    print("\nTesting prediction with sample data...")
    sample_answers = [3, 2, 3, 2, 3, 1, 2, 1, 2, 1, 2, 3, 3, 2, 3]  # Visual learner pattern
    prediction = predictor.predict_learning_style(sample_answers)
    print(f"Sample prediction: {prediction}")
    
    print("\nModel training completed successfully!")

if __name__ == '__main__':
    train_models()
