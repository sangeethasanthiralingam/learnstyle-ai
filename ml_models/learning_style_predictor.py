"""
LearnStyle AI - Machine Learning Engine
Learning Style Prediction with Random Forest and Decision Tree
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from typing import Dict, List, Tuple
import random

class LearningStylePredictor:
    """
    Machine learning model for predicting learning styles based on quiz responses
    """
    
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.dt_model = DecisionTreeClassifier(random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_importance = None
        
    def generate_synthetic_dataset(self, n_samples: int = 500) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic training data with realistic patterns
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            X: Features (quiz answers), y: Labels (learning styles)
        """
        np.random.seed(42)
        random.seed(42)
        
        data = []
        labels = []
        
        # Define question patterns for each learning style
        visual_patterns = {
            'prefer_charts': [2, 3],  # Questions about visual aids
            'like_diagrams': [2, 3],
            'remember_faces': [2, 3],
            'spatial_awareness': [2, 3],
            'color_coding': [2, 3]
        }
        
        auditory_patterns = {
            'prefer_lectures': [2, 3],
            'remember_names': [2, 3], 
            'like_discussions': [2, 3],
            'music_helps': [2, 3],
            'verbal_instructions': [2, 3]
        }
        
        kinesthetic_patterns = {
            'hands_on_learning': [2, 3],
            'need_movement': [2, 3],
            'learn_by_doing': [2, 3],
            'physical_activities': [2, 3],
            'touch_and_feel': [2, 3]
        }
        
        styles = ['visual', 'auditory', 'kinesthetic']
        
        for _ in range(n_samples):
            # Choose a dominant learning style
            dominant_style = random.choice(styles)
            
            # Generate 15 quiz answers (1-3 scale)
            answers = []
            
            for q in range(15):
                if dominant_style == 'visual' and q < 5:
                    # Visual learners prefer visual methods
                    answer = random.choice([2, 3, 3])  # Bias towards agree/strongly agree
                elif dominant_style == 'auditory' and 5 <= q < 10:
                    # Auditory learners prefer auditory methods  
                    answer = random.choice([2, 3, 3])
                elif dominant_style == 'kinesthetic' and q >= 10:
                    # Kinesthetic learners prefer hands-on methods
                    answer = random.choice([2, 3, 3])
                else:
                    # Other questions get more neutral responses
                    answer = random.choice([1, 2, 3])
                
                answers.append(answer)
            
            data.append(answers)
            labels.append(dominant_style)
        
        # Convert to DataFrame
        columns = [f'question_{i+1}' for i in range(15)]
        X = pd.DataFrame(data, columns=columns)
        y = pd.Series(labels)
        
        return X, y
    
    def preprocess_features(self, quiz_answers: List[int]) -> np.array:
        """
        Process raw quiz answers into ML features
        
        Args:
            quiz_answers: List of 15 quiz responses (1-3)
            
        Returns:
            Processed feature array
        """
        # Ensure we have exactly 15 answers
        if len(quiz_answers) != 15:
            raise ValueError("Quiz must have exactly 15 answers")
        
        # Convert to numpy array and normalize if needed
        features = np.array(quiz_answers).reshape(1, -1)
        return features
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train both Random Forest and Decision Tree models
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Training results and performance metrics
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest
        self.rf_model.fit(X_train, y_train)
        rf_pred = self.rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        # Train Decision Tree  
        self.dt_model.fit(X_train, y_train)
        dt_pred = self.dt_model.predict(X_test)
        dt_accuracy = accuracy_score(y_test, dt_pred)
        
        # Cross-validation
        rf_cv_scores = cross_val_score(self.rf_model, X_train, y_train, cv=5)
        dt_cv_scores = cross_val_score(self.dt_model, X_train, y_train, cv=5)
        
        # Feature importance (from Random Forest)
        self.feature_importance = dict(zip(
            X.columns, 
            self.rf_model.feature_importances_
        ))
        
        self.is_trained = True
        
        results = {
            'rf_accuracy': rf_accuracy,
            'dt_accuracy': dt_accuracy,
            'rf_cv_mean': rf_cv_scores.mean(),
            'rf_cv_std': rf_cv_scores.std(),
            'dt_cv_mean': dt_cv_scores.mean(),
            'dt_cv_std': dt_cv_scores.std(),
            'rf_classification_report': classification_report(y_test, rf_pred),
            'dt_classification_report': classification_report(y_test, dt_pred),
            'feature_importance': self.feature_importance
        }
        
        return results
    
    def predict_learning_style(self, quiz_answers: List[int]) -> Dict:
        """
        Predict learning style from quiz answers
        
        Args:
            quiz_answers: List of 15 quiz responses (1-3)
            
        Returns:
            Dictionary with style probabilities and dominant style
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess features
        features = self.preprocess_features(quiz_answers)
        
        # Get probabilities from Random Forest (better model typically)
        probabilities = self.rf_model.predict_proba(features)[0]
        classes = self.rf_model.classes_
        
        # Create result dictionary
        result = {}
        for i, style in enumerate(classes):
            result[style] = float(probabilities[i])
        
        # Find dominant style
        dominant_style = max(result.keys(), key=lambda k: result[k])
        result['dominant_style'] = dominant_style
        
        return result
    
    def save_models(self, model_dir: str = 'ml_models/saved_models'):
        """Save trained models to disk"""
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.rf_model, os.path.join(model_dir, 'random_forest_model.pkl'))
        joblib.dump(self.dt_model, os.path.join(model_dir, 'decision_tree_model.pkl'))
        joblib.dump(self.feature_importance, os.path.join(model_dir, 'feature_importance.pkl'))
        
        print(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str = 'ml_models/saved_models'):
        """Load trained models from disk"""
        try:
            self.rf_model = joblib.load(os.path.join(model_dir, 'random_forest_model.pkl'))
            self.dt_model = joblib.load(os.path.join(model_dir, 'decision_tree_model.pkl'))
            self.feature_importance = joblib.load(os.path.join(model_dir, 'feature_importance.pkl'))
            self.is_trained = True
            print(f"Models loaded from {model_dir}")
        except FileNotFoundError:
            print(f"No saved models found in {model_dir}")
    
    def evaluate_model_performance(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Generate comprehensive performance metrics
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Performance metrics dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Predictions
        rf_pred = self.rf_model.predict(X_test)
        dt_pred = self.dt_model.predict(X_test)
        
        metrics = {
            'rf_accuracy': accuracy_score(y_test, rf_pred),
            'dt_accuracy': accuracy_score(y_test, dt_pred),
            'rf_confusion_matrix': confusion_matrix(y_test, rf_pred).tolist(),
            'dt_confusion_matrix': confusion_matrix(y_test, dt_pred).tolist(),
            'rf_classification_report': classification_report(y_test, rf_pred, output_dict=True),
            'dt_classification_report': classification_report(y_test, dt_pred, output_dict=True)
        }
        
        return metrics

# Continuous learning functions
def update_user_learning_profile(user_id: int, interaction_data: Dict):
    """
    Update learning style weights based on user interactions
    This would be called periodically to adapt to user behavior
    """
    # This would integrate with the main application database
    # to update learning profiles based on:
    # - Content completion rates by type
    # - Time spent on different content formats  
    # - Performance on style-specific exercises
    # - User feedback and ratings
    pass

def retrain_model_periodically():
    """
    Scheduled retraining with new user data
    This would be run as a background task
    """
    # Collect anonymized user interaction data
    # Retrain model monthly/weekly
    # A/B test new model performance
    pass

if __name__ == "__main__":
    # Example usage and training
    predictor = LearningStylePredictor()
    
    # Generate training data
    print("Generating synthetic training data...")
    X, y = predictor.generate_synthetic_dataset(n_samples=1000)
    
    # Train models
    print("Training models...")
    results = predictor.train_models(X, y)
    
    print(f"Random Forest Accuracy: {results['rf_accuracy']:.3f}")
    print(f"Decision Tree Accuracy: {results['dt_accuracy']:.3f}")
    print(f"RF Cross-validation: {results['rf_cv_mean']:.3f} (+/- {results['rf_cv_std']*2:.3f})")
    
    # Save models
    predictor.save_models()
    
    # Example prediction
    sample_answers = [3, 2, 3, 2, 3, 1, 2, 1, 2, 1, 2, 3, 3, 2, 3]  # Visual learner pattern
    prediction = predictor.predict_learning_style(sample_answers)
    print(f"\\nSample prediction: {prediction}")