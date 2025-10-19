"""
Biometric Fusion Engine Module

This module provides comprehensive biometric data fusion for learning optimization including:
- Multi-modal biometric data integration
- Holistic learning state assessment
- Real-time learning optimization
- Biometric-based content adaptation

Author: LearnStyle AI Team
Version: 1.0.0
"""

import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

class BiometricState(Enum):
    """Combined biometric states"""
    OPTIMAL_LEARNING = "optimal_learning"
    STRESSED_LEARNING = "stressed_learning"
    FATIGUED_LEARNING = "fatigued_learning"
    OVERSTIMULATED = "overstimulated"
    UNDERSTIMULATED = "understimulated"
    RECOVERING = "recovering"
    NOT_RECOMMENDED = "not_recommended"

class LearningOptimization(Enum):
    """Learning optimization strategies"""
    INCREASE_DIFFICULTY = "increase_difficulty"
    MAINTAIN_CURRENT = "maintain_current"
    DECREASE_DIFFICULTY = "decrease_difficulty"
    TAKE_BREAK = "take_break"
    CHANGE_CONTENT_TYPE = "change_content_type"
    REDUCE_COGNITIVE_LOAD = "reduce_cognitive_load"
    INCREASE_ENGAGEMENT = "increase_engagement"

@dataclass
class FusedBiometricState:
    """Fused biometric state data"""
    timestamp: datetime
    hrv_state: str
    gsr_state: str
    combined_state: BiometricState
    learning_readiness: float
    stress_level: float
    engagement_level: float
    fatigue_level: float
    arousal_level: float
    emotional_state: str
    cognitive_load: float
    attention_level: float
    motivation_level: float
    confidence: float
    recommendations: List[str]

class BiometricFusionEngine:
    """
    Advanced biometric data fusion system
    """
    
    def __init__(self):
        """Initialize biometric fusion engine"""
        self.fusion_history = []
        self.learning_models = {}
        self.scaler = StandardScaler()
        self.optimization_strategies = {}
        
        # Initialize learning models
        self._initialize_learning_models()
        
        logger.info("Biometric Fusion Engine initialized")
    
    def fuse_biometric_data(self, hrv_metrics: Dict[str, Any], 
                           gsr_metrics: Dict[str, Any],
                           additional_data: Optional[Dict[str, Any]] = None) -> FusedBiometricState:
        """
        Fuse multiple biometric data sources
        
        Args:
            hrv_metrics: HRV analysis metrics
            gsr_metrics: GSR analysis metrics
            additional_data: Additional biometric data (EEG, eye-tracking, etc.)
            
        Returns:
            Fused biometric state
        """
        try:
            # Extract key features from HRV
            hrv_features = self._extract_hrv_features(hrv_metrics)
            
            # Extract key features from GSR
            gsr_features = self._extract_gsr_features(gsr_metrics)
            
            # Extract additional features if available
            additional_features = self._extract_additional_features(additional_data)
            
            # Combine all features
            combined_features = {**hrv_features, **gsr_features, **additional_features}
            
            # Normalize features
            normalized_features = self._normalize_features(combined_features)
            
            # Predict learning state
            learning_state = self._predict_learning_state(normalized_features)
            
            # Calculate derived metrics
            derived_metrics = self._calculate_derived_metrics(hrv_features, gsr_features, additional_features)
            
            # Assess combined biometric state
            combined_state = self._assess_combined_state(derived_metrics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(combined_state, derived_metrics)
            
            # Create fused biometric state
            fused_state = FusedBiometricState(
                timestamp=datetime.now(),
                hrv_state=hrv_metrics.get('hrv_state', 'unknown'),
                gsr_state=gsr_metrics.get('emotional_state', 'unknown'),
                combined_state=combined_state,
                learning_readiness=derived_metrics['learning_readiness'],
                stress_level=derived_metrics['stress_level'],
                engagement_level=derived_metrics['engagement_level'],
                fatigue_level=derived_metrics['fatigue_level'],
                arousal_level=derived_metrics['arousal_level'],
                emotional_state=derived_metrics['emotional_state'],
                cognitive_load=derived_metrics['cognitive_load'],
                attention_level=derived_metrics['attention_level'],
                motivation_level=derived_metrics['motivation_level'],
                confidence=derived_metrics['confidence'],
                recommendations=recommendations
            )
            
            # Store in history
            self.fusion_history.append(fused_state)
            
            return fused_state
            
        except Exception as e:
            logger.error(f"Error fusing biometric data: {str(e)}")
            return self._create_default_state()
    
    def optimize_learning_content(self, biometric_state: FusedBiometricState, 
                                current_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize learning content based on biometric state
        
        Args:
            biometric_state: Current fused biometric state
            current_content: Current learning content parameters
            
        Returns:
            Optimized content parameters
        """
        try:
            # Determine optimization strategy
            strategy = self._determine_optimization_strategy(biometric_state)
            
            # Apply optimization
            optimized_content = self._apply_optimization(strategy, current_content, biometric_state)
            
            return optimized_content
            
        except Exception as e:
            logger.error(f"Error optimizing learning content: {str(e)}")
            return current_content
    
    def predict_learning_outcome(self, biometric_state: FusedBiometricState, 
                               content_difficulty: float) -> Dict[str, Any]:
        """
        Predict learning outcome based on biometric state
        
        Args:
            biometric_state: Current fused biometric state
            content_difficulty: Content difficulty level (0-1)
            
        Returns:
            Learning outcome prediction
        """
        try:
            # Prepare features for prediction
            features = self._prepare_prediction_features(biometric_state, content_difficulty)
            
            # Predict learning outcome
            outcome_prediction = self._predict_outcome(features)
            
            return outcome_prediction
            
        except Exception as e:
            logger.error(f"Error predicting learning outcome: {str(e)}")
            return {'success_probability': 0.5, 'estimated_time': 30, 'confidence': 0.3}
    
    def _extract_hrv_features(self, hrv_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract key features from HRV metrics"""
        try:
            return {
                'hrv_rmssd': hrv_metrics.get('rmssd', 0),
                'hrv_sdnn': hrv_metrics.get('sdnn', 0),
                'hrv_lf_hf_ratio': hrv_metrics.get('lf_hf_ratio', 0),
                'hrv_stress_index': hrv_metrics.get('stress_index', 0),
                'hrv_autonomic_balance': hrv_metrics.get('autonomic_balance', 0),
                'hrv_recovery_index': hrv_metrics.get('recovery_index', 0)
            }
        except Exception as e:
            logger.error(f"Error extracting HRV features: {str(e)}")
            return {}
    
    def _extract_gsr_features(self, gsr_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract key features from GSR metrics"""
        try:
            return {
                'gsr_tonic': gsr_metrics.get('tonic_gsr', 0),
                'gsr_phasic': gsr_metrics.get('phasic_gsr', 0),
                'gsr_arousal_index': gsr_metrics.get('arousal_index', 0),
                'gsr_emotional_reactivity': gsr_metrics.get('emotional_reactivity', 0),
                'gsr_stress_response': gsr_metrics.get('stress_response', 0),
                'gsr_engagement_score': gsr_metrics.get('engagement_score', 0)
            }
        except Exception as e:
            logger.error(f"Error extracting GSR features: {str(e)}")
            return {}
    
    def _extract_additional_features(self, additional_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Extract features from additional biometric data"""
        try:
            if not additional_data:
                return {}
            
            features = {}
            
            # EEG features
            if 'eeg' in additional_data:
                eeg_data = additional_data['eeg']
                features.update({
                    'eeg_alpha_power': eeg_data.get('alpha_power', 0),
                    'eeg_beta_power': eeg_data.get('beta_power', 0),
                    'eeg_theta_power': eeg_data.get('theta_power', 0),
                    'eeg_focus_score': eeg_data.get('focus_score', 0)
                })
            
            # Eye-tracking features
            if 'eye_tracking' in additional_data:
                eye_data = additional_data['eye_tracking']
                features.update({
                    'eye_attention_score': eye_data.get('attention_score', 0),
                    'eye_fixation_duration': eye_data.get('fixation_duration', 0),
                    'eye_saccade_rate': eye_data.get('saccade_rate', 0)
                })
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting additional features: {str(e)}")
            return {}
    
    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features for model input"""
        try:
            # Convert to array for scaling
            feature_names = list(features.keys())
            feature_values = np.array(list(features.values())).reshape(1, -1)
            
            # Normalize features
            normalized_values = self.scaler.fit_transform(feature_values)[0]
            
            # Convert back to dictionary
            normalized_features = dict(zip(feature_names, normalized_values))
            
            return normalized_features
            
        except Exception as e:
            logger.error(f"Error normalizing features: {str(e)}")
            return features
    
    def _predict_learning_state(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict learning state from features"""
        try:
            # Simple rule-based prediction (can be replaced with ML model)
            learning_readiness = self._calculate_learning_readiness(features)
            stress_level = self._calculate_stress_level(features)
            engagement_level = self._calculate_engagement_level(features)
            
            return {
                'learning_readiness': learning_readiness,
                'stress_level': stress_level,
                'engagement_level': engagement_level
            }
            
        except Exception as e:
            logger.error(f"Error predicting learning state: {str(e)}")
            return {'learning_readiness': 0.5, 'stress_level': 0.5, 'engagement_level': 0.5}
    
    def _calculate_derived_metrics(self, hrv_features: Dict[str, float], 
                                 gsr_features: Dict[str, float], 
                                 additional_features: Dict[str, float]) -> Dict[str, float]:
        """Calculate derived metrics from all features"""
        try:
            # Learning readiness (combination of HRV and GSR)
            hrv_readiness = 1.0 - hrv_features.get('hrv_stress_index', 0.5)
            gsr_readiness = gsr_features.get('gsr_engagement_score', 0.5)
            learning_readiness = (hrv_readiness + gsr_readiness) / 2
            
            # Stress level (weighted combination)
            hrv_stress = hrv_features.get('hrv_stress_index', 0.5)
            gsr_stress = gsr_features.get('gsr_stress_response', 0.5)
            stress_level = (hrv_stress * 0.6 + gsr_stress * 0.4)
            
            # Engagement level
            engagement_level = gsr_features.get('gsr_engagement_score', 0.5)
            
            # Fatigue level (inverse of recovery)
            fatigue_level = 1.0 - hrv_features.get('hrv_recovery_index', 0.5) / 10.0
            
            # Arousal level
            arousal_level = gsr_features.get('gsr_arousal_index', 0.5)
            
            # Emotional state (categorical)
            emotional_state = self._determine_emotional_state(hrv_features, gsr_features)
            
            # Cognitive load (combination of stress and attention)
            cognitive_load = (stress_level + (1.0 - additional_features.get('eeg_focus_score', 0.5))) / 2
            
            # Attention level
            attention_level = additional_features.get('eeg_focus_score', 0.5)
            
            # Motivation level (combination of engagement and arousal)
            motivation_level = (engagement_level + arousal_level) / 2
            
            # Confidence (based on data quality)
            confidence = self._calculate_confidence(hrv_features, gsr_features, additional_features)
            
            return {
                'learning_readiness': learning_readiness,
                'stress_level': stress_level,
                'engagement_level': engagement_level,
                'fatigue_level': fatigue_level,
                'arousal_level': arousal_level,
                'emotional_state': emotional_state,
                'cognitive_load': cognitive_load,
                'attention_level': attention_level,
                'motivation_level': motivation_level,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error calculating derived metrics: {str(e)}")
            return {
                'learning_readiness': 0.5, 'stress_level': 0.5, 'engagement_level': 0.5,
                'fatigue_level': 0.5, 'arousal_level': 0.5, 'emotional_state': 'neutral',
                'cognitive_load': 0.5, 'attention_level': 0.5, 'motivation_level': 0.5, 'confidence': 0.5
            }
    
    def _assess_combined_state(self, derived_metrics: Dict[str, float]) -> FusedBiometricState:
        """Assess combined biometric state"""
        try:
            learning_readiness = derived_metrics['learning_readiness']
            stress_level = derived_metrics['stress_level']
            fatigue_level = derived_metrics['fatigue_level']
            engagement_level = derived_metrics['engagement_level']
            
            # Optimal learning conditions
            if (learning_readiness > 0.7 and stress_level < 0.4 and 
                fatigue_level < 0.3 and engagement_level > 0.6):
                return BiometricState.OPTIMAL_LEARNING
            
            # Stressed learning
            elif stress_level > 0.7:
                return BiometricState.STRESSED_LEARNING
            
            # Fatigued learning
            elif fatigue_level > 0.7:
                return BiometricState.FATIGUED_LEARNING
            
            # Overstimulated
            elif (stress_level > 0.6 and engagement_level > 0.8):
                return BiometricState.OVERSTIMULATED
            
            # Understimulated
            elif (engagement_level < 0.3 and learning_readiness < 0.4):
                return BiometricState.UNDERSTIMULATED
            
            # Recovering
            elif (learning_readiness > 0.5 and stress_level < 0.6):
                return BiometricState.RECOVERING
            
            # Not recommended
            else:
                return BiometricState.NOT_RECOMMENDED
                
        except Exception as e:
            logger.error(f"Error assessing combined state: {str(e)}")
            return BiometricState.OPTIMAL_LEARNING
    
    def _generate_recommendations(self, combined_state: FusedBiometricState, 
                                derived_metrics: Dict[str, float]) -> List[str]:
        """Generate learning recommendations based on combined state"""
        try:
            recommendations = []
            
            if combined_state == BiometricState.OPTIMAL_LEARNING:
                recommendations.extend([
                    "Optimal learning state - proceed with challenging content",
                    "Consider advanced topics or complex problem-solving",
                    "Perfect time for intensive study sessions"
                ])
            elif combined_state == BiometricState.STRESSED_LEARNING:
                recommendations.extend([
                    "High stress detected - take a break",
                    "Practice stress management techniques",
                    "Consider easier content or relaxation exercises"
                ])
            elif combined_state == BiometricState.FATIGUED_LEARNING:
                recommendations.extend([
                    "Fatigue detected - take a rest break",
                    "Consider light physical activity",
                    "Ensure adequate sleep before next session"
                ])
            elif combined_state == BiometricState.OVERSTIMULATED:
                recommendations.extend([
                    "Overstimulation detected - reduce content intensity",
                    "Take a longer break",
                    "Consider calming activities"
                ])
            elif combined_state == BiometricState.UNDERSTIMULATED:
                recommendations.extend([
                    "Understimulation detected - increase content difficulty",
                    "Try more engaging or interactive content",
                    "Consider changing learning environment"
                ])
            elif combined_state == BiometricState.RECOVERING:
                recommendations.extend([
                    "Recovery state - gradually increase learning intensity",
                    "Monitor for signs of stress or fatigue",
                    "Consider moderate difficulty content"
                ])
            else:  # NOT_RECOMMENDED
                recommendations.extend([
                    "Learning not recommended - take a break",
                    "Focus on stress reduction and recovery",
                    "Resume learning when feeling better"
                ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Monitor your physiological state and adjust learning accordingly"]
    
    def _determine_optimization_strategy(self, biometric_state: FusedBiometricState) -> LearningOptimization:
        """Determine optimization strategy based on biometric state"""
        try:
            if biometric_state == BiometricState.OPTIMAL_LEARNING:
                return LearningOptimization.INCREASE_DIFFICULTY
            elif biometric_state in [BiometricState.STRESSED_LEARNING, BiometricState.OVERSTIMULATED]:
                return LearningOptimization.DECREASE_DIFFICULTY
            elif biometric_state == BiometricState.FATIGUED_LEARNING:
                return LearningOptimization.TAKE_BREAK
            elif biometric_state == BiometricState.UNDERSTIMULATED:
                return LearningOptimization.INCREASE_ENGAGEMENT
            elif biometric_state == BiometricState.RECOVERING:
                return LearningOptimization.MAINTAIN_CURRENT
            else:  # NOT_RECOMMENDED
                return LearningOptimization.TAKE_BREAK
                
        except Exception as e:
            logger.error(f"Error determining optimization strategy: {str(e)}")
            return LearningOptimization.MAINTAIN_CURRENT
    
    def _apply_optimization(self, strategy: LearningOptimization, 
                          current_content: Dict[str, Any], 
                          biometric_state: FusedBiometricState) -> Dict[str, Any]:
        """Apply optimization strategy to content"""
        try:
            optimized_content = current_content.copy()
            
            if strategy == LearningOptimization.INCREASE_DIFFICULTY:
                optimized_content['difficulty'] = min(1.0, current_content.get('difficulty', 0.5) + 0.2)
                optimized_content['complexity'] = min(1.0, current_content.get('complexity', 0.5) + 0.1)
            elif strategy == LearningOptimization.DECREASE_DIFFICULTY:
                optimized_content['difficulty'] = max(0.1, current_content.get('difficulty', 0.5) - 0.2)
                optimized_content['complexity'] = max(0.1, current_content.get('complexity', 0.5) - 0.1)
            elif strategy == LearningOptimization.INCREASE_ENGAGEMENT:
                optimized_content['interactivity'] = min(1.0, current_content.get('interactivity', 0.5) + 0.3)
                optimized_content['gamification'] = min(1.0, current_content.get('gamification', 0.5) + 0.2)
            elif strategy == LearningOptimization.REDUCE_COGNITIVE_LOAD:
                optimized_content['cognitive_load'] = max(0.1, current_content.get('cognitive_load', 0.5) - 0.3)
                optimized_content['chunk_size'] = max(0.1, current_content.get('chunk_size', 0.5) - 0.2)
            
            return optimized_content
            
        except Exception as e:
            logger.error(f"Error applying optimization: {str(e)}")
            return current_content
    
    def _prepare_prediction_features(self, biometric_state: FusedBiometricState, 
                                   content_difficulty: float) -> np.ndarray:
        """Prepare features for outcome prediction"""
        try:
            features = [
                biometric_state.learning_readiness,
                biometric_state.stress_level,
                biometric_state.engagement_level,
                biometric_state.fatigue_level,
                biometric_state.arousal_level,
                biometric_state.cognitive_load,
                biometric_state.attention_level,
                biometric_state.motivation_level,
                content_difficulty
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing prediction features: {str(e)}")
            return np.array([0.5] * 9).reshape(1, -1)
    
    def _predict_outcome(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict learning outcome"""
        try:
            # Simple rule-based prediction (can be replaced with ML model)
            learning_readiness = features[0, 0]
            stress_level = features[0, 1]
            engagement_level = features[0, 2]
            content_difficulty = features[0, 8]
            
            # Calculate success probability
            success_probability = (learning_readiness + engagement_level - stress_level) / 2
            success_probability = max(0.0, min(1.0, success_probability))
            
            # Adjust for content difficulty
            if content_difficulty > learning_readiness:
                success_probability *= 0.7
            
            # Estimate time (in minutes)
            base_time = 30
            time_multiplier = 1.0 + (1.0 - learning_readiness) + stress_level
            estimated_time = base_time * time_multiplier
            
            # Confidence
            confidence = (learning_readiness + engagement_level) / 2
            
            return {
                'success_probability': success_probability,
                'estimated_time': estimated_time,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error predicting outcome: {str(e)}")
            return {'success_probability': 0.5, 'estimated_time': 30, 'confidence': 0.3}
    
    def _calculate_learning_readiness(self, features: Dict[str, float]) -> float:
        """Calculate learning readiness from features"""
        try:
            hrv_readiness = 1.0 - features.get('hrv_stress_index', 0.5)
            gsr_readiness = features.get('gsr_engagement_score', 0.5)
            return (hrv_readiness + gsr_readiness) / 2
        except Exception as e:
            logger.error(f"Error calculating learning readiness: {str(e)}")
            return 0.5
    
    def _calculate_stress_level(self, features: Dict[str, float]) -> float:
        """Calculate stress level from features"""
        try:
            hrv_stress = features.get('hrv_stress_index', 0.5)
            gsr_stress = features.get('gsr_stress_response', 0.5)
            return (hrv_stress + gsr_stress) / 2
        except Exception as e:
            logger.error(f"Error calculating stress level: {str(e)}")
            return 0.5
    
    def _calculate_engagement_level(self, features: Dict[str, float]) -> float:
        """Calculate engagement level from features"""
        try:
            return features.get('gsr_engagement_score', 0.5)
        except Exception as e:
            logger.error(f"Error calculating engagement level: {str(e)}")
            return 0.5
    
    def _determine_emotional_state(self, hrv_features: Dict[str, float], 
                                 gsr_features: Dict[str, float]) -> str:
        """Determine emotional state from features"""
        try:
            stress_level = (hrv_features.get('hrv_stress_index', 0.5) + 
                          gsr_features.get('gsr_stress_response', 0.5)) / 2
            arousal_level = gsr_features.get('gsr_arousal_index', 0.5)
            
            if stress_level > 0.7:
                return 'stressed'
            elif stress_level > 0.5 and arousal_level > 0.6:
                return 'anxious'
            elif arousal_level > 0.7:
                return 'excited'
            elif arousal_level < 0.3:
                return 'calm'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error determining emotional state: {str(e)}")
            return 'neutral'
    
    def _calculate_confidence(self, hrv_features: Dict[str, float], 
                            gsr_features: Dict[str, float], 
                            additional_features: Dict[str, float]) -> float:
        """Calculate confidence in fused state"""
        try:
            # Base confidence on data availability
            data_quality = 0.5  # Base quality
            
            if hrv_features:
                data_quality += 0.2
            if gsr_features:
                data_quality += 0.2
            if additional_features:
                data_quality += 0.1
            
            return min(1.0, data_quality)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def _initialize_learning_models(self):
        """Initialize machine learning models"""
        try:
            # Initialize models for different predictions
            self.learning_models = {
                'learning_outcome': RandomForestRegressor(n_estimators=100, random_state=42),
                'content_optimization': RandomForestRegressor(n_estimators=50, random_state=42)
            }
            
            logger.info("Learning models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing learning models: {str(e)}")
    
    def _create_default_state(self) -> FusedBiometricState:
        """Create default biometric state when fusion fails"""
        return FusedBiometricState(
            timestamp=datetime.now(),
            hrv_state='unknown',
            gsr_state='unknown',
            combined_state=BiometricState.OPTIMAL_LEARNING,
            learning_readiness=0.5,
            stress_level=0.5,
            engagement_level=0.5,
            fatigue_level=0.5,
            arousal_level=0.5,
            emotional_state='neutral',
            cognitive_load=0.5,
            attention_level=0.5,
            motivation_level=0.5,
            confidence=0.3,
            recommendations=["Monitor your physiological state and adjust learning accordingly"]
        )
    
    def get_fusion_statistics(self) -> Dict[str, int]:
        """Get biometric fusion statistics"""
        try:
            return {
                'total_fusions': len(self.fusion_history),
                'learning_models': len(self.learning_models),
                'optimization_strategies': len(self.optimization_strategies)
            }
            
        except Exception as e:
            logger.error(f"Error getting fusion statistics: {str(e)}")
            return {
                'total_fusions': 0,
                'learning_models': 0,
                'optimization_strategies': 0
            }
