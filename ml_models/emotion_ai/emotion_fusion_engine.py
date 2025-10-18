"""
Emotion Fusion Engine Module

This module provides comprehensive emotion fusion and analysis including:
- Multi-modal emotion data fusion
- Emotion state prediction
- Emotional trend analysis
- Emotion-based learning adaptation
- Real-time emotion monitoring

Author: LearnStyle AI Team
Version: 1.0.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EmotionState(Enum):
    """Overall emotion state classifications"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"
    MIXED = "mixed"
    CONFUSED = "confused"
    OVERWHELMED = "overwhelmed"

class EmotionTrend(Enum):
    """Emotion trend classifications"""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"

class LearningEmotionState(Enum):
    """Learning-specific emotion states"""
    MOTIVATED = "motivated"
    ENGAGED = "engaged"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    BORED = "bored"
    OVERWHELMED = "overwhelmed"
    CONFIDENT = "confident"
    ANXIOUS = "anxious"

@dataclass
class FusedEmotionMetrics:
    """Comprehensive fused emotion analysis metrics"""
    overall_emotion_state: EmotionState
    emotion_confidence: float
    emotion_intensity: float
    emotion_trend: EmotionTrend
    learning_emotion_state: LearningEmotionState
    facial_emotion_weight: float
    voice_emotion_weight: float
    attention_emotion_weight: float
    emotion_stability: float
    emotional_engagement: float
    learning_readiness: float
    emotion_timeline: List[Dict]
    confidence: float

class EmotionFusionEngine:
    """
    Advanced emotion fusion and analysis system
    """
    
    def __init__(self, 
                 fusion_threshold: float = 0.6,
                 trend_window: int = 10,
                 stability_threshold: float = 0.7):
        """
        Initialize emotion fusion engine
        
        Args:
            fusion_threshold: Threshold for emotion fusion confidence
            trend_window: Window size for trend analysis
            stability_threshold: Threshold for emotion stability
        """
        self.fusion_threshold = fusion_threshold
        self.trend_window = trend_window
        self.stability_threshold = stability_threshold
        
        # Emotion state thresholds
        self.emotion_thresholds = {
            EmotionState.VERY_POSITIVE: 0.8,
            EmotionState.POSITIVE: 0.6,
            EmotionState.NEUTRAL: 0.4,
            EmotionState.NEGATIVE: 0.2,
            EmotionState.VERY_NEGATIVE: 0.0
        }
        
        # Learning emotion state mappings
        self.learning_emotion_mappings = {
            'happy': LearningEmotionState.MOTIVATED,
            'excited': LearningEmotionState.ENGAGED,
            'focused': LearningEmotionState.CONFIDENT,
            'confused': LearningEmotionState.CONFUSED,
            'frustrated': LearningEmotionState.FRUSTRATED,
            'bored': LearningEmotionState.BORED,
            'overwhelmed': LearningEmotionState.OVERWHELMED,
            'anxious': LearningEmotionState.ANXIOUS
        }
        
        # Historical data for trend analysis
        self.emotion_history = []
        self.fusion_history = []
        
        logger.info("Emotion Fusion Engine initialized")
    
    def fuse_emotions(self, facial_emotion: Dict, voice_emotion: Dict, 
                     attention_emotion: Dict) -> FusedEmotionMetrics:
        """
        Fuse emotions from multiple modalities
        
        Args:
            facial_emotion: Facial emotion analysis data
            voice_emotion: Voice emotion analysis data
            attention_emotion: Attention emotion analysis data
            
        Returns:
            FusedEmotionMetrics object with comprehensive analysis
        """
        try:
            # Extract emotion data from each modality
            facial_data = self._extract_facial_emotion_data(facial_emotion)
            voice_data = self._extract_voice_emotion_data(voice_emotion)
            attention_data = self._extract_attention_emotion_data(attention_emotion)
            
            # Calculate modality weights based on confidence and availability
            weights = self._calculate_modality_weights(facial_data, voice_data, attention_data)
            
            # Fuse emotions using weighted combination
            fused_emotion, emotion_confidence = self._fuse_emotion_scores(
                facial_data, voice_data, attention_data, weights
            )
            
            # Calculate emotion intensity
            emotion_intensity = self._calculate_emotion_intensity(
                facial_data, voice_data, attention_data, weights
            )
            
            # Determine overall emotion state
            overall_emotion_state = self._classify_overall_emotion_state(fused_emotion, emotion_confidence)
            
            # Determine learning emotion state
            learning_emotion_state = self._classify_learning_emotion_state(
                facial_data, voice_data, attention_data
            )
            
            # Analyze emotion trend
            emotion_trend = self._analyze_emotion_trend()
            
            # Calculate emotion stability
            emotion_stability = self._calculate_emotion_stability()
            
            # Calculate emotional engagement
            emotional_engagement = self._calculate_emotional_engagement(
                facial_data, voice_data, attention_data, weights
            )
            
            # Calculate learning readiness
            learning_readiness = self._calculate_learning_readiness(
                overall_emotion_state, learning_emotion_state, emotional_engagement
            )
            
            # Create emotion timeline entry
            emotion_entry = {
                'timestamp': datetime.now().isoformat(),
                'fused_emotion': fused_emotion,
                'emotion_confidence': emotion_confidence,
                'emotion_intensity': emotion_intensity,
                'overall_state': overall_emotion_state.value,
                'learning_state': learning_emotion_state.value,
                'emotional_engagement': emotional_engagement,
                'learning_readiness': learning_readiness
            }
            
            # Update history
            self._update_history(emotion_entry)
            
            # Calculate overall confidence
            confidence = self._calculate_fusion_confidence(facial_data, voice_data, attention_data, weights)
            
            return FusedEmotionMetrics(
                overall_emotion_state=overall_emotion_state,
                emotion_confidence=emotion_confidence,
                emotion_intensity=emotion_intensity,
                emotion_trend=emotion_trend,
                learning_emotion_state=learning_emotion_state,
                facial_emotion_weight=weights['facial'],
                voice_emotion_weight=weights['voice'],
                attention_emotion_weight=weights['attention'],
                emotion_stability=emotion_stability,
                emotional_engagement=emotional_engagement,
                learning_readiness=learning_readiness,
                emotion_timeline=self.emotion_history[-20:],  # Last 20 entries
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error fusing emotions: {str(e)}")
            return self._get_default_metrics()
    
    def _extract_facial_emotion_data(self, facial_emotion: Dict) -> Dict:
        """Extract facial emotion data"""
        try:
            return {
                'emotion_type': facial_emotion.get('primary_emotion', 'neutral'),
                'confidence': facial_emotion.get('emotion_confidence', 0.5),
                'intensity': facial_emotion.get('emotion_intensity', 0.5),
                'engagement': facial_emotion.get('engagement_score', 0.5),
                'attention': facial_emotion.get('attention_score', 0.5),
                'data_quality': facial_emotion.get('confidence', 0.5)
            }
        except Exception as e:
            logger.error(f"Error extracting facial emotion data: {str(e)}")
            return {'emotion_type': 'neutral', 'confidence': 0.1, 'intensity': 0.5, 
                   'engagement': 0.5, 'attention': 0.5, 'data_quality': 0.1}
    
    def _extract_voice_emotion_data(self, voice_emotion: Dict) -> Dict:
        """Extract voice emotion data"""
        try:
            return {
                'emotion_type': voice_emotion.get('primary_emotion', 'neutral'),
                'confidence': voice_emotion.get('emotion_confidence', 0.5),
                'sentiment': voice_emotion.get('sentiment_score', 0.5),
                'engagement': voice_emotion.get('engagement_score', 0.5),
                'stress': voice_emotion.get('stress_score', 0.5),
                'data_quality': voice_emotion.get('confidence', 0.5)
            }
        except Exception as e:
            logger.error(f"Error extracting voice emotion data: {str(e)}")
            return {'emotion_type': 'neutral', 'confidence': 0.1, 'sentiment': 0.5,
                   'engagement': 0.5, 'stress': 0.5, 'data_quality': 0.1}
    
    def _extract_attention_emotion_data(self, attention_emotion: Dict) -> Dict:
        """Extract attention emotion data"""
        try:
            return {
                'attention_state': attention_emotion.get('attention_state', 'moderately_attentive'),
                'attention_score': attention_emotion.get('attention_score', 0.5),
                'engagement_level': attention_emotion.get('engagement_level', 'medium'),
                'engagement_score': attention_emotion.get('engagement_score', 0.5),
                'focus_quality': attention_emotion.get('focus_quality', 0.5),
                'data_quality': attention_emotion.get('confidence', 0.5)
            }
        except Exception as e:
            logger.error(f"Error extracting attention emotion data: {str(e)}")
            return {'attention_state': 'moderately_attentive', 'attention_score': 0.5,
                   'engagement_level': 'medium', 'engagement_score': 0.5, 'focus_quality': 0.5,
                   'data_quality': 0.1}
    
    def _calculate_modality_weights(self, facial_data: Dict, voice_data: Dict, 
                                  attention_data: Dict) -> Dict[str, float]:
        """Calculate weights for each modality based on data quality and availability"""
        try:
            weights = {}
            
            # Calculate weights based on data quality and confidence
            facial_weight = facial_data['data_quality'] * facial_data['confidence']
            voice_weight = voice_data['data_quality'] * voice_data['confidence']
            attention_weight = attention_data['data_quality'] * attention_data['confidence']
            
            # Normalize weights
            total_weight = facial_weight + voice_weight + attention_weight
            
            if total_weight > 0:
                weights['facial'] = facial_weight / total_weight
                weights['voice'] = voice_weight / total_weight
                weights['attention'] = attention_weight / total_weight
            else:
                # Equal weights if no data
                weights['facial'] = 0.33
                weights['voice'] = 0.33
                weights['attention'] = 0.34
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating modality weights: {str(e)}")
            return {'facial': 0.33, 'voice': 0.33, 'attention': 0.34}
    
    def _fuse_emotion_scores(self, facial_data: Dict, voice_data: Dict, 
                           attention_data: Dict, weights: Dict[str, float]) -> Tuple[str, float]:
        """
        Fuse emotion scores from multiple modalities
        
        Args:
            facial_data: Facial emotion data
            voice_data: Voice emotion data
            attention_data: Attention emotion data
            weights: Modality weights
            
        Returns:
            Tuple of (fused_emotion, confidence)
        """
        try:
            # Convert emotions to numerical scores
            emotion_scores = self._convert_emotions_to_scores()
            
            # Calculate weighted emotion scores
            fused_scores = {}
            for emotion in emotion_scores:
                facial_score = emotion_scores[emotion].get(facial_data['emotion_type'], 0) * facial_data['confidence']
                voice_score = emotion_scores[emotion].get(voice_data['emotion_type'], 0) * voice_data['confidence']
                attention_score = self._get_attention_emotion_score(attention_data, emotion)
                
                fused_scores[emotion] = (
                    facial_score * weights['facial'] +
                    voice_score * weights['voice'] +
                    attention_score * weights['attention']
                )
            
            # Find emotion with highest score
            max_emotion = max(fused_scores.items(), key=lambda x: x[1])
            fused_emotion = max_emotion[0]
            confidence = max_emotion[1]
            
            return fused_emotion, confidence
            
        except Exception as e:
            logger.error(f"Error fusing emotion scores: {str(e)}")
            return 'neutral', 0.1
    
    def _convert_emotions_to_scores(self) -> Dict[str, Dict[str, float]]:
        """Convert emotions to numerical scores for fusion"""
        emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral', 
                   'excited', 'frustrated', 'confused', 'bored', 'focused']
        
        emotion_scores = {}
        for emotion in emotions:
            scores = {}
            for other_emotion in emotions:
                if emotion == other_emotion:
                    scores[other_emotion] = 1.0
                elif self._are_emotions_similar(emotion, other_emotion):
                    scores[other_emotion] = 0.7
                elif self._are_emotions_opposite(emotion, other_emotion):
                    scores[other_emotion] = 0.1
                else:
                    scores[other_emotion] = 0.3
            emotion_scores[emotion] = scores
        
        return emotion_scores
    
    def _are_emotions_similar(self, emotion1: str, emotion2: str) -> bool:
        """Check if two emotions are similar"""
        similar_groups = [
            ['happy', 'excited', 'focused'],
            ['sad', 'bored'],
            ['angry', 'frustrated'],
            ['fear', 'anxious'],
            ['confused', 'overwhelmed']
        ]
        
        for group in similar_groups:
            if emotion1 in group and emotion2 in group:
                return True
        return False
    
    def _are_emotions_opposite(self, emotion1: str, emotion2: str) -> bool:
        """Check if two emotions are opposite"""
        opposite_pairs = [
            ('happy', 'sad'),
            ('excited', 'bored'),
            ('confident', 'anxious'),
            ('focused', 'confused')
        ]
        
        for pair in opposite_pairs:
            if (emotion1 in pair and emotion2 in pair) and emotion1 != emotion2:
                return True
        return False
    
    def _get_attention_emotion_score(self, attention_data: Dict, emotion: str) -> float:
        """Get emotion score from attention data"""
        try:
            attention_state = attention_data['attention_state']
            engagement_level = attention_data['engagement_level']
            
            # Map attention states to emotions
            attention_emotion_mapping = {
                'highly_focused': 'focused',
                'focused': 'focused',
                'moderately_attentive': 'neutral',
                'distracted': 'confused',
                'highly_distracted': 'bored',
                'confused': 'confused',
                'overwhelmed': 'overwhelmed'
            }
            
            attention_emotion = attention_emotion_mapping.get(attention_state, 'neutral')
            
            if attention_emotion == emotion:
                return 1.0
            elif self._are_emotions_similar(attention_emotion, emotion):
                return 0.7
            else:
                return 0.3
                
        except Exception as e:
            logger.error(f"Error getting attention emotion score: {str(e)}")
            return 0.3
    
    def _calculate_emotion_intensity(self, facial_data: Dict, voice_data: Dict, 
                                   attention_data: Dict, weights: Dict[str, float]) -> float:
        """Calculate fused emotion intensity"""
        try:
            facial_intensity = facial_data['intensity'] * facial_data['confidence']
            voice_intensity = voice_data['sentiment'] * voice_data['confidence']  # Use sentiment as intensity proxy
            attention_intensity = attention_data['focus_quality'] * attention_data['data_quality']
            
            fused_intensity = (
                facial_intensity * weights['facial'] +
                voice_intensity * weights['voice'] +
                attention_intensity * weights['attention']
            )
            
            return max(0, min(1, fused_intensity))
            
        except Exception as e:
            logger.error(f"Error calculating emotion intensity: {str(e)}")
            return 0.5
    
    def _classify_overall_emotion_state(self, fused_emotion: str, confidence: float) -> EmotionState:
        """Classify overall emotion state from fused emotion"""
        try:
            # Map emotions to states
            emotion_state_mapping = {
                'happy': EmotionState.POSITIVE,
                'excited': EmotionState.VERY_POSITIVE,
                'focused': EmotionState.POSITIVE,
                'neutral': EmotionState.NEUTRAL,
                'sad': EmotionState.NEGATIVE,
                'bored': EmotionState.NEGATIVE,
                'angry': EmotionState.NEGATIVE,
                'frustrated': EmotionState.NEGATIVE,
                'fear': EmotionState.NEGATIVE,
                'anxious': EmotionState.NEGATIVE,
                'confused': EmotionState.CONFUSED,
                'overwhelmed': EmotionState.OVERWHELMED
            }
            
            base_state = emotion_state_mapping.get(fused_emotion, EmotionState.NEUTRAL)
            
            # Adjust based on confidence
            if confidence < 0.3:
                return EmotionState.MIXED
            elif confidence > 0.8 and base_state in [EmotionState.POSITIVE, EmotionState.NEGATIVE]:
                if base_state == EmotionState.POSITIVE:
                    return EmotionState.VERY_POSITIVE
                else:
                    return EmotionState.VERY_NEGATIVE
            
            return base_state
            
        except Exception as e:
            logger.error(f"Error classifying overall emotion state: {str(e)}")
            return EmotionState.NEUTRAL
    
    def _classify_learning_emotion_state(self, facial_data: Dict, voice_data: Dict, 
                                       attention_data: Dict) -> LearningEmotionState:
        """Classify learning-specific emotion state"""
        try:
            # Get primary emotions from each modality
            facial_emotion = facial_data['emotion_type']
            voice_emotion = voice_data['emotion_type']
            attention_state = attention_data['attention_state']
            
            # Map to learning emotions
            learning_emotions = []
            
            if facial_emotion in self.learning_emotion_mappings:
                learning_emotions.append(self.learning_emotion_mappings[facial_emotion])
            
            if voice_emotion in self.learning_emotion_mappings:
                learning_emotions.append(self.learning_emotion_mappings[voice_emotion])
            
            # Map attention states to learning emotions
            attention_learning_mapping = {
                'highly_focused': LearningEmotionState.CONFIDENT,
                'focused': LearningEmotionState.ENGAGED,
                'moderately_attentive': LearningEmotionState.MOTIVATED,
                'distracted': LearningEmotionState.CONFUSED,
                'highly_distracted': LearningEmotionState.BORED,
                'confused': LearningEmotionState.CONFUSED,
                'overwhelmed': LearningEmotionState.OVERWHELMED
            }
            
            if attention_state in attention_learning_mapping:
                learning_emotions.append(attention_learning_mapping[attention_state])
            
            # Return most common learning emotion
            if learning_emotions:
                emotion_counts = {}
                for emotion in learning_emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                return max(emotion_counts.items(), key=lambda x: x[1])[0]
            else:
                return LearningEmotionState.MOTIVATED
                
        except Exception as e:
            logger.error(f"Error classifying learning emotion state: {str(e)}")
            return LearningEmotionState.MOTIVATED
    
    def _analyze_emotion_trend(self) -> EmotionTrend:
        """Analyze emotion trend from historical data"""
        try:
            if len(self.emotion_history) < self.trend_window:
                return EmotionTrend.STABLE
            
            # Get recent emotion scores
            recent_emotions = self.emotion_history[-self.trend_window:]
            emotion_scores = [entry['emotion_confidence'] for entry in recent_emotions]
            
            # Calculate trend
            if len(emotion_scores) > 1:
                trend_slope = np.polyfit(range(len(emotion_scores)), emotion_scores, 1)[0]
                
                if trend_slope > 0.1:
                    return EmotionTrend.IMPROVING
                elif trend_slope < -0.1:
                    return EmotionTrend.DECLINING
                else:
                    # Check for volatility
                    emotion_variance = np.var(emotion_scores)
                    if emotion_variance > 0.1:
                        return EmotionTrend.VOLATILE
                    else:
                        return EmotionTrend.STABLE
            else:
                return EmotionTrend.STABLE
                
        except Exception as e:
            logger.error(f"Error analyzing emotion trend: {str(e)}")
            return EmotionTrend.STABLE
    
    def _calculate_emotion_stability(self) -> float:
        """Calculate emotion stability from historical data"""
        try:
            if len(self.emotion_history) < 5:
                return 0.5
            
            # Get recent emotion confidence scores
            recent_emotions = self.emotion_history[-10:]
            confidence_scores = [entry['emotion_confidence'] for entry in recent_emotions]
            
            # Stability is inverse of variance
            stability = 1.0 - np.var(confidence_scores)
            
            return max(0, min(1, stability))
            
        except Exception as e:
            logger.error(f"Error calculating emotion stability: {str(e)}")
            return 0.5
    
    def _calculate_emotional_engagement(self, facial_data: Dict, voice_data: Dict, 
                                      attention_data: Dict, weights: Dict[str, float]) -> float:
        """Calculate emotional engagement from fused data"""
        try:
            facial_engagement = facial_data['engagement'] * facial_data['confidence']
            voice_engagement = voice_data['engagement'] * voice_data['confidence']
            attention_engagement = attention_data['engagement_score'] * attention_data['data_quality']
            
            emotional_engagement = (
                facial_engagement * weights['facial'] +
                voice_engagement * weights['voice'] +
                attention_engagement * weights['attention']
            )
            
            return max(0, min(1, emotional_engagement))
            
        except Exception as e:
            logger.error(f"Error calculating emotional engagement: {str(e)}")
            return 0.5
    
    def _calculate_learning_readiness(self, overall_emotion_state: EmotionState, 
                                    learning_emotion_state: LearningEmotionState, 
                                    emotional_engagement: float) -> float:
        """Calculate learning readiness from emotion data"""
        try:
            # Base readiness from emotion state
            emotion_readiness = {
                EmotionState.VERY_POSITIVE: 0.9,
                EmotionState.POSITIVE: 0.8,
                EmotionState.NEUTRAL: 0.6,
                EmotionState.NEGATIVE: 0.3,
                EmotionState.VERY_NEGATIVE: 0.1,
                EmotionState.MIXED: 0.5,
                EmotionState.CONFUSED: 0.4,
                EmotionState.OVERWHELMED: 0.2
            }
            
            base_readiness = emotion_readiness.get(overall_emotion_state, 0.5)
            
            # Adjust based on learning emotion state
            learning_adjustment = {
                LearningEmotionState.MOTIVATED: 0.2,
                LearningEmotionState.ENGAGED: 0.2,
                LearningEmotionState.CONFIDENT: 0.1,
                LearningEmotionState.CONFUSED: -0.1,
                LearningEmotionState.FRUSTRATED: -0.2,
                LearningEmotionState.BORED: -0.3,
                LearningEmotionState.OVERWHELMED: -0.4,
                LearningEmotionState.ANXIOUS: -0.2
            }
            
            adjustment = learning_adjustment.get(learning_emotion_state, 0.0)
            
            # Factor in emotional engagement
            engagement_factor = emotional_engagement * 0.3
            
            learning_readiness = base_readiness + adjustment + engagement_factor
            
            return max(0, min(1, learning_readiness))
            
        except Exception as e:
            logger.error(f"Error calculating learning readiness: {str(e)}")
            return 0.5
    
    def _calculate_fusion_confidence(self, facial_data: Dict, voice_data: Dict, 
                                   attention_data: Dict, weights: Dict[str, float]) -> float:
        """Calculate overall confidence in fusion"""
        try:
            # Weighted average of individual confidences
            facial_conf = facial_data['confidence'] * facial_data['data_quality']
            voice_conf = voice_data['confidence'] * voice_data['data_quality']
            attention_conf = attention_data['data_quality']
            
            fusion_confidence = (
                facial_conf * weights['facial'] +
                voice_conf * weights['voice'] +
                attention_conf * weights['attention']
            )
            
            return max(0.1, min(1.0, fusion_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating fusion confidence: {str(e)}")
            return 0.1
    
    def _update_history(self, emotion_entry: Dict):
        """Update emotion history for trend analysis"""
        self.emotion_history.append(emotion_entry)
        
        # Keep only recent history
        if len(self.emotion_history) > 100:
            self.emotion_history = self.emotion_history[-100:]
    
    def _get_default_metrics(self) -> FusedEmotionMetrics:
        """Return default metrics when analysis fails"""
        return FusedEmotionMetrics(
            overall_emotion_state=EmotionState.NEUTRAL,
            emotion_confidence=0.1,
            emotion_intensity=0.5,
            emotion_trend=EmotionTrend.STABLE,
            learning_emotion_state=LearningEmotionState.MOTIVATED,
            facial_emotion_weight=0.33,
            voice_emotion_weight=0.33,
            attention_emotion_weight=0.34,
            emotion_stability=0.5,
            emotional_engagement=0.5,
            learning_readiness=0.5,
            emotion_timeline=[],
            confidence=0.1
        )
    
    def get_emotion_statistics(self) -> Dict[str, float]:
        """
        Get emotion statistics from historical data
        
        Returns:
            Dictionary with emotion statistics
        """
        if not self.emotion_history:
            return {
                'dominant_emotion': 'neutral',
                'average_confidence': 0.0,
                'emotion_stability': 0.0,
                'learning_readiness_trend': 0.0
            }
        
        # Calculate statistics
        emotions = [entry['fused_emotion'] for entry in self.emotion_history]
        confidences = [entry['emotion_confidence'] for entry in self.emotion_history]
        learning_readiness = [entry['learning_readiness'] for entry in self.emotion_history]
        
        # Dominant emotion
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Average confidence
        average_confidence = np.mean(confidences)
        
        # Emotion stability
        emotion_stability = 1.0 - np.var(confidences)
        
        # Learning readiness trend
        if len(learning_readiness) > 1:
            learning_readiness_trend = np.polyfit(range(len(learning_readiness)), learning_readiness, 1)[0]
        else:
            learning_readiness_trend = 0.0
        
        return {
            'dominant_emotion': dominant_emotion,
            'average_confidence': float(average_confidence),
            'emotion_stability': float(emotion_stability),
            'learning_readiness_trend': float(learning_readiness_trend)
        }
