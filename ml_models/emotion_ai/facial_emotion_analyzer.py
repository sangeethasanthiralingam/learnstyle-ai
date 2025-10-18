"""
Facial Emotion Analysis Module

This module provides advanced facial expression analysis including:
- Real-time facial emotion detection
- Micro-expression recognition
- Emotion intensity measurement
- Engagement level assessment
- Attention state classification

Author: LearnStyle AI Team
Version: 1.0.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class EmotionType(Enum):
    """Types of emotions detected"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    CONFUSED = "confused"
    FOCUSED = "focused"
    BORED = "bored"

class EngagementLevel(Enum):
    """Engagement level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class AttentionState(Enum):
    """Attention state classifications"""
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    CONFUSED = "confused"
    BORED = "bored"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"

@dataclass
class FacialLandmarks:
    """Facial landmark data"""
    left_eye: Tuple[float, float]
    right_eye: Tuple[float, float]
    nose: Tuple[float, float]
    left_mouth: Tuple[float, float]
    right_mouth: Tuple[float, float]
    chin: Tuple[float, float]
    eyebrow_left: Tuple[float, float]
    eyebrow_right: Tuple[float, float]

@dataclass
class EmotionMetrics:
    """Comprehensive emotion analysis metrics"""
    primary_emotion: EmotionType
    emotion_confidence: float
    emotion_intensity: float
    engagement_level: EngagementLevel
    attention_state: AttentionState
    facial_landmarks: Optional[FacialLandmarks]
    micro_expressions: List[Dict]
    emotion_timeline: List[Dict]
    engagement_score: float
    attention_score: float
    confidence: float

class FacialEmotionAnalyzer:
    """
    Advanced facial emotion analysis system
    """
    
    def __init__(self, 
                 emotion_threshold: float = 0.6,
                 engagement_threshold: float = 0.5,
                 attention_threshold: float = 0.6):
        """
        Initialize facial emotion analyzer
        
        Args:
            emotion_threshold: Threshold for emotion detection confidence
            engagement_threshold: Threshold for engagement level classification
            attention_threshold: Threshold for attention state classification
        """
        self.emotion_threshold = emotion_threshold
        self.engagement_threshold = engagement_threshold
        self.attention_threshold = attention_threshold
        
        # Emotion detection models (simplified - in production would use actual ML models)
        self.emotion_models = self._initialize_emotion_models()
        
        # Engagement level thresholds
        self.engagement_thresholds = {
            EngagementLevel.VERY_LOW: 0.2,
            EngagementLevel.LOW: 0.4,
            EngagementLevel.MEDIUM: 0.6,
            EngagementLevel.HIGH: 0.8,
            EngagementLevel.VERY_HIGH: 0.9
        }
        
        # Historical data for trend analysis
        self.emotion_history = []
        self.engagement_history = []
        
        logger.info("Facial Emotion Analyzer initialized")
    
    def analyze_facial_emotion(self, facial_data: Dict) -> EmotionMetrics:
        """
        Analyze facial emotion from facial data
        
        Args:
            facial_data: Dictionary containing facial analysis data
            
        Returns:
            EmotionMetrics object with comprehensive analysis
        """
        try:
            # Extract facial landmarks
            landmarks = self._extract_facial_landmarks(facial_data)
            
            # Detect primary emotion
            primary_emotion, emotion_confidence = self._detect_primary_emotion(facial_data)
            
            # Calculate emotion intensity
            emotion_intensity = self._calculate_emotion_intensity(facial_data, primary_emotion)
            
            # Detect micro-expressions
            micro_expressions = self._detect_micro_expressions(facial_data)
            
            # Calculate engagement level
            engagement_level, engagement_score = self._calculate_engagement_level(
                facial_data, primary_emotion, emotion_intensity
            )
            
            # Determine attention state
            attention_state, attention_score = self._determine_attention_state(
                facial_data, primary_emotion, engagement_score
            )
            
            # Create emotion timeline entry
            emotion_timeline_entry = {
                'timestamp': datetime.now().isoformat(),
                'emotion': primary_emotion.value,
                'confidence': emotion_confidence,
                'intensity': emotion_intensity,
                'engagement': engagement_score,
                'attention': attention_score
            }
            
            # Update history
            self._update_history(emotion_timeline_entry)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(facial_data, emotion_confidence)
            
            return EmotionMetrics(
                primary_emotion=primary_emotion,
                emotion_confidence=emotion_confidence,
                emotion_intensity=emotion_intensity,
                engagement_level=engagement_level,
                attention_state=attention_state,
                facial_landmarks=landmarks,
                micro_expressions=micro_expressions,
                emotion_timeline=self.emotion_history[-10:],  # Last 10 entries
                engagement_score=engagement_score,
                attention_score=attention_score,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing facial emotion: {str(e)}")
            return self._get_default_metrics()
    
    def _extract_facial_landmarks(self, facial_data: Dict) -> Optional[FacialLandmarks]:
        """
        Extract facial landmarks from facial data
        
        Args:
            facial_data: Facial analysis data
            
        Returns:
            FacialLandmarks object or None
        """
        try:
            landmarks_data = facial_data.get('landmarks', {})
            
            if not landmarks_data:
                return None
            
            return FacialLandmarks(
                left_eye=landmarks_data.get('left_eye', (0, 0)),
                right_eye=landmarks_data.get('right_eye', (0, 0)),
                nose=landmarks_data.get('nose', (0, 0)),
                left_mouth=landmarks_data.get('left_mouth', (0, 0)),
                right_mouth=landmarks_data.get('right_mouth', (0, 0)),
                chin=landmarks_data.get('chin', (0, 0)),
                eyebrow_left=landmarks_data.get('eyebrow_left', (0, 0)),
                eyebrow_right=landmarks_data.get('eyebrow_right', (0, 0))
            )
            
        except Exception as e:
            logger.warning(f"Error extracting facial landmarks: {str(e)}")
            return None
    
    def _detect_primary_emotion(self, facial_data: Dict) -> Tuple[EmotionType, float]:
        """
        Detect primary emotion from facial data
        
        Args:
            facial_data: Facial analysis data
            
        Returns:
            Tuple of (emotion_type, confidence)
        """
        try:
            # Extract emotion scores from facial data
            emotion_scores = facial_data.get('emotion_scores', {})
            
            if not emotion_scores:
                # Generate simulated emotion scores for demonstration
                emotion_scores = self._generate_simulated_emotion_scores()
            
            # Find emotion with highest score
            max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            emotion_type = EmotionType(max_emotion[0])
            confidence = max_emotion[1]
            
            # Apply threshold
            if confidence < self.emotion_threshold:
                emotion_type = EmotionType.NEUTRAL
                confidence = 0.5
            
            return emotion_type, confidence
            
        except Exception as e:
            logger.error(f"Error detecting primary emotion: {str(e)}")
            return EmotionType.NEUTRAL, 0.1
    
    def _generate_simulated_emotion_scores(self) -> Dict[str, float]:
        """Generate simulated emotion scores for demonstration"""
        emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral', 'confused', 'focused', 'bored']
        scores = {}
        
        # Generate random scores with some bias
        for emotion in emotions:
            if emotion == 'neutral':
                scores[emotion] = np.random.uniform(0.3, 0.7)
            elif emotion == 'focused':
                scores[emotion] = np.random.uniform(0.4, 0.8)
            else:
                scores[emotion] = np.random.uniform(0.1, 0.5)
        
        # Normalize scores
        total_score = sum(scores.values())
        for emotion in scores:
            scores[emotion] = scores[emotion] / total_score
        
        return scores
    
    def _calculate_emotion_intensity(self, facial_data: Dict, emotion: EmotionType) -> float:
        """
        Calculate emotion intensity
        
        Args:
            facial_data: Facial analysis data
            emotion: Detected emotion type
            
        Returns:
            Emotion intensity (0-1)
        """
        try:
            # Extract intensity indicators
            facial_tension = facial_data.get('facial_tension', 0.5)
            eye_opening = facial_data.get('eye_opening', 0.5)
            mouth_opening = facial_data.get('mouth_opening', 0.5)
            eyebrow_position = facial_data.get('eyebrow_position', 0.5)
            
            # Calculate intensity based on emotion type
            if emotion == EmotionType.HAPPY:
                intensity = (mouth_opening * 0.4 + eye_opening * 0.3 + (1 - facial_tension) * 0.3)
            elif emotion == EmotionType.SAD:
                intensity = (facial_tension * 0.4 + (1 - eye_opening) * 0.3 + eyebrow_position * 0.3)
            elif emotion == EmotionType.ANGRY:
                intensity = (facial_tension * 0.5 + eyebrow_position * 0.3 + (1 - mouth_opening) * 0.2)
            elif emotion == EmotionType.SURPRISE:
                intensity = (eye_opening * 0.5 + mouth_opening * 0.3 + eyebrow_position * 0.2)
            elif emotion == EmotionType.FOCUSED:
                intensity = (eye_opening * 0.4 + (1 - facial_tension) * 0.3 + eyebrow_position * 0.3)
            else:
                intensity = (facial_tension + eye_opening + mouth_opening + eyebrow_position) / 4
            
            return max(0, min(1, intensity))
            
        except Exception as e:
            logger.error(f"Error calculating emotion intensity: {str(e)}")
            return 0.5
    
    def _detect_micro_expressions(self, facial_data: Dict) -> List[Dict]:
        """
        Detect micro-expressions
        
        Args:
            facial_data: Facial analysis data
            
        Returns:
            List of micro-expression data
        """
        try:
            micro_expressions = []
            
            # Check for rapid facial changes (micro-expressions)
            facial_changes = facial_data.get('facial_changes', [])
            
            for change in facial_changes:
                if change.get('duration', 0) < 0.5:  # Micro-expression duration < 500ms
                    micro_expression = {
                        'type': change.get('type', 'unknown'),
                        'duration': change.get('duration', 0),
                        'intensity': change.get('intensity', 0),
                        'timestamp': change.get('timestamp', datetime.now().isoformat())
                    }
                    micro_expressions.append(micro_expression)
            
            return micro_expressions
            
        except Exception as e:
            logger.error(f"Error detecting micro-expressions: {str(e)}")
            return []
    
    def _calculate_engagement_level(self, facial_data: Dict, emotion: EmotionType, 
                                  intensity: float) -> Tuple[EngagementLevel, float]:
        """
        Calculate engagement level from facial data
        
        Args:
            facial_data: Facial analysis data
            emotion: Detected emotion type
            intensity: Emotion intensity
            
        Returns:
            Tuple of (engagement_level, engagement_score)
        """
        try:
            # Base engagement from emotion type
            emotion_engagement = {
                EmotionType.HAPPY: 0.8,
                EmotionType.FOCUSED: 0.9,
                EmotionType.SURPRISE: 0.7,
                EmotionType.NEUTRAL: 0.5,
                EmotionType.CONFUSED: 0.6,
                EmotionType.BORED: 0.2,
                EmotionType.SAD: 0.3,
                EmotionType.ANGRY: 0.4,
                EmotionType.FEAR: 0.3,
                EmotionType.DISGUST: 0.2
            }
            
            base_engagement = emotion_engagement.get(emotion, 0.5)
            
            # Adjust based on intensity
            intensity_factor = 0.5 + (intensity * 0.5)
            
            # Additional factors
            eye_contact = facial_data.get('eye_contact', 0.5)
            head_movement = facial_data.get('head_movement', 0.5)
            facial_activity = facial_data.get('facial_activity', 0.5)
            
            # Calculate final engagement score
            engagement_score = (
                base_engagement * 0.4 +
                intensity_factor * 0.3 +
                eye_contact * 0.2 +
                (1 - head_movement) * 0.05 +  # Less head movement = more focused
                facial_activity * 0.05
            )
            
            # Classify engagement level
            if engagement_score >= self.engagement_thresholds[EngagementLevel.VERY_HIGH]:
                engagement_level = EngagementLevel.VERY_HIGH
            elif engagement_score >= self.engagement_thresholds[EngagementLevel.HIGH]:
                engagement_level = EngagementLevel.HIGH
            elif engagement_score >= self.engagement_thresholds[EngagementLevel.MEDIUM]:
                engagement_level = EngagementLevel.MEDIUM
            elif engagement_score >= self.engagement_thresholds[EngagementLevel.LOW]:
                engagement_level = EngagementLevel.LOW
            else:
                engagement_level = EngagementLevel.VERY_LOW
            
            return engagement_level, engagement_score
            
        except Exception as e:
            logger.error(f"Error calculating engagement level: {str(e)}")
            return EngagementLevel.MEDIUM, 0.5
    
    def _determine_attention_state(self, facial_data: Dict, emotion: EmotionType, 
                                 engagement_score: float) -> Tuple[AttentionState, float]:
        """
        Determine attention state from facial data
        
        Args:
            facial_data: Facial analysis data
            emotion: Detected emotion type
            engagement_score: Engagement score
            
        Returns:
            Tuple of (attention_state, attention_score)
        """
        try:
            # Base attention from emotion and engagement
            if emotion == EmotionType.FOCUSED and engagement_score > 0.7:
                attention_state = AttentionState.FOCUSED
                attention_score = engagement_score
            elif emotion == EmotionType.CONFUSED:
                attention_state = AttentionState.CONFUSED
                attention_score = 0.6
            elif emotion == EmotionType.BORED or engagement_score < 0.3:
                attention_state = AttentionState.BORED
                attention_score = 0.2
            elif emotion == EmotionType.HAPPY and engagement_score > 0.8:
                attention_state = AttentionState.EXCITED
                attention_score = engagement_score
            elif emotion == EmotionType.ANGRY:
                attention_state = AttentionState.FRUSTRATED
                attention_score = 0.4
            else:
                attention_state = AttentionState.DISTRACTED
                attention_score = 0.5
            
            # Adjust based on additional factors
            eye_contact = facial_data.get('eye_contact', 0.5)
            head_stillness = 1 - facial_data.get('head_movement', 0.5)
            
            attention_score = (attention_score * 0.7 + eye_contact * 0.2 + head_stillness * 0.1)
            
            return attention_state, max(0, min(1, attention_score))
            
        except Exception as e:
            logger.error(f"Error determining attention state: {str(e)}")
            return AttentionState.DISTRACTED, 0.5
    
    def _calculate_confidence(self, facial_data: Dict, emotion_confidence: float) -> float:
        """
        Calculate overall confidence in analysis
        
        Args:
            facial_data: Facial analysis data
            emotion_confidence: Emotion detection confidence
            
        Returns:
            Overall confidence score (0-1)
        """
        try:
            # Factors affecting confidence
            data_quality = facial_data.get('data_quality', 0.5)
            landmark_quality = facial_data.get('landmark_quality', 0.5)
            lighting_quality = facial_data.get('lighting_quality', 0.5)
            
            # Calculate confidence
            confidence = (
                emotion_confidence * 0.4 +
                data_quality * 0.3 +
                landmark_quality * 0.2 +
                lighting_quality * 0.1
            )
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.1
    
    def _update_history(self, emotion_entry: Dict):
        """Update emotion history for trend analysis"""
        self.emotion_history.append(emotion_entry)
        
        # Keep only recent history
        if len(self.emotion_history) > 100:
            self.emotion_history = self.emotion_history[-100:]
    
    def _initialize_emotion_models(self) -> Dict:
        """Initialize emotion detection models"""
        # In production, this would load actual ML models
        return {
            'emotion_classifier': 'simulated_model',
            'landmark_detector': 'simulated_model',
            'micro_expression_detector': 'simulated_model'
        }
    
    def _get_default_metrics(self) -> EmotionMetrics:
        """Return default metrics when analysis fails"""
        return EmotionMetrics(
            primary_emotion=EmotionType.NEUTRAL,
            emotion_confidence=0.1,
            emotion_intensity=0.5,
            engagement_level=EngagementLevel.MEDIUM,
            attention_state=AttentionState.DISTRACTED,
            facial_landmarks=None,
            micro_expressions=[],
            emotion_timeline=[],
            engagement_score=0.5,
            attention_score=0.5,
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
                'average_engagement': 0.0,
                'emotion_stability': 0.0,
                'engagement_trend': 0.0
            }
        
        # Calculate statistics
        emotions = [entry['emotion'] for entry in self.emotion_history]
        engagements = [entry['engagement'] for entry in self.emotion_history]
        
        # Dominant emotion
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Average engagement
        average_engagement = np.mean(engagements)
        
        # Emotion stability (inverse of emotion variance)
        emotion_stability = 1.0 - np.var(engagements)
        
        # Engagement trend
        if len(engagements) > 1:
            engagement_trend = np.polyfit(range(len(engagements)), engagements, 1)[0]
        else:
            engagement_trend = 0.0
        
        return {
            'dominant_emotion': dominant_emotion,
            'average_engagement': float(average_engagement),
            'emotion_stability': float(emotion_stability),
            'engagement_trend': float(engagement_trend)
        }
