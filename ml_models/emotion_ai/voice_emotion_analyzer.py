"""
Voice Emotion Analysis Module

This module provides advanced voice emotion detection including:
- Real-time voice emotion recognition
- Sentiment analysis from speech
- Voice stress and fatigue detection
- Engagement level assessment from voice
- Multi-language emotion detection

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

class VoiceEmotionType(Enum):
    """Types of voice emotions detected"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    BORED = "bored"

class SentimentType(Enum):
    """Sentiment classifications"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class VoiceStressLevel(Enum):
    """Voice stress level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class VoiceFeatures:
    """Voice feature data"""
    pitch: float
    pitch_variance: float
    volume: float
    volume_variance: float
    speaking_rate: float
    pause_frequency: float
    jitter: float
    shimmer: float
    hnr: float  # Harmonics-to-noise ratio
    mfcc: List[float]  # Mel-frequency cepstral coefficients

@dataclass
class VoiceEmotionMetrics:
    """Comprehensive voice emotion analysis metrics"""
    primary_emotion: VoiceEmotionType
    emotion_confidence: float
    sentiment: SentimentType
    sentiment_score: float
    stress_level: VoiceStressLevel
    stress_score: float
    engagement_score: float
    fatigue_score: float
    voice_features: VoiceFeatures
    emotion_timeline: List[Dict]
    confidence: float

class VoiceEmotionAnalyzer:
    """
    Advanced voice emotion analysis system
    """
    
    def __init__(self, 
                 emotion_threshold: float = 0.6,
                 stress_threshold: float = 0.5,
                 sample_rate: int = 16000):
        """
        Initialize voice emotion analyzer
        
        Args:
            emotion_threshold: Threshold for emotion detection confidence
            stress_threshold: Threshold for stress level classification
            sample_rate: Audio sample rate in Hz
        """
        self.emotion_threshold = emotion_threshold
        self.stress_threshold = stress_threshold
        self.sample_rate = sample_rate
        
        # Voice emotion models (simplified - in production would use actual ML models)
        self.emotion_models = self._initialize_emotion_models()
        
        # Stress level thresholds
        self.stress_thresholds = {
            VoiceStressLevel.VERY_LOW: 0.2,
            VoiceStressLevel.LOW: 0.4,
            VoiceStressLevel.MEDIUM: 0.6,
            VoiceStressLevel.HIGH: 0.8,
            VoiceStressLevel.VERY_HIGH: 0.9
        }
        
        # Historical data for trend analysis
        self.emotion_history = []
        self.stress_history = []
        
        logger.info("Voice Emotion Analyzer initialized")
    
    def analyze_voice_emotion(self, audio_data: Dict) -> VoiceEmotionMetrics:
        """
        Analyze voice emotion from audio data
        
        Args:
            audio_data: Dictionary containing audio analysis data
            
        Returns:
            VoiceEmotionMetrics object with comprehensive analysis
        """
        try:
            # Extract voice features
            voice_features = self._extract_voice_features(audio_data)
            
            # Detect primary emotion
            primary_emotion, emotion_confidence = self._detect_voice_emotion(voice_features)
            
            # Analyze sentiment
            sentiment, sentiment_score = self._analyze_sentiment(audio_data, voice_features)
            
            # Assess stress level
            stress_level, stress_score = self._assess_stress_level(voice_features)
            
            # Calculate engagement score
            engagement_score = self._calculate_voice_engagement(voice_features, primary_emotion)
            
            # Calculate fatigue score
            fatigue_score = self._calculate_voice_fatigue(voice_features)
            
            # Create emotion timeline entry
            emotion_timeline_entry = {
                'timestamp': datetime.now().isoformat(),
                'emotion': primary_emotion.value,
                'confidence': emotion_confidence,
                'sentiment': sentiment.value,
                'sentiment_score': sentiment_score,
                'stress_score': stress_score,
                'engagement_score': engagement_score,
                'fatigue_score': fatigue_score
            }
            
            # Update history
            self._update_history(emotion_timeline_entry)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(audio_data, emotion_confidence)
            
            return VoiceEmotionMetrics(
                primary_emotion=primary_emotion,
                emotion_confidence=emotion_confidence,
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                stress_level=stress_level,
                stress_score=stress_score,
                engagement_score=engagement_score,
                fatigue_score=fatigue_score,
                voice_features=voice_features,
                emotion_timeline=self.emotion_history[-10:],  # Last 10 entries
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing voice emotion: {str(e)}")
            return self._get_default_metrics()
    
    def _extract_voice_features(self, audio_data: Dict) -> VoiceFeatures:
        """
        Extract voice features from audio data
        
        Args:
            audio_data: Audio analysis data
            
        Returns:
            VoiceFeatures object
        """
        try:
            # Extract or generate voice features
            pitch = audio_data.get('pitch', 150.0)  # Hz
            pitch_variance = audio_data.get('pitch_variance', 20.0)
            volume = audio_data.get('volume', 0.5)  # Normalized
            volume_variance = audio_data.get('volume_variance', 0.1)
            speaking_rate = audio_data.get('speaking_rate', 2.0)  # Words per second
            pause_frequency = audio_data.get('pause_frequency', 0.1)  # Pauses per second
            jitter = audio_data.get('jitter', 0.02)  # Pitch period variation
            shimmer = audio_data.get('shimmer', 0.05)  # Amplitude variation
            hnr = audio_data.get('hnr', 20.0)  # Harmonics-to-noise ratio
            mfcc = audio_data.get('mfcc', [0.0] * 13)  # 13 MFCC coefficients
            
            return VoiceFeatures(
                pitch=pitch,
                pitch_variance=pitch_variance,
                volume=volume,
                volume_variance=volume_variance,
                speaking_rate=speaking_rate,
                pause_frequency=pause_frequency,
                jitter=jitter,
                shimmer=shimmer,
                hnr=hnr,
                mfcc=mfcc
            )
            
        except Exception as e:
            logger.error(f"Error extracting voice features: {str(e)}")
            return VoiceFeatures(
                pitch=150.0, pitch_variance=20.0, volume=0.5, volume_variance=0.1,
                speaking_rate=2.0, pause_frequency=0.1, jitter=0.02, shimmer=0.05,
                hnr=20.0, mfcc=[0.0] * 13
            )
    
    def _detect_voice_emotion(self, voice_features: VoiceFeatures) -> Tuple[VoiceEmotionType, float]:
        """
        Detect voice emotion from voice features
        
        Args:
            voice_features: Extracted voice features
            
        Returns:
            Tuple of (emotion_type, confidence)
        """
        try:
            # Generate emotion scores based on voice features
            emotion_scores = self._calculate_emotion_scores(voice_features)
            
            # Find emotion with highest score
            max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            emotion_type = VoiceEmotionType(max_emotion[0])
            confidence = max_emotion[1]
            
            # Apply threshold
            if confidence < self.emotion_threshold:
                emotion_type = VoiceEmotionType.NEUTRAL
                confidence = 0.5
            
            return emotion_type, confidence
            
        except Exception as e:
            logger.error(f"Error detecting voice emotion: {str(e)}")
            return VoiceEmotionType.NEUTRAL, 0.1
    
    def _calculate_emotion_scores(self, voice_features: VoiceFeatures) -> Dict[str, float]:
        """
        Calculate emotion scores from voice features
        
        Args:
            voice_features: Voice features
            
        Returns:
            Dictionary of emotion scores
        """
        try:
            # Normalize features
            pitch_norm = min(1.0, voice_features.pitch / 300.0)
            volume_norm = voice_features.volume
            rate_norm = min(1.0, voice_features.speaking_rate / 4.0)
            jitter_norm = min(1.0, voice_features.jitter * 50)
            shimmer_norm = min(1.0, voice_features.shimmer * 20)
            
            # Calculate emotion scores based on voice characteristics
            emotion_scores = {}
            
            # Happy: Higher pitch, faster rate, lower jitter
            emotion_scores['happy'] = (
                pitch_norm * 0.3 +
                rate_norm * 0.3 +
                (1 - jitter_norm) * 0.2 +
                volume_norm * 0.2
            )
            
            # Sad: Lower pitch, slower rate, higher jitter
            emotion_scores['sad'] = (
                (1 - pitch_norm) * 0.3 +
                (1 - rate_norm) * 0.3 +
                jitter_norm * 0.2 +
                (1 - volume_norm) * 0.2
            )
            
            # Angry: Higher pitch, faster rate, higher jitter, higher volume
            emotion_scores['angry'] = (
                pitch_norm * 0.25 +
                rate_norm * 0.25 +
                jitter_norm * 0.25 +
                volume_norm * 0.25
            )
            
            # Fear: Higher pitch, variable rate, higher jitter
            emotion_scores['fear'] = (
                pitch_norm * 0.3 +
                voice_features.pitch_variance * 0.3 +
                jitter_norm * 0.2 +
                (1 - volume_norm) * 0.2
            )
            
            # Surprise: Higher pitch, faster rate, lower jitter
            emotion_scores['surprise'] = (
                pitch_norm * 0.4 +
                rate_norm * 0.3 +
                (1 - jitter_norm) * 0.3
            )
            
            # Excited: Higher pitch, faster rate, higher volume
            emotion_scores['excited'] = (
                pitch_norm * 0.3 +
                rate_norm * 0.3 +
                volume_norm * 0.4
            )
            
            # Frustrated: Medium pitch, variable rate, higher jitter
            emotion_scores['frustrated'] = (
                0.5 * 0.3 +
                voice_features.pitch_variance * 0.3 +
                jitter_norm * 0.4
            )
            
            # Confused: Variable pitch, slower rate, higher jitter
            emotion_scores['confused'] = (
                voice_features.pitch_variance * 0.4 +
                (1 - rate_norm) * 0.3 +
                jitter_norm * 0.3
            )
            
            # Bored: Lower pitch, slower rate, lower volume
            emotion_scores['bored'] = (
                (1 - pitch_norm) * 0.4 +
                (1 - rate_norm) * 0.4 +
                (1 - volume_norm) * 0.2
            )
            
            # Neutral: Medium values, low variance
            emotion_scores['neutral'] = (
                (1 - abs(pitch_norm - 0.5)) * 0.3 +
                (1 - abs(rate_norm - 0.5)) * 0.3 +
                (1 - jitter_norm) * 0.2 +
                (1 - shimmer_norm) * 0.2
            )
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                for emotion in emotion_scores:
                    emotion_scores[emotion] = emotion_scores[emotion] / total_score
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Error calculating emotion scores: {str(e)}")
            return {'neutral': 1.0}
    
    def _analyze_sentiment(self, audio_data: Dict, voice_features: VoiceFeatures) -> Tuple[SentimentType, float]:
        """
        Analyze sentiment from voice data
        
        Args:
            audio_data: Audio analysis data
            voice_features: Voice features
            
        Returns:
            Tuple of (sentiment_type, sentiment_score)
        """
        try:
            # Extract text if available
            text = audio_data.get('text', '')
            
            # Calculate sentiment from voice features
            voice_sentiment = self._calculate_voice_sentiment(voice_features)
            
            # Calculate sentiment from text if available
            text_sentiment = 0.5  # Neutral by default
            if text:
                text_sentiment = self._analyze_text_sentiment(text)
            
            # Combine voice and text sentiment
            combined_sentiment = (voice_sentiment * 0.7 + text_sentiment * 0.3)
            
            # Classify sentiment
            if combined_sentiment >= 0.8:
                sentiment = SentimentType.VERY_POSITIVE
            elif combined_sentiment >= 0.6:
                sentiment = SentimentType.POSITIVE
            elif combined_sentiment >= 0.4:
                sentiment = SentimentType.NEUTRAL
            elif combined_sentiment >= 0.2:
                sentiment = SentimentType.NEGATIVE
            else:
                sentiment = SentimentType.VERY_NEGATIVE
            
            return sentiment, combined_sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return SentimentType.NEUTRAL, 0.5
    
    def _calculate_voice_sentiment(self, voice_features: VoiceFeatures) -> float:
        """Calculate sentiment from voice features"""
        try:
            # Positive sentiment indicators
            positive_indicators = (
                voice_features.pitch / 300.0 * 0.3 +  # Higher pitch
                voice_features.volume * 0.3 +  # Higher volume
                voice_features.speaking_rate / 4.0 * 0.2 +  # Faster rate
                (1 - voice_features.jitter) * 0.2  # Lower jitter
            )
            
            return max(0, min(1, positive_indicators))
            
        except Exception as e:
            logger.error(f"Error calculating voice sentiment: {str(e)}")
            return 0.5
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment from text (simplified implementation)"""
        try:
            # Simple keyword-based sentiment analysis
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'worst', 'angry']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count + negative_count == 0:
                return 0.5  # Neutral
            
            sentiment = positive_count / (positive_count + negative_count)
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {str(e)}")
            return 0.5
    
    def _assess_stress_level(self, voice_features: VoiceFeatures) -> Tuple[VoiceStressLevel, float]:
        """
        Assess stress level from voice features
        
        Args:
            voice_features: Voice features
            
        Returns:
            Tuple of (stress_level, stress_score)
        """
        try:
            # Stress indicators
            stress_indicators = (
                voice_features.jitter * 0.3 +  # Higher jitter
                voice_features.shimmer * 0.3 +  # Higher shimmer
                voice_features.pitch_variance * 0.2 +  # Higher pitch variance
                (1 - voice_features.hnr) * 0.2  # Lower harmonics-to-noise ratio
            )
            
            stress_score = max(0, min(1, stress_indicators))
            
            # Classify stress level
            if stress_score >= self.stress_thresholds[VoiceStressLevel.VERY_HIGH]:
                stress_level = VoiceStressLevel.VERY_HIGH
            elif stress_score >= self.stress_thresholds[VoiceStressLevel.HIGH]:
                stress_level = VoiceStressLevel.HIGH
            elif stress_score >= self.stress_thresholds[VoiceStressLevel.MEDIUM]:
                stress_level = VoiceStressLevel.MEDIUM
            elif stress_score >= self.stress_thresholds[VoiceStressLevel.LOW]:
                stress_level = VoiceStressLevel.LOW
            else:
                stress_level = VoiceStressLevel.VERY_LOW
            
            return stress_level, stress_score
            
        except Exception as e:
            logger.error(f"Error assessing stress level: {str(e)}")
            return VoiceStressLevel.MEDIUM, 0.5
    
    def _calculate_voice_engagement(self, voice_features: VoiceFeatures, 
                                  emotion: VoiceEmotionType) -> float:
        """
        Calculate engagement score from voice features
        
        Args:
            voice_features: Voice features
            emotion: Detected emotion type
            
        Returns:
            Engagement score (0-1)
        """
        try:
            # Base engagement from emotion type
            emotion_engagement = {
                VoiceEmotionType.EXCITED: 0.9,
                VoiceEmotionType.HAPPY: 0.8,
                VoiceEmotionType.SURPRISE: 0.7,
                VoiceEmotionType.NEUTRAL: 0.5,
                VoiceEmotionType.CONFUSED: 0.6,
                VoiceEmotionType.BORED: 0.2,
                VoiceEmotionType.SAD: 0.3,
                VoiceEmotionType.ANGRY: 0.4,
                VoiceEmotionType.FEAR: 0.3,
                VoiceEmotionType.FRUSTRATED: 0.4,
                VoiceEmotionType.DISGUST: 0.2
            }
            
            base_engagement = emotion_engagement.get(emotion, 0.5)
            
            # Adjust based on voice characteristics
            voice_engagement = (
                voice_features.volume * 0.3 +  # Higher volume = more engaged
                voice_features.speaking_rate / 4.0 * 0.3 +  # Faster rate = more engaged
                (1 - voice_features.pause_frequency) * 0.2 +  # Fewer pauses = more engaged
                (1 - voice_features.jitter) * 0.2  # Lower jitter = more engaged
            )
            
            # Combined engagement score
            engagement_score = (base_engagement * 0.6 + voice_engagement * 0.4)
            
            return max(0, min(1, engagement_score))
            
        except Exception as e:
            logger.error(f"Error calculating voice engagement: {str(e)}")
            return 0.5
    
    def _calculate_voice_fatigue(self, voice_features: VoiceFeatures) -> float:
        """
        Calculate fatigue score from voice features
        
        Args:
            voice_features: Voice features
            
        Returns:
            Fatigue score (0-1)
        """
        try:
            # Fatigue indicators
            fatigue_indicators = (
                (1 - voice_features.volume) * 0.3 +  # Lower volume
                (1 - voice_features.speaking_rate / 4.0) * 0.3 +  # Slower rate
                voice_features.pause_frequency * 0.2 +  # More pauses
                voice_features.jitter * 0.2  # Higher jitter
            )
            
            fatigue_score = max(0, min(1, fatigue_indicators))
            
            return fatigue_score
            
        except Exception as e:
            logger.error(f"Error calculating voice fatigue: {str(e)}")
            return 0.5
    
    def _calculate_confidence(self, audio_data: Dict, emotion_confidence: float) -> float:
        """
        Calculate overall confidence in analysis
        
        Args:
            audio_data: Audio analysis data
            emotion_confidence: Emotion detection confidence
            
        Returns:
            Overall confidence score (0-1)
        """
        try:
            # Factors affecting confidence
            audio_quality = audio_data.get('audio_quality', 0.5)
            signal_strength = audio_data.get('signal_strength', 0.5)
            noise_level = audio_data.get('noise_level', 0.5)
            
            # Calculate confidence
            confidence = (
                emotion_confidence * 0.4 +
                audio_quality * 0.3 +
                signal_strength * 0.2 +
                (1 - noise_level) * 0.1
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
        """Initialize voice emotion detection models"""
        # In production, this would load actual ML models
        return {
            'emotion_classifier': 'simulated_model',
            'sentiment_analyzer': 'simulated_model',
            'stress_detector': 'simulated_model'
        }
    
    def _get_default_metrics(self) -> VoiceEmotionMetrics:
        """Return default metrics when analysis fails"""
        return VoiceEmotionMetrics(
            primary_emotion=VoiceEmotionType.NEUTRAL,
            emotion_confidence=0.1,
            sentiment=SentimentType.NEUTRAL,
            sentiment_score=0.5,
            stress_level=VoiceStressLevel.MEDIUM,
            stress_score=0.5,
            engagement_score=0.5,
            fatigue_score=0.5,
            voice_features=VoiceFeatures(
                pitch=150.0, pitch_variance=20.0, volume=0.5, volume_variance=0.1,
                speaking_rate=2.0, pause_frequency=0.1, jitter=0.02, shimmer=0.05,
                hnr=20.0, mfcc=[0.0] * 13
            ),
            emotion_timeline=[],
            confidence=0.1
        )
    
    def get_voice_statistics(self) -> Dict[str, float]:
        """
        Get voice emotion statistics from historical data
        
        Returns:
            Dictionary with voice statistics
        """
        if not self.emotion_history:
            return {
                'dominant_emotion': 'neutral',
                'average_engagement': 0.0,
                'average_stress': 0.0,
                'sentiment_trend': 0.0
            }
        
        # Calculate statistics
        emotions = [entry['emotion'] for entry in self.emotion_history]
        engagements = [entry['engagement_score'] for entry in self.emotion_history]
        stresses = [entry['stress_score'] for entry in self.emotion_history]
        sentiments = [entry['sentiment_score'] for entry in self.emotion_history]
        
        # Dominant emotion
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Average metrics
        average_engagement = np.mean(engagements)
        average_stress = np.mean(stresses)
        
        # Sentiment trend
        if len(sentiments) > 1:
            sentiment_trend = np.polyfit(range(len(sentiments)), sentiments, 1)[0]
        else:
            sentiment_trend = 0.0
        
        return {
            'dominant_emotion': dominant_emotion,
            'average_engagement': float(average_engagement),
            'average_stress': float(average_stress),
            'sentiment_trend': float(sentiment_trend)
        }
