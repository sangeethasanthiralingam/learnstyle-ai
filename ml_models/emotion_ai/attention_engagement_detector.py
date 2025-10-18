"""
Attention Engagement Detection Module

This module provides comprehensive attention and engagement detection including:
- Multi-modal attention state analysis
- Engagement level assessment
- Attention span measurement
- Distraction detection
- Learning focus optimization

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

class AttentionState(Enum):
    """Attention state classifications"""
    HIGHLY_FOCUSED = "highly_focused"
    FOCUSED = "focused"
    MODERATELY_ATTENTIVE = "moderately_attentive"
    DISTRACTED = "distracted"
    HIGHLY_DISTRACTED = "highly_distracted"
    CONFUSED = "confused"
    OVERWHELMED = "overwhelmed"

class EngagementLevel(Enum):
    """Engagement level classifications"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

class DistractionType(Enum):
    """Types of distractions detected"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    COGNITIVE = "cognitive"
    EMOTIONAL = "emotional"
    PHYSICAL = "physical"
    ENVIRONMENTAL = "environmental"

@dataclass
class AttentionMetrics:
    """Comprehensive attention analysis metrics"""
    attention_state: AttentionState
    attention_score: float
    engagement_level: EngagementLevel
    engagement_score: float
    attention_span: float
    distraction_level: float
    distraction_types: List[DistractionType]
    focus_quality: float
    learning_readiness: float
    attention_timeline: List[Dict]
    confidence: float

class AttentionEngagementDetector:
    """
    Advanced attention and engagement detection system
    """
    
    def __init__(self, 
                 attention_threshold: float = 0.6,
                 engagement_threshold: float = 0.5,
                 distraction_threshold: float = 0.4):
        """
        Initialize attention engagement detector
        
        Args:
            attention_threshold: Threshold for attention state classification
            engagement_threshold: Threshold for engagement level classification
            distraction_threshold: Threshold for distraction detection
        """
        self.attention_threshold = attention_threshold
        self.engagement_threshold = engagement_threshold
        self.distraction_threshold = distraction_threshold
        
        # Attention level thresholds
        self.attention_thresholds = {
            AttentionState.HIGHLY_FOCUSED: 0.9,
            AttentionState.FOCUSED: 0.7,
            AttentionState.MODERATELY_ATTENTIVE: 0.5,
            AttentionState.DISTRACTED: 0.3,
            AttentionState.HIGHLY_DISTRACTED: 0.1
        }
        
        # Engagement level thresholds
        self.engagement_thresholds = {
            EngagementLevel.VERY_HIGH: 0.9,
            EngagementLevel.HIGH: 0.7,
            EngagementLevel.MEDIUM: 0.5,
            EngagementLevel.LOW: 0.3,
            EngagementLevel.VERY_LOW: 0.1
        }
        
        # Historical data for trend analysis
        self.attention_history = []
        self.engagement_history = []
        
        logger.info("Attention Engagement Detector initialized")
    
    def detect_attention_engagement(self, multi_modal_data: Dict) -> AttentionMetrics:
        """
        Detect attention and engagement from multi-modal data
        
        Args:
            multi_modal_data: Dictionary containing various sensor data
            
        Returns:
            AttentionMetrics object with comprehensive analysis
        """
        try:
            # Extract data from different modalities
            facial_data = multi_modal_data.get('facial_data', {})
            voice_data = multi_modal_data.get('voice_data', {})
            eye_tracking_data = multi_modal_data.get('eye_tracking_data', {})
            neurofeedback_data = multi_modal_data.get('neurofeedback_data', {})
            behavioral_data = multi_modal_data.get('behavioral_data', {})
            
            # Calculate attention score
            attention_score = self._calculate_attention_score(
                facial_data, voice_data, eye_tracking_data, neurofeedback_data, behavioral_data
            )
            
            # Calculate engagement score
            engagement_score = self._calculate_engagement_score(
                facial_data, voice_data, eye_tracking_data, neurofeedback_data, behavioral_data
            )
            
            # Determine attention state
            attention_state = self._classify_attention_state(attention_score)
            
            # Determine engagement level
            engagement_level = self._classify_engagement_level(engagement_score)
            
            # Calculate attention span
            attention_span = self._calculate_attention_span(attention_score)
            
            # Detect distractions
            distraction_level, distraction_types = self._detect_distractions(
                facial_data, voice_data, eye_tracking_data, neurofeedback_data, behavioral_data
            )
            
            # Calculate focus quality
            focus_quality = self._calculate_focus_quality(attention_score, engagement_score, distraction_level)
            
            # Calculate learning readiness
            learning_readiness = self._calculate_learning_readiness(
                attention_score, engagement_score, distraction_level, focus_quality
            )
            
            # Create attention timeline entry
            attention_entry = {
                'timestamp': datetime.now().isoformat(),
                'attention_score': attention_score,
                'engagement_score': engagement_score,
                'attention_state': attention_state.value,
                'engagement_level': engagement_level.value,
                'distraction_level': distraction_level,
                'focus_quality': focus_quality,
                'learning_readiness': learning_readiness
            }
            
            # Update history
            self._update_history(attention_entry)
            
            # Calculate confidence
            confidence = self._calculate_confidence(multi_modal_data, attention_score, engagement_score)
            
            return AttentionMetrics(
                attention_state=attention_state,
                attention_score=attention_score,
                engagement_level=engagement_level,
                engagement_score=engagement_score,
                attention_span=attention_span,
                distraction_level=distraction_level,
                distraction_types=distraction_types,
                focus_quality=focus_quality,
                learning_readiness=learning_readiness,
                attention_timeline=self.attention_history[-20:],  # Last 20 entries
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error detecting attention engagement: {str(e)}")
            return self._get_default_metrics()
    
    def _calculate_attention_score(self, facial_data: Dict, voice_data: Dict, 
                                 eye_tracking_data: Dict, neurofeedback_data: Dict, 
                                 behavioral_data: Dict) -> float:
        """
        Calculate overall attention score from multi-modal data
        
        Args:
            facial_data: Facial analysis data
            voice_data: Voice analysis data
            eye_tracking_data: Eye-tracking data
            neurofeedback_data: Neurofeedback data
            behavioral_data: Behavioral data
            
        Returns:
            Attention score (0-1)
        """
        try:
            attention_scores = []
            weights = []
            
            # Facial attention indicators
            if facial_data:
                facial_attention = self._extract_facial_attention(facial_data)
                attention_scores.append(facial_attention)
                weights.append(0.25)
            
            # Voice attention indicators
            if voice_data:
                voice_attention = self._extract_voice_attention(voice_data)
                attention_scores.append(voice_attention)
                weights.append(0.2)
            
            # Eye-tracking attention indicators
            if eye_tracking_data:
                eye_attention = self._extract_eye_attention(eye_tracking_data)
                attention_scores.append(eye_attention)
                weights.append(0.3)
            
            # Neurofeedback attention indicators
            if neurofeedback_data:
                neuro_attention = self._extract_neuro_attention(neurofeedback_data)
                attention_scores.append(neuro_attention)
                weights.append(0.15)
            
            # Behavioral attention indicators
            if behavioral_data:
                behavioral_attention = self._extract_behavioral_attention(behavioral_data)
                attention_scores.append(behavioral_attention)
                weights.append(0.1)
            
            # Calculate weighted average
            if attention_scores and weights:
                total_weight = sum(weights)
                weighted_score = sum(score * weight for score, weight in zip(attention_scores, weights))
                attention_score = weighted_score / total_weight
            else:
                attention_score = 0.5  # Default neutral score
            
            return max(0, min(1, attention_score))
            
        except Exception as e:
            logger.error(f"Error calculating attention score: {str(e)}")
            return 0.5
    
    def _extract_facial_attention(self, facial_data: Dict) -> float:
        """Extract attention indicators from facial data"""
        try:
            # Eye contact and focus indicators
            eye_contact = facial_data.get('eye_contact', 0.5)
            head_stillness = 1 - facial_data.get('head_movement', 0.5)
            facial_activity = facial_data.get('facial_activity', 0.5)
            
            # Attention from facial features
            attention = (eye_contact * 0.5 + head_stillness * 0.3 + facial_activity * 0.2)
            
            return max(0, min(1, attention))
            
        except Exception as e:
            logger.error(f"Error extracting facial attention: {str(e)}")
            return 0.5
    
    def _extract_voice_attention(self, voice_data: Dict) -> float:
        """Extract attention indicators from voice data"""
        try:
            # Voice characteristics indicating attention
            speaking_rate = voice_data.get('speaking_rate', 2.0)
            pause_frequency = voice_data.get('pause_frequency', 0.1)
            volume_consistency = 1 - voice_data.get('volume_variance', 0.1)
            
            # Normalize speaking rate
            rate_score = min(1.0, speaking_rate / 3.0)  # Optimal around 2-3 words per second
            
            # Attention from voice features
            attention = (rate_score * 0.4 + (1 - pause_frequency) * 0.3 + volume_consistency * 0.3)
            
            return max(0, min(1, attention))
            
        except Exception as e:
            logger.error(f"Error extracting voice attention: {str(e)}")
            return 0.5
    
    def _extract_eye_attention(self, eye_tracking_data: Dict) -> float:
        """Extract attention indicators from eye-tracking data"""
        try:
            # Eye-tracking attention indicators
            fixation_duration = eye_tracking_data.get('average_fixation_duration', 200)
            saccade_frequency = eye_tracking_data.get('saccade_count', 0)
            scanpath_efficiency = eye_tracking_data.get('scanpath_efficiency', 0.5)
            
            # Normalize fixation duration (optimal around 200-400ms)
            fixation_score = 1.0 - abs(fixation_duration - 300) / 300.0
            fixation_score = max(0, min(1, fixation_score))
            
            # Normalize saccade frequency (moderate frequency is good)
            saccade_score = 1.0 - abs(saccade_frequency - 10) / 20.0
            saccade_score = max(0, min(1, saccade_score))
            
            # Attention from eye-tracking features
            attention = (fixation_score * 0.4 + saccade_score * 0.3 + scanpath_efficiency * 0.3)
            
            return max(0, min(1, attention))
            
        except Exception as e:
            logger.error(f"Error extracting eye attention: {str(e)}")
            return 0.5
    
    def _extract_neuro_attention(self, neurofeedback_data: Dict) -> float:
        """Extract attention indicators from neurofeedback data"""
        try:
            # Neurofeedback attention indicators
            focus_level = neurofeedback_data.get('focus_level', 0.5)
            alpha_beta_ratio = neurofeedback_data.get('alpha_beta_ratio', 1.0)
            cognitive_load = neurofeedback_data.get('cognitive_load', 0.5)
            
            # Optimal alpha/beta ratio for attention (around 0.8-1.2)
            ratio_score = 1.0 - abs(alpha_beta_ratio - 1.0) / 1.0
            ratio_score = max(0, min(1, ratio_score))
            
            # Moderate cognitive load is optimal for attention
            load_score = 1.0 - abs(cognitive_load - 0.6) / 0.6
            load_score = max(0, min(1, load_score))
            
            # Attention from neurofeedback features
            attention = (focus_level * 0.5 + ratio_score * 0.3 + load_score * 0.2)
            
            return max(0, min(1, attention))
            
        except Exception as e:
            logger.error(f"Error extracting neuro attention: {str(e)}")
            return 0.5
    
    def _extract_behavioral_attention(self, behavioral_data: Dict) -> float:
        """Extract attention indicators from behavioral data"""
        try:
            # Behavioral attention indicators
            mouse_movement = behavioral_data.get('mouse_movement', 0.5)
            click_frequency = behavioral_data.get('click_frequency', 0.5)
            scroll_behavior = behavioral_data.get('scroll_behavior', 0.5)
            keyboard_activity = behavioral_data.get('keyboard_activity', 0.5)
            
            # Moderate activity indicates attention
            activity_score = (
                (1 - abs(mouse_movement - 0.5)) * 0.3 +
                (1 - abs(click_frequency - 0.5)) * 0.3 +
                (1 - abs(scroll_behavior - 0.5)) * 0.2 +
                (1 - abs(keyboard_activity - 0.5)) * 0.2
            )
            
            return max(0, min(1, activity_score))
            
        except Exception as e:
            logger.error(f"Error extracting behavioral attention: {str(e)}")
            return 0.5
    
    def _calculate_engagement_score(self, facial_data: Dict, voice_data: Dict, 
                                  eye_tracking_data: Dict, neurofeedback_data: Dict, 
                                  behavioral_data: Dict) -> float:
        """
        Calculate overall engagement score from multi-modal data
        
        Args:
            facial_data: Facial analysis data
            voice_data: Voice analysis data
            eye_tracking_data: Eye-tracking data
            neurofeedback_data: Neurofeedback data
            behavioral_data: Behavioral data
            
        Returns:
            Engagement score (0-1)
        """
        try:
            engagement_scores = []
            weights = []
            
            # Facial engagement indicators
            if facial_data:
                facial_engagement = self._extract_facial_engagement(facial_data)
                engagement_scores.append(facial_engagement)
                weights.append(0.3)
            
            # Voice engagement indicators
            if voice_data:
                voice_engagement = self._extract_voice_engagement(voice_data)
                engagement_scores.append(voice_engagement)
                weights.append(0.25)
            
            # Eye-tracking engagement indicators
            if eye_tracking_data:
                eye_engagement = self._extract_eye_engagement(eye_tracking_data)
                engagement_scores.append(eye_engagement)
                weights.append(0.25)
            
            # Neurofeedback engagement indicators
            if neurofeedback_data:
                neuro_engagement = self._extract_neuro_engagement(neurofeedback_data)
                engagement_scores.append(neuro_engagement)
                weights.append(0.1)
            
            # Behavioral engagement indicators
            if behavioral_data:
                behavioral_engagement = self._extract_behavioral_engagement(behavioral_data)
                engagement_scores.append(behavioral_engagement)
                weights.append(0.1)
            
            # Calculate weighted average
            if engagement_scores and weights:
                total_weight = sum(weights)
                weighted_score = sum(score * weight for score, weight in zip(engagement_scores, weights))
                engagement_score = weighted_score / total_weight
            else:
                engagement_score = 0.5  # Default neutral score
            
            return max(0, min(1, engagement_score))
            
        except Exception as e:
            logger.error(f"Error calculating engagement score: {str(e)}")
            return 0.5
    
    def _extract_facial_engagement(self, facial_data: Dict) -> float:
        """Extract engagement indicators from facial data"""
        try:
            # Facial engagement indicators
            emotion_engagement = facial_data.get('emotion_engagement', 0.5)
            facial_activity = facial_data.get('facial_activity', 0.5)
            eye_contact = facial_data.get('eye_contact', 0.5)
            
            engagement = (emotion_engagement * 0.5 + facial_activity * 0.3 + eye_contact * 0.2)
            
            return max(0, min(1, engagement))
            
        except Exception as e:
            logger.error(f"Error extracting facial engagement: {str(e)}")
            return 0.5
    
    def _extract_voice_engagement(self, voice_data: Dict) -> float:
        """Extract engagement indicators from voice data"""
        try:
            # Voice engagement indicators
            volume = voice_data.get('volume', 0.5)
            speaking_rate = voice_data.get('speaking_rate', 2.0)
            pitch_variance = voice_data.get('pitch_variance', 20.0)
            
            # Normalize features
            volume_score = volume
            rate_score = min(1.0, speaking_rate / 3.0)
            variance_score = min(1.0, pitch_variance / 50.0)
            
            engagement = (volume_score * 0.4 + rate_score * 0.3 + variance_score * 0.3)
            
            return max(0, min(1, engagement))
            
        except Exception as e:
            logger.error(f"Error extracting voice engagement: {str(e)}")
            return 0.5
    
    def _extract_eye_engagement(self, eye_tracking_data: Dict) -> float:
        """Extract engagement indicators from eye-tracking data"""
        try:
            # Eye-tracking engagement indicators
            engagement_score = eye_tracking_data.get('engagement_score', 0.5)
            fixation_count = eye_tracking_data.get('fixation_count', 0)
            visual_search_efficiency = eye_tracking_data.get('visual_search_efficiency', 0.5)
            
            # Normalize fixation count
            fixation_score = min(1.0, fixation_count / 50.0)
            
            engagement = (engagement_score * 0.5 + fixation_score * 0.3 + visual_search_efficiency * 0.2)
            
            return max(0, min(1, engagement))
            
        except Exception as e:
            logger.error(f"Error extracting eye engagement: {str(e)}")
            return 0.5
    
    def _extract_neuro_engagement(self, neurofeedback_data: Dict) -> float:
        """Extract engagement indicators from neurofeedback data"""
        try:
            # Neurofeedback engagement indicators
            focus_level = neurofeedback_data.get('focus_level', 0.5)
            cognitive_load = neurofeedback_data.get('cognitive_load', 0.5)
            
            # Moderate cognitive load indicates good engagement
            load_score = 1.0 - abs(cognitive_load - 0.6) / 0.6
            load_score = max(0, min(1, load_score))
            
            engagement = (focus_level * 0.6 + load_score * 0.4)
            
            return max(0, min(1, engagement))
            
        except Exception as e:
            logger.error(f"Error extracting neuro engagement: {str(e)}")
            return 0.5
    
    def _extract_behavioral_engagement(self, behavioral_data: Dict) -> float:
        """Extract engagement indicators from behavioral data"""
        try:
            # Behavioral engagement indicators
            interaction_frequency = behavioral_data.get('interaction_frequency', 0.5)
            response_time = behavioral_data.get('response_time', 1.0)
            task_completion = behavioral_data.get('task_completion', 0.5)
            
            # Normalize response time (faster is better)
            response_score = max(0, 1.0 - response_time)
            
            engagement = (interaction_frequency * 0.4 + response_score * 0.3 + task_completion * 0.3)
            
            return max(0, min(1, engagement))
            
        except Exception as e:
            logger.error(f"Error extracting behavioral engagement: {str(e)}")
            return 0.5
    
    def _classify_attention_state(self, attention_score: float) -> AttentionState:
        """Classify attention state from attention score"""
        if attention_score >= self.attention_thresholds[AttentionState.HIGHLY_FOCUSED]:
            return AttentionState.HIGHLY_FOCUSED
        elif attention_score >= self.attention_thresholds[AttentionState.FOCUSED]:
            return AttentionState.FOCUSED
        elif attention_score >= self.attention_thresholds[AttentionState.MODERATELY_ATTENTIVE]:
            return AttentionState.MODERATELY_ATTENTIVE
        elif attention_score >= self.attention_thresholds[AttentionState.DISTRACTED]:
            return AttentionState.DISTRACTED
        else:
            return AttentionState.HIGHLY_DISTRACTED
    
    def _classify_engagement_level(self, engagement_score: float) -> EngagementLevel:
        """Classify engagement level from engagement score"""
        if engagement_score >= self.engagement_thresholds[EngagementLevel.VERY_HIGH]:
            return EngagementLevel.VERY_HIGH
        elif engagement_score >= self.engagement_thresholds[EngagementLevel.HIGH]:
            return EngagementLevel.HIGH
        elif engagement_score >= self.engagement_thresholds[EngagementLevel.MEDIUM]:
            return EngagementLevel.MEDIUM
        elif engagement_score >= self.engagement_thresholds[EngagementLevel.LOW]:
            return EngagementLevel.LOW
        else:
            return EngagementLevel.VERY_LOW
    
    def _calculate_attention_span(self, attention_score: float) -> float:
        """Calculate attention span from attention score"""
        # Attention span is proportional to attention score
        # Convert to minutes (0-60 minutes)
        attention_span_minutes = attention_score * 60.0
        return attention_span_minutes
    
    def _detect_distractions(self, facial_data: Dict, voice_data: Dict, 
                           eye_tracking_data: Dict, neurofeedback_data: Dict, 
                           behavioral_data: Dict) -> Tuple[float, List[DistractionType]]:
        """
        Detect distractions from multi-modal data
        
        Args:
            facial_data: Facial analysis data
            voice_data: Voice analysis data
            eye_tracking_data: Eye-tracking data
            neurofeedback_data: Neurofeedback data
            behavioral_data: Behavioral data
            
        Returns:
            Tuple of (distraction_level, distraction_types)
        """
        try:
            distraction_types = []
            distraction_scores = []
            
            # Visual distractions
            if eye_tracking_data:
                visual_distraction = self._detect_visual_distractions(eye_tracking_data)
                if visual_distraction > self.distraction_threshold:
                    distraction_types.append(DistractionType.VISUAL)
                    distraction_scores.append(visual_distraction)
            
            # Auditory distractions
            if voice_data:
                auditory_distraction = self._detect_auditory_distractions(voice_data)
                if auditory_distraction > self.distraction_threshold:
                    distraction_types.append(DistractionType.AUDITORY)
                    distraction_scores.append(auditory_distraction)
            
            # Cognitive distractions
            if neurofeedback_data:
                cognitive_distraction = self._detect_cognitive_distractions(neurofeedback_data)
                if cognitive_distraction > self.distraction_threshold:
                    distraction_types.append(DistractionType.COGNITIVE)
                    distraction_scores.append(cognitive_distraction)
            
            # Emotional distractions
            if facial_data:
                emotional_distraction = self._detect_emotional_distractions(facial_data)
                if emotional_distraction > self.distraction_threshold:
                    distraction_types.append(DistractionType.EMOTIONAL)
                    distraction_scores.append(emotional_distraction)
            
            # Physical distractions
            if behavioral_data:
                physical_distraction = self._detect_physical_distractions(behavioral_data)
                if physical_distraction > self.distraction_threshold:
                    distraction_types.append(DistractionType.PHYSICAL)
                    distraction_scores.append(physical_distraction)
            
            # Calculate overall distraction level
            if distraction_scores:
                distraction_level = max(distraction_scores)
            else:
                distraction_level = 0.0
            
            return distraction_level, distraction_types
            
        except Exception as e:
            logger.error(f"Error detecting distractions: {str(e)}")
            return 0.0, []
    
    def _detect_visual_distractions(self, eye_tracking_data: Dict) -> float:
        """Detect visual distractions from eye-tracking data"""
        try:
            # Indicators of visual distraction
            scanpath_efficiency = eye_tracking_data.get('scanpath_efficiency', 0.5)
            fixation_duration = eye_tracking_data.get('average_fixation_duration', 200)
            saccade_frequency = eye_tracking_data.get('saccade_count', 0)
            
            # Low efficiency and irregular patterns indicate distraction
            efficiency_score = 1.0 - scanpath_efficiency
            duration_score = abs(fixation_duration - 300) / 300.0  # Optimal around 300ms
            frequency_score = abs(saccade_frequency - 10) / 20.0  # Optimal around 10
            
            distraction = (efficiency_score * 0.4 + duration_score * 0.3 + frequency_score * 0.3)
            
            return max(0, min(1, distraction))
            
        except Exception as e:
            logger.error(f"Error detecting visual distractions: {str(e)}")
            return 0.0
    
    def _detect_auditory_distractions(self, voice_data: Dict) -> float:
        """Detect auditory distractions from voice data"""
        try:
            # Indicators of auditory distraction
            jitter = voice_data.get('jitter', 0.02)
            shimmer = voice_data.get('shimmer', 0.05)
            pause_frequency = voice_data.get('pause_frequency', 0.1)
            
            # High jitter, shimmer, and pause frequency indicate distraction
            jitter_score = min(1.0, jitter * 50)
            shimmer_score = min(1.0, shimmer * 20)
            pause_score = min(1.0, pause_frequency * 10)
            
            distraction = (jitter_score * 0.4 + shimmer_score * 0.3 + pause_score * 0.3)
            
            return max(0, min(1, distraction))
            
        except Exception as e:
            logger.error(f"Error detecting auditory distractions: {str(e)}")
            return 0.0
    
    def _detect_cognitive_distractions(self, neurofeedback_data: Dict) -> float:
        """Detect cognitive distractions from neurofeedback data"""
        try:
            # Indicators of cognitive distraction
            focus_level = neurofeedback_data.get('focus_level', 0.5)
            cognitive_load = neurofeedback_data.get('cognitive_load', 0.5)
            fatigue_level = neurofeedback_data.get('fatigue_level', 0.5)
            
            # Low focus, high load, or high fatigue indicate distraction
            focus_score = 1.0 - focus_level
            load_score = cognitive_load
            fatigue_score = fatigue_level
            
            distraction = (focus_score * 0.4 + load_score * 0.3 + fatigue_score * 0.3)
            
            return max(0, min(1, distraction))
            
        except Exception as e:
            logger.error(f"Error detecting cognitive distractions: {str(e)}")
            return 0.0
    
    def _detect_emotional_distractions(self, facial_data: Dict) -> float:
        """Detect emotional distractions from facial data"""
        try:
            # Indicators of emotional distraction
            emotion_type = facial_data.get('emotion_type', 'neutral')
            emotion_intensity = facial_data.get('emotion_intensity', 0.5)
            
            # Negative emotions or high intensity indicate distraction
            negative_emotions = ['angry', 'sad', 'fear', 'disgust', 'frustrated']
            is_negative = emotion_type in negative_emotions
            
            emotion_score = 1.0 if is_negative else 0.0
            intensity_score = emotion_intensity
            
            distraction = (emotion_score * 0.6 + intensity_score * 0.4)
            
            return max(0, min(1, distraction))
            
        except Exception as e:
            logger.error(f"Error detecting emotional distractions: {str(e)}")
            return 0.0
    
    def _detect_physical_distractions(self, behavioral_data: Dict) -> float:
        """Detect physical distractions from behavioral data"""
        try:
            # Indicators of physical distraction
            mouse_movement = behavioral_data.get('mouse_movement', 0.5)
            click_frequency = behavioral_data.get('click_frequency', 0.5)
            response_time = behavioral_data.get('response_time', 1.0)
            
            # High movement, high clicks, or slow response indicate distraction
            movement_score = mouse_movement
            click_score = click_frequency
            response_score = min(1.0, response_time)
            
            distraction = (movement_score * 0.4 + click_score * 0.3 + response_score * 0.3)
            
            return max(0, min(1, distraction))
            
        except Exception as e:
            logger.error(f"Error detecting physical distractions: {str(e)}")
            return 0.0
    
    def _calculate_focus_quality(self, attention_score: float, engagement_score: float, 
                               distraction_level: float) -> float:
        """Calculate focus quality from attention, engagement, and distraction"""
        try:
            # Focus quality is high when attention and engagement are high, and distraction is low
            focus_quality = (
                attention_score * 0.4 +
                engagement_score * 0.4 +
                (1 - distraction_level) * 0.2
            )
            
            return max(0, min(1, focus_quality))
            
        except Exception as e:
            logger.error(f"Error calculating focus quality: {str(e)}")
            return 0.5
    
    def _calculate_learning_readiness(self, attention_score: float, engagement_score: float, 
                                    distraction_level: float, focus_quality: float) -> float:
        """Calculate learning readiness from multiple factors"""
        try:
            # Learning readiness is optimal when all factors are balanced
            readiness = (
                attention_score * 0.3 +
                engagement_score * 0.3 +
                (1 - distraction_level) * 0.2 +
                focus_quality * 0.2
            )
            
            return max(0, min(1, readiness))
            
        except Exception as e:
            logger.error(f"Error calculating learning readiness: {str(e)}")
            return 0.5
    
    def _calculate_confidence(self, multi_modal_data: Dict, attention_score: float, 
                            engagement_score: float) -> float:
        """Calculate overall confidence in analysis"""
        try:
            # Count available data sources
            data_sources = 0
            if multi_modal_data.get('facial_data'):
                data_sources += 1
            if multi_modal_data.get('voice_data'):
                data_sources += 1
            if multi_modal_data.get('eye_tracking_data'):
                data_sources += 1
            if multi_modal_data.get('neurofeedback_data'):
                data_sources += 1
            if multi_modal_data.get('behavioral_data'):
                data_sources += 1
            
            # Base confidence on data sources
            source_confidence = min(1.0, data_sources / 5.0)
            
            # Confidence based on score consistency
            score_consistency = 1.0 - abs(attention_score - engagement_score)
            
            # Overall confidence
            confidence = (source_confidence * 0.6 + score_consistency * 0.4)
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.1
    
    def _update_history(self, attention_entry: Dict):
        """Update attention history for trend analysis"""
        self.attention_history.append(attention_entry)
        
        # Keep only recent history
        if len(self.attention_history) > 100:
            self.attention_history = self.attention_history[-100:]
    
    def _get_default_metrics(self) -> AttentionMetrics:
        """Return default metrics when analysis fails"""
        return AttentionMetrics(
            attention_state=AttentionState.MODERATELY_ATTENTIVE,
            attention_score=0.5,
            engagement_level=EngagementLevel.MEDIUM,
            engagement_score=0.5,
            attention_span=30.0,
            distraction_level=0.5,
            distraction_types=[],
            focus_quality=0.5,
            learning_readiness=0.5,
            attention_timeline=[],
            confidence=0.1
        )
    
    def get_attention_statistics(self) -> Dict[str, float]:
        """
        Get attention statistics from historical data
        
        Returns:
            Dictionary with attention statistics
        """
        if not self.attention_history:
            return {
                'average_attention': 0.0,
                'average_engagement': 0.0,
                'attention_stability': 0.0,
                'engagement_trend': 0.0,
                'distraction_frequency': 0.0
            }
        
        # Calculate statistics
        attention_scores = [entry['attention_score'] for entry in self.attention_history]
        engagement_scores = [entry['engagement_score'] for entry in self.attention_history]
        distraction_levels = [entry['distraction_level'] for entry in self.attention_history]
        
        # Average scores
        average_attention = np.mean(attention_scores)
        average_engagement = np.mean(engagement_scores)
        
        # Attention stability (inverse of variance)
        attention_stability = 1.0 - np.var(attention_scores)
        
        # Engagement trend
        if len(engagement_scores) > 1:
            engagement_trend = np.polyfit(range(len(engagement_scores)), engagement_scores, 1)[0]
        else:
            engagement_trend = 0.0
        
        # Distraction frequency
        distraction_frequency = np.mean(distraction_levels)
        
        return {
            'average_attention': float(average_attention),
            'average_engagement': float(average_engagement),
            'attention_stability': float(attention_stability),
            'engagement_trend': float(engagement_trend),
            'distraction_frequency': float(distraction_frequency)
        }
