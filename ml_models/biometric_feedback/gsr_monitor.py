"""
Galvanic Skin Response (GSR) Monitor Module

This module provides comprehensive GSR monitoring for learning optimization including:
- Real-time GSR monitoring and analysis
- Arousal level detection and management
- Emotional state assessment
- Learning engagement evaluation
- Stress and relaxation monitoring

Author: LearnStyle AI Team
Version: 1.0.0
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from scipy import signal
from scipy.stats import entropy
import joblib

logger = logging.getLogger(__name__)

class ArousalLevel(Enum):
    """Arousal level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

class EmotionalState(Enum):
    """Emotional state classifications"""
    CALM = "calm"
    RELAXED = "relaxed"
    FOCUSED = "focused"
    EXCITED = "excited"
    STRESSED = "stressed"
    ANXIOUS = "anxious"
    OVERWHELMED = "overwhelmed"
    BORED = "bored"

class EngagementLevel(Enum):
    """Learning engagement levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class GSRMetrics:
    """GSR analysis metrics"""
    timestamp: datetime
    raw_gsr: float  # Raw GSR value in microsiemens
    tonic_gsr: float  # Tonic (baseline) GSR level
    phasic_gsr: float  # Phasic (response) GSR level
    gsr_amplitude: float  # GSR response amplitude
    gsr_frequency: float  # GSR response frequency
    skin_conductance_level: float  # SCL in microsiemens
    skin_conductance_response: float  # SCR amplitude
    arousal_index: float  # Arousal index (0-1)
    emotional_reactivity: float  # Emotional reactivity index
    stress_response: float  # Stress response index
    engagement_score: float  # Learning engagement score
    arousal_level: ArousalLevel
    emotional_state: EmotionalState
    engagement_level: EngagementLevel
    confidence: float

class GSRMonitor:
    """
    Advanced Galvanic Skin Response monitoring system
    """
    
    def __init__(self):
        """Initialize GSR monitor"""
        self.gsr_history = []
        self.baseline_gsr = None
        self.learning_thresholds = {
            'optimal_arousal': (0.3, 0.7),
            'optimal_engagement': (0.4, 0.8),
            'stress_threshold': 0.8,
            'boredom_threshold': 0.2,
            'overwhelm_threshold': 0.9
        }
        
        logger.info("GSR Monitor initialized")
    
    def analyze_gsr(self, gsr_data: List[float], 
                   sampling_rate: float = 100.0) -> GSRMetrics:
        """
        Analyze GSR data for learning optimization
        
        Args:
            gsr_data: List of GSR values in microsiemens
            sampling_rate: Sampling rate in Hz
            
        Returns:
            GSR analysis metrics
        """
        try:
            # Clean and validate GSR data
            cleaned_gsr = self._clean_gsr_data(gsr_data)
            
            if len(cleaned_gsr) < 10:
                return self._create_default_metrics()
            
            # Calculate basic GSR metrics
            basic_metrics = self._calculate_basic_metrics(cleaned_gsr)
            
            # Separate tonic and phasic components
            tonic_phasic = self._separate_tonic_phasic(cleaned_gsr, sampling_rate)
            
            # Calculate response metrics
            response_metrics = self._calculate_response_metrics(cleaned_gsr, sampling_rate)
            
            # Calculate derived metrics
            derived_metrics = self._calculate_derived_metrics(basic_metrics, tonic_phasic, response_metrics)
            
            # Assess physiological states
            arousal_level = self._assess_arousal_level(derived_metrics)
            emotional_state = self._assess_emotional_state(derived_metrics, arousal_level)
            engagement_level = self._assess_engagement_level(derived_metrics, emotional_state)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(cleaned_gsr, basic_metrics)
            
            # Create GSR metrics
            metrics = GSRMetrics(
                timestamp=datetime.now(),
                raw_gsr=cleaned_gsr[-1],
                tonic_gsr=tonic_phasic['tonic'],
                phasic_gsr=tonic_phasic['phasic'],
                gsr_amplitude=response_metrics['amplitude'],
                gsr_frequency=response_metrics['frequency'],
                skin_conductance_level=basic_metrics['scl'],
                skin_conductance_response=response_metrics['scr'],
                arousal_index=derived_metrics['arousal_index'],
                emotional_reactivity=derived_metrics['emotional_reactivity'],
                stress_response=derived_metrics['stress_response'],
                engagement_score=derived_metrics['engagement_score'],
                arousal_level=arousal_level,
                emotional_state=emotional_state,
                engagement_level=engagement_level,
                confidence=confidence
            )
            
            # Store in history
            self.gsr_history.append(metrics)
            
            # Update baseline if needed
            self._update_baseline(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing GSR: {str(e)}")
            return self._create_default_metrics()
    
    def get_learning_recommendations(self, metrics: GSRMetrics) -> List[str]:
        """
        Get learning recommendations based on GSR metrics
        
        Args:
            metrics: GSR analysis metrics
            
        Returns:
            List of learning recommendations
        """
        try:
            recommendations = []
            
            # Arousal-based recommendations
            if metrics.arousal_level == ArousalLevel.VERY_LOW:
                recommendations.extend([
                    "Very low arousal detected - try engaging content",
                    "Consider interactive or challenging materials",
                    "Take a short walk or do light exercise"
                ])
            elif metrics.arousal_level == ArousalLevel.LOW:
                recommendations.extend([
                    "Low arousal detected - increase content difficulty",
                    "Consider more stimulating learning activities",
                    "Try changing learning environment"
                ])
            elif metrics.arousal_level == ArousalLevel.MODERATE:
                recommendations.extend([
                    "Optimal arousal level - continue current approach",
                    "Good balance for learning - maintain focus",
                    "Consider slightly more challenging content"
                ])
            elif metrics.arousal_level == ArousalLevel.HIGH:
                recommendations.extend([
                    "High arousal detected - take a short break",
                    "Consider relaxation techniques",
                    "Reduce content difficulty temporarily"
                ])
            else:  # VERY_HIGH
                recommendations.extend([
                    "Very high arousal - take immediate break",
                    "Practice deep breathing or meditation",
                    "Avoid high-pressure learning situations"
                ])
            
            # Emotional state recommendations
            if metrics.emotional_state == EmotionalState.STRESSED:
                recommendations.extend([
                    "Stress detected - practice stress management",
                    "Consider mindfulness or relaxation exercises",
                    "Break down content into smaller chunks"
                ])
            elif metrics.emotional_state == EmotionalState.ANXIOUS:
                recommendations.extend([
                    "Anxiety detected - focus on breathing",
                    "Consider progressive muscle relaxation",
                    "Start with easier content to build confidence"
                ])
            elif metrics.emotional_state == EmotionalState.BORED:
                recommendations.extend([
                    "Boredom detected - increase content variety",
                    "Try different learning methods or formats",
                    "Set more challenging goals"
                ])
            elif metrics.emotional_state == EmotionalState.OVERWHELMED:
                recommendations.extend([
                    "Overwhelm detected - take a longer break",
                    "Simplify content and reduce cognitive load",
                    "Consider seeking support or guidance"
                ])
            
            # Engagement-based recommendations
            if metrics.engagement_level == EngagementLevel.VERY_LOW:
                recommendations.extend([
                    "Very low engagement - try different content type",
                    "Consider hands-on or interactive learning",
                    "Take a break and return with fresh perspective"
                ])
            elif metrics.engagement_level == EngagementLevel.LOW:
                recommendations.extend([
                    "Low engagement - increase content relevance",
                    "Try connecting content to personal interests",
                    "Consider group learning or discussion"
                ])
            elif metrics.engagement_level == EngagementLevel.HIGH:
                recommendations.extend([
                    "High engagement - excellent learning state",
                    "Consider advanced or challenging content",
                    "Maintain current learning approach"
                ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting learning recommendations: {str(e)}")
            return ["Monitor your emotional state and adjust learning accordingly"]
    
    def detect_engagement_patterns(self, window_minutes: int = 30) -> Dict[str, Any]:
        """
        Detect engagement patterns over time window
        
        Args:
            window_minutes: Time window for analysis in minutes
            
        Returns:
            Engagement pattern analysis
        """
        try:
            # Get recent metrics within window
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            recent_metrics = [m for m in self.gsr_history if m.timestamp >= cutoff_time]
            
            if len(recent_metrics) < 3:
                return {'error': 'Insufficient data for pattern analysis'}
            
            # Analyze engagement trends
            engagement_scores = [m.engagement_score for m in recent_metrics]
            arousal_levels = [m.arousal_index for m in recent_metrics]
            timestamps = [m.timestamp for m in recent_metrics]
            
            # Calculate trends
            engagement_trend = np.polyfit(range(len(engagement_scores)), engagement_scores, 1)[0]
            arousal_trend = np.polyfit(range(len(arousal_levels)), arousal_levels, 1)[0]
            
            # Detect engagement spikes and drops
            engagement_mean = np.mean(engagement_scores)
            engagement_std = np.std(engagement_scores)
            engagement_spikes = [i for i, val in enumerate(engagement_scores) 
                               if val > engagement_mean + 2 * engagement_std]
            engagement_drops = [i for i, val in enumerate(engagement_scores) 
                              if val < engagement_mean - 2 * engagement_std]
            
            # Determine pattern type
            if engagement_trend > 0.1:
                pattern_type = "increasing_engagement"
            elif engagement_trend < -0.1:
                pattern_type = "decreasing_engagement"
            elif len(engagement_spikes) > len(engagement_drops):
                pattern_type = "variable_high_engagement"
            elif len(engagement_drops) > len(engagement_spikes):
                pattern_type = "variable_low_engagement"
            else:
                pattern_type = "stable_engagement"
            
            return {
                'pattern_type': pattern_type,
                'engagement_trend': engagement_trend,
                'arousal_trend': arousal_trend,
                'engagement_spikes': len(engagement_spikes),
                'engagement_drops': len(engagement_drops),
                'average_engagement': engagement_mean,
                'engagement_variability': engagement_std,
                'max_engagement': np.max(engagement_scores),
                'min_engagement': np.min(engagement_scores),
                'recommendations': self._get_engagement_pattern_recommendations(pattern_type, engagement_trend)
            }
            
        except Exception as e:
            logger.error(f"Error detecting engagement patterns: {str(e)}")
            return {'error': str(e)}
    
    def _clean_gsr_data(self, gsr_data: List[float]) -> List[float]:
        """Clean and validate GSR data"""
        try:
            # Remove outliers (GSR values outside physiological range)
            cleaned = [gsr for gsr in gsr_data if 0.1 <= gsr <= 50.0]  # 0.1-50 microsiemens
            
            # Remove artifacts using median filter
            if len(cleaned) > 5:
                cleaned = signal.medfilt(cleaned, kernel_size=3).tolist()
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning GSR data: {str(e)}")
            return gsr_data
    
    def _calculate_basic_metrics(self, gsr_data: List[float]) -> Dict[str, float]:
        """Calculate basic GSR metrics"""
        try:
            gsr_array = np.array(gsr_data)
            
            # Skin Conductance Level (SCL) - mean GSR
            scl = np.mean(gsr_array)
            
            # GSR variability
            gsr_std = np.std(gsr_array)
            
            # GSR range
            gsr_range = np.max(gsr_array) - np.min(gsr_array)
            
            return {
                'scl': scl,
                'gsr_std': gsr_std,
                'gsr_range': gsr_range
            }
            
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {str(e)}")
            return {'scl': 0, 'gsr_std': 0, 'gsr_range': 0}
    
    def _separate_tonic_phasic(self, gsr_data: List[float], 
                              sampling_rate: float) -> Dict[str, float]:
        """Separate tonic and phasic GSR components"""
        try:
            gsr_array = np.array(gsr_data)
            
            # Low-pass filter for tonic component (cutoff ~0.05 Hz)
            nyquist = sampling_rate / 2
            cutoff = 0.05 / nyquist
            b, a = signal.butter(4, cutoff, btype='low')
            tonic = signal.filtfilt(b, a, gsr_array)
            
            # Phasic component is the difference
            phasic = gsr_array - tonic
            
            return {
                'tonic': np.mean(tonic),
                'phasic': np.mean(np.abs(phasic))
            }
            
        except Exception as e:
            logger.error(f"Error separating tonic/phasic: {str(e)}")
            return {'tonic': 0, 'phasic': 0}
    
    def _calculate_response_metrics(self, gsr_data: List[float], 
                                  sampling_rate: float) -> Dict[str, float]:
        """Calculate GSR response metrics"""
        try:
            gsr_array = np.array(gsr_data)
            
            # Find peaks (SCRs)
            peaks, _ = signal.find_peaks(gsr_array, height=np.mean(gsr_array) + np.std(gsr_array))
            
            # Calculate SCR amplitude
            scr_amplitude = np.mean(gsr_array[peaks]) - np.mean(gsr_array) if len(peaks) > 0 else 0
            
            # Calculate response frequency
            response_frequency = len(peaks) / (len(gsr_array) / sampling_rate) if len(gsr_array) > 0 else 0
            
            return {
                'amplitude': scr_amplitude,
                'frequency': response_frequency,
                'scr': scr_amplitude
            }
            
        except Exception as e:
            logger.error(f"Error calculating response metrics: {str(e)}")
            return {'amplitude': 0, 'frequency': 0, 'scr': 0}
    
    def _calculate_derived_metrics(self, basic_metrics: Dict[str, float], 
                                 tonic_phasic: Dict[str, float], 
                                 response_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate derived GSR metrics"""
        try:
            # Arousal index (normalized SCL)
            arousal_index = min(1.0, basic_metrics['scl'] / 20.0)  # Normalize to 0-1
            
            # Emotional reactivity (phasic component)
            emotional_reactivity = min(1.0, tonic_phasic['phasic'] / 5.0)  # Normalize to 0-1
            
            # Stress response (combination of SCL and SCR)
            stress_response = (arousal_index + emotional_reactivity) / 2
            
            # Engagement score (optimal arousal + moderate reactivity)
            optimal_arousal = 1.0 - abs(arousal_index - 0.5) * 2  # Peak at 0.5
            optimal_reactivity = 1.0 - abs(emotional_reactivity - 0.3) * 2  # Peak at 0.3
            engagement_score = (optimal_arousal + optimal_reactivity) / 2
            
            return {
                'arousal_index': arousal_index,
                'emotional_reactivity': emotional_reactivity,
                'stress_response': stress_response,
                'engagement_score': engagement_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating derived metrics: {str(e)}")
            return {'arousal_index': 0.5, 'emotional_reactivity': 0.3, 'stress_response': 0.4, 'engagement_score': 0.5}
    
    def _assess_arousal_level(self, derived_metrics: Dict[str, float]) -> ArousalLevel:
        """Assess arousal level from derived metrics"""
        try:
            arousal_index = derived_metrics['arousal_index']
            
            if arousal_index < 0.2:
                return ArousalLevel.VERY_LOW
            elif arousal_index < 0.4:
                return ArousalLevel.LOW
            elif arousal_index < 0.6:
                return ArousalLevel.MODERATE
            elif arousal_index < 0.8:
                return ArousalLevel.HIGH
            else:
                return ArousalLevel.VERY_HIGH
                
        except Exception as e:
            logger.error(f"Error assessing arousal level: {str(e)}")
            return ArousalLevel.MODERATE
    
    def _assess_emotional_state(self, derived_metrics: Dict[str, float], 
                              arousal_level: ArousalLevel) -> EmotionalState:
        """Assess emotional state from GSR metrics"""
        try:
            arousal_index = derived_metrics['arousal_index']
            emotional_reactivity = derived_metrics['emotional_reactivity']
            stress_response = derived_metrics['stress_response']
            
            # High stress/overwhelm
            if stress_response > 0.8:
                return EmotionalState.OVERWHELMED
            elif stress_response > 0.6:
                return EmotionalState.STRESSED
            elif arousal_index > 0.7 and emotional_reactivity > 0.5:
                return EmotionalState.ANXIOUS
            elif arousal_index < 0.3 and emotional_reactivity < 0.2:
                return EmotionalState.BORED
            elif arousal_index > 0.6 and emotional_reactivity > 0.4:
                return EmotionalState.EXCITED
            elif arousal_index > 0.4 and emotional_reactivity < 0.3:
                return EmotionalState.FOCUSED
            elif arousal_index < 0.5 and emotional_reactivity < 0.3:
                return EmotionalState.RELAXED
            else:
                return EmotionalState.CALM
                
        except Exception as e:
            logger.error(f"Error assessing emotional state: {str(e)}")
            return EmotionalState.CALM
    
    def _assess_engagement_level(self, derived_metrics: Dict[str, float], 
                               emotional_state: EmotionalState) -> EngagementLevel:
        """Assess learning engagement level"""
        try:
            engagement_score = derived_metrics['engagement_score']
            
            # Adjust based on emotional state
            if emotional_state in [EmotionalState.BORED, EmotionalState.OVERWHELMED]:
                engagement_score *= 0.5
            elif emotional_state in [EmotionalState.FOCUSED, EmotionalState.EXCITED]:
                engagement_score *= 1.2
            
            # Clamp to 0-1 range
            engagement_score = min(1.0, max(0.0, engagement_score))
            
            if engagement_score < 0.2:
                return EngagementLevel.VERY_LOW
            elif engagement_score < 0.4:
                return EngagementLevel.LOW
            elif engagement_score < 0.6:
                return EngagementLevel.MODERATE
            elif engagement_score < 0.8:
                return EngagementLevel.HIGH
            else:
                return EngagementLevel.VERY_HIGH
                
        except Exception as e:
            logger.error(f"Error assessing engagement level: {str(e)}")
            return EngagementLevel.MODERATE
    
    def _calculate_confidence_score(self, gsr_data: List[float], 
                                  basic_metrics: Dict[str, float]) -> float:
        """Calculate confidence score for GSR analysis"""
        try:
            # Base confidence on data quality
            data_quality = min(1.0, len(gsr_data) / 100)  # More data = higher confidence
            
            # Confidence based on signal stability
            signal_stability = 1.0 - min(1.0, basic_metrics['gsr_std'] / 5.0)  # Lower std = more stable
            
            # Overall confidence
            confidence = (data_quality + signal_stability) / 2
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.5
    
    def _update_baseline(self, metrics: GSRMetrics):
        """Update baseline GSR metrics"""
        try:
            if self.baseline_gsr is None:
                self.baseline_gsr = metrics
            else:
                # Update baseline with exponential moving average
                alpha = 0.1  # Learning rate
                self.baseline_gsr.tonic_gsr = (1 - alpha) * self.baseline_gsr.tonic_gsr + alpha * metrics.tonic_gsr
                self.baseline_gsr.phasic_gsr = (1 - alpha) * self.baseline_gsr.phasic_gsr + alpha * metrics.phasic_gsr
                self.baseline_gsr.arousal_index = (1 - alpha) * self.baseline_gsr.arousal_index + alpha * metrics.arousal_index
                
        except Exception as e:
            logger.error(f"Error updating baseline: {str(e)}")
    
    def _get_engagement_pattern_recommendations(self, pattern_type: str, 
                                              engagement_trend: float) -> List[str]:
        """Get recommendations based on engagement patterns"""
        try:
            recommendations = []
            
            if pattern_type == "increasing_engagement":
                recommendations.extend([
                    "Engagement is increasing - excellent progress",
                    "Consider maintaining current learning approach",
                    "Gradually increase content difficulty"
                ])
            elif pattern_type == "decreasing_engagement":
                recommendations.extend([
                    "Engagement is decreasing - try different content",
                    "Consider changing learning methods or environment",
                    "Take breaks and return with fresh perspective"
                ])
            elif pattern_type == "variable_high_engagement":
                recommendations.extend([
                    "Variable high engagement - identify peak conditions",
                    "Try to replicate high-engagement learning sessions",
                    "Consider scheduling learning during peak times"
                ])
            elif pattern_type == "variable_low_engagement":
                recommendations.extend([
                    "Variable low engagement - address underlying issues",
                    "Consider stress management or motivation techniques",
                    "Try different content types or learning styles"
                ])
            else:  # stable_engagement
                recommendations.extend([
                    "Stable engagement levels - maintain current approach",
                    "Consider optimizing learning schedule",
                    "Monitor for opportunities to increase engagement"
                ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting engagement pattern recommendations: {str(e)}")
            return ["Monitor engagement levels and adjust learning accordingly"]
    
    def _create_default_metrics(self) -> GSRMetrics:
        """Create default GSR metrics when analysis fails"""
        return GSRMetrics(
            timestamp=datetime.now(),
            raw_gsr=5.0,
            tonic_gsr=4.5,
            phasic_gsr=0.5,
            gsr_amplitude=0.3,
            gsr_frequency=0.1,
            skin_conductance_level=5.0,
            skin_conductance_response=0.3,
            arousal_index=0.5,
            emotional_reactivity=0.3,
            stress_response=0.4,
            engagement_score=0.6,
            arousal_level=ArousalLevel.MODERATE,
            emotional_state=EmotionalState.CALM,
            engagement_level=EngagementLevel.MODERATE,
            confidence=0.3
        )
    
    def get_monitor_statistics(self) -> Dict[str, int]:
        """Get GSR monitor statistics"""
        try:
            return {
                'total_analyses': len(self.gsr_history),
                'baseline_established': 1 if self.baseline_gsr else 0,
                'learning_thresholds': len(self.learning_thresholds)
            }
            
        except Exception as e:
            logger.error(f"Error getting monitor statistics: {str(e)}")
            return {
                'total_analyses': 0,
                'baseline_established': 0,
                'learning_thresholds': 0
            }
