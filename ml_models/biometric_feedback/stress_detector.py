"""
Stress Detector Module

This module provides comprehensive stress detection and intervention for learning optimization.

Author: LearnStyle AI Team
Version: 1.0.0
"""

import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class StressLevel(Enum):
    """Stress level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

class StressType(Enum):
    """Types of stress"""
    ACUTE = "acute"
    CHRONIC = "chronic"
    EUSTRESS = "eustress"
    DISTRESS = "distress"

@dataclass
class StressMetrics:
    """Stress detection metrics"""
    timestamp: datetime
    stress_level: StressLevel
    stress_type: StressType
    stress_intensity: float
    stress_duration: int  # minutes
    physiological_indicators: Dict[str, float]
    behavioral_indicators: Dict[str, float]
    cognitive_indicators: Dict[str, float]
    intervention_required: bool
    recommended_interventions: List[str]
    confidence: float

class StressDetector:
    """
    Advanced stress detection and intervention system
    """
    
    def __init__(self):
        """Initialize stress detector"""
        self.stress_history = []
        self.intervention_history = []
        
        logger.info("Stress Detector initialized")
    
    def detect_stress(self, biometric_data: Dict[str, Any], 
                     behavioral_data: Optional[Dict[str, Any]] = None) -> StressMetrics:
        """
        Detect stress from biometric and behavioral data
        
        Args:
            biometric_data: Biometric sensor data
            behavioral_data: Behavioral indicators data
            
        Returns:
            Stress detection metrics
        """
        try:
            # Analyze physiological indicators
            physiological_indicators = self._analyze_physiological_indicators(biometric_data)
            
            # Analyze behavioral indicators
            behavioral_indicators = self._analyze_behavioral_indicators(behavioral_data)
            
            # Analyze cognitive indicators
            cognitive_indicators = self._analyze_cognitive_indicators(biometric_data)
            
            # Calculate overall stress level
            stress_level = self._calculate_stress_level(physiological_indicators, 
                                                      behavioral_indicators, 
                                                      cognitive_indicators)
            
            # Determine stress type
            stress_type = self._determine_stress_type(stress_level, physiological_indicators)
            
            # Calculate stress intensity
            stress_intensity = self._calculate_stress_intensity(physiological_indicators)
            
            # Calculate stress duration
            stress_duration = self._calculate_stress_duration(stress_level)
            
            # Determine if intervention is required
            intervention_required = self._assess_intervention_need(stress_level, stress_intensity)
            
            # Generate intervention recommendations
            recommended_interventions = self._generate_interventions(stress_level, stress_type, 
                                                                   physiological_indicators)
            
            # Calculate confidence
            confidence = self._calculate_confidence(physiological_indicators, behavioral_indicators)
            
            # Create stress metrics
            metrics = StressMetrics(
                timestamp=datetime.now(),
                stress_level=stress_level,
                stress_type=stress_type,
                stress_intensity=stress_intensity,
                stress_duration=stress_duration,
                physiological_indicators=physiological_indicators,
                behavioral_indicators=behavioral_indicators,
                cognitive_indicators=cognitive_indicators,
                intervention_required=intervention_required,
                recommended_interventions=recommended_interventions,
                confidence=confidence
            )
            
            # Store in history
            self.stress_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error detecting stress: {str(e)}")
            return self._create_default_metrics()
    
    def _analyze_physiological_indicators(self, biometric_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze physiological stress indicators"""
        try:
            indicators = {}
            
            # HRV indicators
            if 'hrv' in biometric_data:
                hrv_data = biometric_data['hrv']
                indicators['hrv_stress_index'] = hrv_data.get('stress_index', 0.5)
                indicators['hrv_rmssd'] = hrv_data.get('rmssd', 30.0)
                indicators['hrv_lf_hf_ratio'] = hrv_data.get('lf_hf_ratio', 1.0)
            
            # GSR indicators
            if 'gsr' in biometric_data:
                gsr_data = biometric_data['gsr']
                indicators['gsr_arousal'] = gsr_data.get('arousal_index', 0.5)
                indicators['gsr_stress_response'] = gsr_data.get('stress_response', 0.5)
                indicators['gsr_tonic'] = gsr_data.get('tonic_gsr', 5.0)
            
            # EEG indicators
            if 'eeg' in biometric_data:
                eeg_data = biometric_data['eeg']
                indicators['eeg_alpha_power'] = eeg_data.get('alpha_power', 0.5)
                indicators['eeg_beta_power'] = eeg_data.get('beta_power', 0.5)
                indicators['eeg_theta_power'] = eeg_data.get('theta_power', 0.5)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error analyzing physiological indicators: {str(e)}")
            return {}
    
    def _analyze_behavioral_indicators(self, behavioral_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze behavioral stress indicators"""
        try:
            if not behavioral_data:
                return {}
            
            indicators = {}
            
            # Learning behavior indicators
            indicators['response_time'] = behavioral_data.get('response_time', 0.5)
            indicators['error_rate'] = behavioral_data.get('error_rate', 0.1)
            indicators['completion_rate'] = behavioral_data.get('completion_rate', 0.8)
            indicators['engagement_level'] = behavioral_data.get('engagement_level', 0.5)
            
            # Interaction patterns
            indicators['click_frequency'] = behavioral_data.get('click_frequency', 0.5)
            indicators['scroll_speed'] = behavioral_data.get('scroll_speed', 0.5)
            indicators['pause_frequency'] = behavioral_data.get('pause_frequency', 0.5)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error analyzing behavioral indicators: {str(e)}")
            return {}
    
    def _analyze_cognitive_indicators(self, biometric_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze cognitive stress indicators"""
        try:
            indicators = {}
            
            # Attention indicators
            if 'eye_tracking' in biometric_data:
                eye_data = biometric_data['eye_tracking']
                indicators['attention_score'] = eye_data.get('attention_score', 0.5)
                indicators['fixation_duration'] = eye_data.get('fixation_duration', 0.5)
                indicators['saccade_rate'] = eye_data.get('saccade_rate', 0.5)
            
            # Cognitive load indicators
            if 'neurofeedback' in biometric_data:
                neuro_data = biometric_data['neurofeedback']
                indicators['cognitive_load'] = neuro_data.get('cognitive_load', 0.5)
                indicators['focus_score'] = neuro_data.get('focus_score', 0.5)
                indicators['fatigue_level'] = neuro_data.get('fatigue_level', 0.5)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error analyzing cognitive indicators: {str(e)}")
            return {}
    
    def _calculate_stress_level(self, physiological: Dict[str, float], 
                              behavioral: Dict[str, float], 
                              cognitive: Dict[str, float]) -> StressLevel:
        """Calculate overall stress level"""
        try:
            # Physiological stress score
            phys_score = 0.0
            if physiological:
                phys_score = (physiological.get('hrv_stress_index', 0.5) + 
                            physiological.get('gsr_stress_response', 0.5)) / 2
            
            # Behavioral stress score
            behav_score = 0.0
            if behavioral:
                behav_score = (behavioral.get('error_rate', 0.1) + 
                             (1.0 - behavioral.get('completion_rate', 0.8)) + 
                             behavioral.get('pause_frequency', 0.5)) / 3
            
            # Cognitive stress score
            cog_score = 0.0
            if cognitive:
                cog_score = (cognitive.get('cognitive_load', 0.5) + 
                           (1.0 - cognitive.get('attention_score', 0.5)) + 
                           cognitive.get('fatigue_level', 0.5)) / 3
            
            # Overall stress score (weighted average)
            overall_score = (phys_score * 0.5 + behav_score * 0.3 + cog_score * 0.2)
            
            # Classify stress level
            if overall_score < 0.2:
                return StressLevel.VERY_LOW
            elif overall_score < 0.4:
                return StressLevel.LOW
            elif overall_score < 0.6:
                return StressLevel.MODERATE
            elif overall_score < 0.8:
                return StressLevel.HIGH
            else:
                return StressLevel.VERY_HIGH
                
        except Exception as e:
            logger.error(f"Error calculating stress level: {str(e)}")
            return StressLevel.MODERATE
    
    def _determine_stress_type(self, stress_level: StressLevel, 
                             physiological: Dict[str, float]) -> StressType:
        """Determine type of stress"""
        try:
            # Check for chronic stress indicators
            if (stress_level in [StressLevel.HIGH, StressLevel.VERY_HIGH] and 
                physiological.get('hrv_lf_hf_ratio', 1.0) > 2.0):
                return StressType.CHRONIC
            
            # Check for acute stress indicators
            elif (stress_level in [StressLevel.HIGH, StressLevel.VERY_HIGH] and 
                  physiological.get('gsr_arousal', 0.5) > 0.7):
                return StressType.ACUTE
            
            # Check for eustress (positive stress)
            elif (stress_level in [StressLevel.MODERATE, StressLevel.HIGH] and 
                  physiological.get('gsr_arousal', 0.5) > 0.5 and 
                  physiological.get('hrv_stress_index', 0.5) < 0.6):
                return StressType.EUSTRESS
            
            # Default to distress
            else:
                return StressType.DISTRESS
                
        except Exception as e:
            logger.error(f"Error determining stress type: {str(e)}")
            return StressType.DISTRESS
    
    def _calculate_stress_intensity(self, physiological: Dict[str, float]) -> float:
        """Calculate stress intensity (0-1)"""
        try:
            if not physiological:
                return 0.5
            
            # Combine multiple indicators
            intensity = (physiological.get('hrv_stress_index', 0.5) + 
                        physiological.get('gsr_stress_response', 0.5) + 
                        physiological.get('gsr_arousal', 0.5)) / 3
            
            return min(1.0, max(0.0, intensity))
            
        except Exception as e:
            logger.error(f"Error calculating stress intensity: {str(e)}")
            return 0.5
    
    def _calculate_stress_duration(self, stress_level: StressLevel) -> int:
        """Calculate estimated stress duration in minutes"""
        try:
            # Base duration on stress level
            duration_map = {
                StressLevel.VERY_LOW: 5,
                StressLevel.LOW: 10,
                StressLevel.MODERATE: 20,
                StressLevel.HIGH: 45,
                StressLevel.VERY_HIGH: 90
            }
            
            return duration_map.get(stress_level, 20)
            
        except Exception as e:
            logger.error(f"Error calculating stress duration: {str(e)}")
            return 20
    
    def _assess_intervention_need(self, stress_level: StressLevel, 
                                stress_intensity: float) -> bool:
        """Assess if intervention is needed"""
        try:
            # Intervention needed for high stress levels or high intensity
            return (stress_level in [StressLevel.HIGH, StressLevel.VERY_HIGH] or 
                   stress_intensity > 0.7)
            
        except Exception as e:
            logger.error(f"Error assessing intervention need: {str(e)}")
            return False
    
    def _generate_interventions(self, stress_level: StressLevel, 
                              stress_type: StressType, 
                              physiological: Dict[str, float]) -> List[str]:
        """Generate stress intervention recommendations"""
        try:
            interventions = []
            
            # Immediate interventions for high stress
            if stress_level in [StressLevel.HIGH, StressLevel.VERY_HIGH]:
                interventions.extend([
                    "Take immediate break (5-15 minutes)",
                    "Practice deep breathing exercises",
                    "Step away from learning content"
                ])
            
            # Type-specific interventions
            if stress_type == StressType.ACUTE:
                interventions.extend([
                    "Focus on immediate stress relief techniques",
                    "Practice grounding exercises",
                    "Use progressive muscle relaxation"
                ])
            elif stress_type == StressType.CHRONIC:
                interventions.extend([
                    "Consider long-term stress management strategies",
                    "Evaluate learning schedule and workload",
                    "Practice regular mindfulness or meditation"
                ])
            elif stress_type == StressType.EUSTRESS:
                interventions.extend([
                    "Channel positive stress into learning motivation",
                    "Maintain current pace but monitor for overstimulation",
                    "Use stress as a learning catalyst"
                ])
            
            # Physiological-based interventions
            if physiological.get('hrv_stress_index', 0.5) > 0.7:
                interventions.extend([
                    "Practice heart rate variability training",
                    "Use biofeedback techniques",
                    "Consider meditation or yoga"
                ])
            
            if physiological.get('gsr_arousal', 0.5) > 0.8:
                interventions.extend([
                    "Practice calming techniques",
                    "Use aromatherapy or calming music",
                    "Consider environmental adjustments"
                ])
            
            return interventions[:5]  # Return top 5 interventions
            
        except Exception as e:
            logger.error(f"Error generating interventions: {str(e)}")
            return ["Take a break and practice stress management techniques"]
    
    def _calculate_confidence(self, physiological: Dict[str, float], 
                            behavioral: Dict[str, float]) -> float:
        """Calculate confidence in stress detection"""
        try:
            # Base confidence on data availability
            data_quality = 0.0
            
            if physiological:
                data_quality += 0.6
            if behavioral:
                data_quality += 0.4
            
            # Adjust for data consistency
            if physiological and len(physiological) > 3:
                data_quality += 0.1
            
            return min(1.0, data_quality)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def _create_default_metrics(self) -> StressMetrics:
        """Create default stress metrics when detection fails"""
        return StressMetrics(
            timestamp=datetime.now(),
            stress_level=StressLevel.MODERATE,
            stress_type=StressType.DISTRESS,
            stress_intensity=0.5,
            stress_duration=20,
            physiological_indicators={},
            behavioral_indicators={},
            cognitive_indicators={},
            intervention_required=False,
            recommended_interventions=["Monitor stress levels and take breaks as needed"],
            confidence=0.3
        )
    
    def get_detector_statistics(self) -> Dict[str, int]:
        """Get stress detector statistics"""
        try:
            return {
                'total_detections': len(self.stress_history),
                'interventions_given': len(self.intervention_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting detector statistics: {str(e)}")
            return {
                'total_detections': 0,
                'interventions_given': 0
            }
