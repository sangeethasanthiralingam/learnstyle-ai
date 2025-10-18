"""
Mental Fatigue Monitoring Module

This module provides real-time mental fatigue detection and monitoring:
- Theta/Alpha ratio analysis for fatigue detection
- Cognitive load assessment
- Break recommendation system
- Learning session optimization

Author: LearnStyle AI Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FatigueLevel(Enum):
    """Fatigue level classifications"""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

class BreakType(Enum):
    """Break type recommendations"""
    MICRO_BREAK = "micro_break"      # 1-2 minutes
    SHORT_BREAK = "short_break"      # 5-10 minutes
    MEDIUM_BREAK = "medium_break"    # 15-30 minutes
    LONG_BREAK = "long_break"        # 30+ minutes
    SESSION_END = "session_end"      # End learning session

@dataclass
class FatigueMetrics:
    """Mental fatigue measurement metrics"""
    fatigue_level: float
    fatigue_classification: FatigueLevel
    theta_alpha_ratio: float
    cognitive_load: float
    break_recommendation: BreakType
    break_duration: int  # minutes
    confidence: float
    session_quality: str  # 'excellent', 'good', 'fair', 'poor'
    recommendation: str

class FatigueMonitor:
    """
    Real-time mental fatigue monitoring and break recommendation system
    """
    
    def __init__(self, 
                 fatigue_threshold: float = 0.6,
                 monitoring_window: int = 15,
                 session_duration_limit: int = 120):
        """
        Initialize fatigue monitor
        
        Args:
            fatigue_threshold: Threshold for fatigue detection
            monitoring_window: Window size for fatigue analysis (seconds)
            session_duration_limit: Maximum recommended session duration (minutes)
        """
        self.fatigue_threshold = fatigue_threshold
        self.monitoring_window = monitoring_window
        self.session_duration_limit = session_duration_limit
        
        # Fatigue level thresholds
        self.fatigue_thresholds = {
            FatigueLevel.NONE: 0.3,
            FatigueLevel.MILD: 0.5,
            FatigueLevel.MODERATE: 0.7,
            FatigueLevel.SEVERE: 0.85,
            FatigueLevel.CRITICAL: 0.95
        }
        
        # Historical data for trend analysis
        self.fatigue_history = []
        self.session_start_time = None
        self.break_history = []
        
        logger.info("Fatigue Monitor initialized")
    
    def monitor_fatigue(self, eeg_features: Dict[str, float], 
                       session_duration: Optional[int] = None) -> FatigueMetrics:
        """
        Monitor mental fatigue from EEG features
        
        Args:
            eeg_features: Dictionary containing EEG band powers and features
            session_duration: Current session duration in minutes
            
        Returns:
            FatigueMetrics object with fatigue analysis
        """
        try:
            # Extract relevant features
            theta_power = eeg_features.get('theta_power_raw', 0)
            alpha_power = eeg_features.get('alpha_power_raw', 0)
            beta_power = eeg_features.get('beta_power_raw', 0)
            theta_alpha_ratio = eeg_features.get('theta_alpha_ratio', 0)
            
            # Calculate fatigue level
            fatigue_level = self._calculate_fatigue_level(theta_power, alpha_power, beta_power)
            
            # Classify fatigue level
            fatigue_classification = self._classify_fatigue_level(fatigue_level)
            
            # Assess cognitive load
            cognitive_load = self._assess_cognitive_load(eeg_features)
            
            # Generate break recommendation
            break_recommendation, break_duration = self._generate_break_recommendation(
                fatigue_classification, session_duration
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(eeg_features)
            
            # Assess session quality
            session_quality = self._assess_session_quality(fatigue_level, session_duration)
            
            # Generate recommendation
            recommendation = self._generate_fatigue_recommendation(
                fatigue_classification, break_recommendation, session_quality
            )
            
            # Update history
            self._update_history(fatigue_level, cognitive_load)
            
            return FatigueMetrics(
                fatigue_level=fatigue_level,
                fatigue_classification=fatigue_classification,
                theta_alpha_ratio=theta_alpha_ratio,
                cognitive_load=cognitive_load,
                break_recommendation=break_recommendation,
                break_duration=break_duration,
                confidence=confidence,
                session_quality=session_quality,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error monitoring fatigue: {str(e)}")
            return self._get_default_metrics()
    
    def _calculate_fatigue_level(self, theta_power: float, 
                               alpha_power: float, beta_power: float) -> float:
        """
        Calculate fatigue level from EEG band powers
        
        Args:
            theta_power: Theta band power
            alpha_power: Alpha band power
            beta_power: Beta band power
            
        Returns:
            Fatigue level (0-1)
        """
        # Avoid division by zero
        if alpha_power <= 0:
            alpha_power = 1e-6
        
        # Theta/Alpha ratio is a strong indicator of fatigue
        theta_alpha_ratio = theta_power / alpha_power
        
        # Beta power decrease also indicates fatigue
        total_power = alpha_power + beta_power + theta_power
        if total_power > 0:
            beta_ratio = beta_power / total_power
        else:
            beta_ratio = 0
        
        # Combine metrics for fatigue level
        # Higher theta/alpha ratio and lower beta ratio indicate more fatigue
        fatigue_level = (theta_alpha_ratio * 0.7 + (1 - beta_ratio) * 0.3)
        
        # Normalize to 0-1 range
        fatigue_level = max(0, min(1, fatigue_level))
        
        return fatigue_level
    
    def _classify_fatigue_level(self, fatigue_level: float) -> FatigueLevel:
        """
        Classify fatigue level into categories
        
        Args:
            fatigue_level: Calculated fatigue level (0-1)
            
        Returns:
            FatigueLevel enum value
        """
        if fatigue_level >= self.fatigue_thresholds[FatigueLevel.CRITICAL]:
            return FatigueLevel.CRITICAL
        elif fatigue_level >= self.fatigue_thresholds[FatigueLevel.SEVERE]:
            return FatigueLevel.SEVERE
        elif fatigue_level >= self.fatigue_thresholds[FatigueLevel.MODERATE]:
            return FatigueLevel.MODERATE
        elif fatigue_level >= self.fatigue_thresholds[FatigueLevel.MILD]:
            return FatigueLevel.MILD
        else:
            return FatigueLevel.NONE
    
    def _assess_cognitive_load(self, eeg_features: Dict[str, float]) -> float:
        """
        Assess cognitive load from EEG features
        
        Args:
            eeg_features: EEG features dictionary
            
        Returns:
            Cognitive load level (0-1)
        """
        # Gamma power indicates high cognitive processing
        gamma_power = eeg_features.get('gamma_power_raw', 0)
        
        # Spectral centroid indicates overall brain activity
        spectral_centroid = eeg_features.get('spectral_centroid', 0)
        
        # Signal complexity indicates cognitive engagement
        complexity = eeg_features.get('complexity', 0)
        
        # Normalize and combine metrics
        gamma_score = min(1.0, gamma_power / 100.0)  # Normalize gamma power
        spectral_score = min(1.0, spectral_centroid / 30.0)  # Normalize spectral centroid
        complexity_score = min(1.0, complexity / 2.0)  # Normalize complexity
        
        # Weighted combination
        cognitive_load = (gamma_score * 0.4 + spectral_score * 0.3 + complexity_score * 0.3)
        
        return max(0, min(1, cognitive_load))
    
    def _generate_break_recommendation(self, fatigue_classification: FatigueLevel,
                                     session_duration: Optional[int]) -> Tuple[BreakType, int]:
        """
        Generate break recommendation based on fatigue level
        
        Args:
            fatigue_classification: Fatigue level classification
            session_duration: Current session duration in minutes
            
        Returns:
            Tuple of (break_type, break_duration_minutes)
        """
        # Check session duration limits
        if session_duration and session_duration >= self.session_duration_limit:
            return BreakType.SESSION_END, 0
        
        # Generate recommendation based on fatigue level
        if fatigue_classification == FatigueLevel.CRITICAL:
            return BreakType.LONG_BREAK, 45
        elif fatigue_classification == FatigueLevel.SEVERE:
            return BreakType.MEDIUM_BREAK, 20
        elif fatigue_classification == FatigueLevel.MODERATE:
            return BreakType.SHORT_BREAK, 10
        elif fatigue_classification == FatigueLevel.MILD:
            return BreakType.MICRO_BREAK, 3
        else:
            return BreakType.MICRO_BREAK, 1
    
    def _calculate_confidence(self, eeg_features: Dict[str, float]) -> float:
        """
        Calculate confidence in fatigue detection
        
        Args:
            eeg_features: EEG features dictionary
            
        Returns:
            Confidence level (0-1)
        """
        # Check signal quality indicators
        complexity = eeg_features.get('complexity', 0)
        spectral_centroid = eeg_features.get('spectral_centroid', 0)
        
        # Higher complexity indicates better signal quality
        complexity_score = min(1.0, complexity / 2.0)
        
        # Reasonable spectral centroid indicates good signal
        spectral_score = 1.0 if 5 <= spectral_centroid <= 30 else 0.5
        
        # Combine scores
        confidence = (complexity_score + spectral_score) / 2.0
        
        return max(0.1, min(1.0, confidence))
    
    def _assess_session_quality(self, fatigue_level: float, 
                               session_duration: Optional[int]) -> str:
        """
        Assess overall session quality
        
        Args:
            fatigue_level: Current fatigue level
            session_duration: Session duration in minutes
            
        Returns:
            Session quality assessment
        """
        if not session_duration:
            return "unknown"
        
        # Calculate quality based on fatigue progression
        if len(self.fatigue_history) < 3:
            return "insufficient_data"
        
        # Check fatigue progression
        initial_fatigue = np.mean(self.fatigue_history[:3])
        current_fatigue = fatigue_level
        fatigue_increase = current_fatigue - initial_fatigue
        
        # Assess quality based on fatigue progression and duration
        if fatigue_increase < 0.1 and session_duration < 60:
            return "excellent"
        elif fatigue_increase < 0.2 and session_duration < 90:
            return "good"
        elif fatigue_increase < 0.3 and session_duration < 120:
            return "fair"
        else:
            return "poor"
    
    def _generate_fatigue_recommendation(self, fatigue_classification: FatigueLevel,
                                       break_recommendation: BreakType,
                                       session_quality: str) -> str:
        """
        Generate personalized fatigue recommendations
        
        Args:
            fatigue_classification: Fatigue level classification
            break_recommendation: Recommended break type
            session_quality: Session quality assessment
            
        Returns:
            Recommendation string
        """
        recommendations = {
            FatigueLevel.NONE: "You're feeling fresh! This is a great time for focused learning.",
            FatigueLevel.MILD: "Slight fatigue detected. Take a quick micro-break to maintain peak performance.",
            FatigueLevel.MODERATE: "Moderate fatigue building up. A short break will help you recharge.",
            FatigueLevel.SEVERE: "Significant fatigue detected. Take a longer break to prevent burnout.",
            FatigueLevel.CRITICAL: "High fatigue level! It's time for a substantial break or to end your session."
        }
        
        base_recommendation = recommendations.get(fatigue_classification, "Monitor your fatigue level.")
        
        # Add break-specific advice
        if break_recommendation == BreakType.MICRO_BREAK:
            base_recommendation += " Try some deep breathing or gentle stretching."
        elif break_recommendation == BreakType.SHORT_BREAK:
            base_recommendation += " Take a walk, get some fresh air, or do light exercise."
        elif break_recommendation == BreakType.MEDIUM_BREAK:
            base_recommendation += " Consider having a healthy snack or doing a relaxing activity."
        elif break_recommendation == BreakType.LONG_BREAK:
            base_recommendation += " Take time for a meal, nap, or enjoyable activity."
        elif break_recommendation == BreakType.SESSION_END:
            base_recommendation += " Great work today! Consider ending your learning session here."
        
        # Add session quality feedback
        if session_quality == "excellent":
            base_recommendation += " Your session quality is excellent - keep up the great work!"
        elif session_quality == "good":
            base_recommendation += " Your session quality is good - you're managing fatigue well."
        elif session_quality == "fair":
            base_recommendation += " Consider taking more frequent breaks to improve session quality."
        elif session_quality == "poor":
            base_recommendation += " Your session quality could be improved with better fatigue management."
        
        return base_recommendation
    
    def _update_history(self, fatigue_level: float, cognitive_load: float):
        """Update historical data for trend analysis"""
        self.fatigue_history.append(fatigue_level)
        
        # Keep only recent history
        max_history = self.monitoring_window * 2
        if len(self.fatigue_history) > max_history:
            self.fatigue_history = self.fatigue_history[-max_history:]
    
    def _get_default_metrics(self) -> FatigueMetrics:
        """Return default metrics when error occurs"""
        return FatigueMetrics(
            fatigue_level=0.3,
            fatigue_classification=FatigueLevel.MILD,
            theta_alpha_ratio=0.5,
            cognitive_load=0.5,
            break_recommendation=BreakType.MICRO_BREAK,
            break_duration=2,
            confidence=0.1,
            session_quality="unknown",
            recommendation="Unable to analyze fatigue level. Please check EEG data quality."
        )
    
    def get_fatigue_statistics(self) -> Dict[str, float]:
        """
        Get fatigue statistics from historical data
        
        Returns:
            Dictionary with fatigue statistics
        """
        if not self.fatigue_history:
            return {
                'mean_fatigue': 0.0,
                'max_fatigue': 0.0,
                'fatigue_trend': 0.0,
                'fatigue_stability': 0.0
            }
        
        fatigue_array = np.array(self.fatigue_history)
        
        # Calculate trend
        if len(fatigue_array) > 1:
            trend = np.polyfit(range(len(fatigue_array)), fatigue_array, 1)[0]
        else:
            trend = 0.0
        
        return {
            'mean_fatigue': float(np.mean(fatigue_array)),
            'max_fatigue': float(np.max(fatigue_array)),
            'fatigue_trend': float(trend),
            'fatigue_stability': float(1.0 - np.std(fatigue_array))
        }
    
    def start_session(self):
        """Start a new learning session"""
        self.session_start_time = datetime.now()
        self.fatigue_history = []
        self.break_history = []
        logger.info("New learning session started")
    
    def end_session(self):
        """End current learning session"""
        if self.session_start_time:
            session_duration = (datetime.now() - self.session_start_time).total_seconds() / 60
            logger.info(f"Learning session ended. Duration: {session_duration:.1f} minutes")
        
        self.session_start_time = None
        self.fatigue_history = []
        self.break_history = []
    
    def get_session_duration(self) -> Optional[float]:
        """Get current session duration in minutes"""
        if not self.session_start_time:
            return None
        
        return (datetime.now() - self.session_start_time).total_seconds() / 60
