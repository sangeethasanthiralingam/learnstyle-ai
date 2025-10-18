"""
Focus Detection Module

This module provides real-time focus level detection based on EEG data:
- Alpha/Beta ratio analysis for focus assessment
- Attention state classification
- Focus trend analysis over time
- Learning optimization recommendations

Author: LearnStyle AI Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class FocusLevel(Enum):
    """Focus level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class AttentionState(Enum):
    """Attention state classifications"""
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    DROWSY = "drowsy"
    HYPERACTIVE = "hyperactive"
    OPTIMAL = "optimal"

@dataclass
class FocusMetrics:
    """Focus measurement metrics"""
    focus_level: float
    focus_classification: FocusLevel
    attention_state: AttentionState
    alpha_beta_ratio: float
    theta_alpha_ratio: float
    confidence: float
    trend: str  # 'improving', 'declining', 'stable'
    recommendation: str

class FocusDetector:
    """
    Real-time focus detection and analysis system
    """
    
    def __init__(self, 
                 focus_threshold: float = 0.6,
                 attention_window: int = 10,
                 trend_window: int = 30):
        """
        Initialize focus detector
        
        Args:
            focus_threshold: Minimum focus level to be considered focused
            attention_window: Window size for attention analysis (seconds)
            trend_window: Window size for trend analysis (seconds)
        """
        self.focus_threshold = focus_threshold
        self.attention_window = attention_window
        self.trend_window = trend_window
        
        # Focus level thresholds
        self.focus_thresholds = {
            FocusLevel.VERY_LOW: 0.2,
            FocusLevel.LOW: 0.4,
            FocusLevel.MEDIUM: 0.6,
            FocusLevel.HIGH: 0.8,
            FocusLevel.VERY_HIGH: 0.9
        }
        
        # Historical data for trend analysis
        self.focus_history = []
        self.attention_history = []
        
        logger.info("Focus Detector initialized")
    
    def detect_focus_level(self, eeg_features: Dict[str, float]) -> FocusMetrics:
        """
        Detect focus level from EEG features
        
        Args:
            eeg_features: Dictionary containing EEG band powers and features
            
        Returns:
            FocusMetrics object with focus analysis
        """
        try:
            # Extract relevant features
            alpha_power = eeg_features.get('alpha_power_raw', 0)
            beta_power = eeg_features.get('beta_power_raw', 0)
            theta_power = eeg_features.get('theta_power_raw', 0)
            alpha_beta_ratio = eeg_features.get('alpha_beta_ratio', 0)
            theta_alpha_ratio = eeg_features.get('theta_alpha_ratio', 0)
            
            # Calculate focus level
            focus_level = self._calculate_focus_level(alpha_power, beta_power, theta_power)
            
            # Classify focus level
            focus_classification = self._classify_focus_level(focus_level)
            
            # Determine attention state
            attention_state = self._determine_attention_state(
                alpha_beta_ratio, theta_alpha_ratio, focus_level
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(eeg_features)
            
            # Analyze trend
            trend = self._analyze_focus_trend(focus_level)
            
            # Generate recommendation
            recommendation = self._generate_focus_recommendation(
                focus_classification, attention_state, trend
            )
            
            # Update history
            self._update_history(focus_level, attention_state)
            
            return FocusMetrics(
                focus_level=focus_level,
                focus_classification=focus_classification,
                attention_state=attention_state,
                alpha_beta_ratio=alpha_beta_ratio,
                theta_alpha_ratio=theta_alpha_ratio,
                confidence=confidence,
                trend=trend,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error detecting focus level: {str(e)}")
            return self._get_default_metrics()
    
    def _calculate_focus_level(self, alpha_power: float, 
                             beta_power: float, theta_power: float) -> float:
        """
        Calculate focus level from EEG band powers
        
        Args:
            alpha_power: Alpha band power
            beta_power: Beta band power
            theta_power: Theta band power
            
        Returns:
            Focus level (0-1)
        """
        # Avoid division by zero
        if beta_power <= 0:
            beta_power = 1e-6
        
        # Alpha/Beta ratio is a good indicator of focus
        alpha_beta_ratio = alpha_power / beta_power
        
        # Theta power indicates drowsiness (inverse relationship with focus)
        theta_factor = max(0, 1 - (theta_power / (alpha_power + beta_power + 1e-6)))
        
        # Combine metrics for focus level
        focus_level = (alpha_beta_ratio * 0.7 + theta_factor * 0.3)
        
        # Normalize to 0-1 range
        focus_level = max(0, min(1, focus_level))
        
        return focus_level
    
    def _classify_focus_level(self, focus_level: float) -> FocusLevel:
        """
        Classify focus level into categories
        
        Args:
            focus_level: Calculated focus level (0-1)
            
        Returns:
            FocusLevel enum value
        """
        if focus_level >= self.focus_thresholds[FocusLevel.VERY_HIGH]:
            return FocusLevel.VERY_HIGH
        elif focus_level >= self.focus_thresholds[FocusLevel.HIGH]:
            return FocusLevel.HIGH
        elif focus_level >= self.focus_thresholds[FocusLevel.MEDIUM]:
            return FocusLevel.MEDIUM
        elif focus_level >= self.focus_thresholds[FocusLevel.LOW]:
            return FocusLevel.LOW
        else:
            return FocusLevel.VERY_LOW
    
    def _determine_attention_state(self, alpha_beta_ratio: float, 
                                 theta_alpha_ratio: float, 
                                 focus_level: float) -> AttentionState:
        """
        Determine attention state from EEG features
        
        Args:
            alpha_beta_ratio: Alpha to Beta power ratio
            theta_alpha_ratio: Theta to Alpha power ratio
            focus_level: Calculated focus level
            
        Returns:
            AttentionState enum value
        """
        # High theta/alpha ratio indicates drowsiness
        if theta_alpha_ratio > 0.8:
            return AttentionState.DROWSY
        
        # Very high alpha/beta ratio might indicate hyperactive state
        elif alpha_beta_ratio > 2.0:
            return AttentionState.HYPERACTIVE
        
        # Optimal focus range
        elif 0.6 <= focus_level <= 0.9 and 0.5 <= alpha_beta_ratio <= 1.5:
            return AttentionState.OPTIMAL
        
        # Focused but not optimal
        elif focus_level >= self.focus_threshold:
            return AttentionState.FOCUSED
        
        # Low focus indicates distraction
        else:
            return AttentionState.DISTRACTED
    
    def _calculate_confidence(self, eeg_features: Dict[str, float]) -> float:
        """
        Calculate confidence in focus detection
        
        Args:
            eeg_features: EEG features dictionary
            
        Returns:
            Confidence level (0-1)
        """
        # Check signal quality indicators
        complexity = eeg_features.get('complexity', 0)
        spectral_centroid = eeg_features.get('spectral_centroid', 0)
        
        # Higher complexity and reasonable spectral centroid indicate good signal
        complexity_score = min(1.0, complexity / 2.0)  # Normalize complexity
        spectral_score = 1.0 if 5 <= spectral_centroid <= 30 else 0.5
        
        # Combine scores
        confidence = (complexity_score + spectral_score) / 2.0
        
        return max(0.1, min(1.0, confidence))
    
    def _analyze_focus_trend(self, current_focus: float) -> str:
        """
        Analyze focus trend over time
        
        Args:
            current_focus: Current focus level
            
        Returns:
            Trend description
        """
        if len(self.focus_history) < 3:
            return "insufficient_data"
        
        # Get recent focus levels
        recent_focus = self.focus_history[-self.trend_window:]
        
        if len(recent_focus) < 3:
            return "insufficient_data"
        
        # Calculate trend
        trend_slope = np.polyfit(range(len(recent_focus)), recent_focus, 1)[0]
        
        if trend_slope > 0.05:
            return "improving"
        elif trend_slope < -0.05:
            return "declining"
        else:
            return "stable"
    
    def _generate_focus_recommendation(self, focus_classification: FocusLevel,
                                     attention_state: AttentionState,
                                     trend: str) -> str:
        """
        Generate personalized focus recommendations
        
        Args:
            focus_classification: Focus level classification
            attention_state: Attention state
            trend: Focus trend
            
        Returns:
            Recommendation string
        """
        recommendations = {
            (FocusLevel.VERY_HIGH, AttentionState.OPTIMAL): 
                "Excellent focus! You're in the optimal learning state. Continue with challenging content.",
            
            (FocusLevel.HIGH, AttentionState.OPTIMAL): 
                "Great focus! You're performing well. Consider taking a short break in 15-20 minutes.",
            
            (FocusLevel.MEDIUM, AttentionState.FOCUSED): 
                "Good focus level. You're engaged but may benefit from a brief break soon.",
            
            (FocusLevel.LOW, AttentionState.DISTRACTED): 
                "Focus is declining. Take a 5-minute break and return refreshed.",
            
            (FocusLevel.VERY_LOW, AttentionState.DISTRACTED): 
                "Low focus detected. Take a 15-minute break and consider changing your environment.",
            
            (FocusLevel.MEDIUM, AttentionState.DROWSY): 
                "You appear drowsy. Take a short walk or do some light exercise to re-energize.",
            
            (FocusLevel.HIGH, AttentionState.HYPERACTIVE): 
                "High energy detected. Consider calming activities or breathing exercises.",
        }
        
        # Get base recommendation
        base_recommendation = recommendations.get(
            (focus_classification, attention_state),
            "Monitor your focus level and adjust your learning environment as needed."
        )
        
        # Add trend-based advice
        if trend == "declining":
            base_recommendation += " Your focus is declining - consider taking a break."
        elif trend == "improving":
            base_recommendation += " Your focus is improving - great job!"
        
        return base_recommendation
    
    def _update_history(self, focus_level: float, attention_state: AttentionState):
        """Update historical data for trend analysis"""
        self.focus_history.append(focus_level)
        self.attention_history.append(attention_state)
        
        # Keep only recent history
        max_history = self.trend_window * 2  # Keep more data for better trends
        if len(self.focus_history) > max_history:
            self.focus_history = self.focus_history[-max_history:]
            self.attention_history = self.attention_history[-max_history:]
    
    def _get_default_metrics(self) -> FocusMetrics:
        """Return default metrics when error occurs"""
        return FocusMetrics(
            focus_level=0.5,
            focus_classification=FocusLevel.MEDIUM,
            attention_state=AttentionState.FOCUSED,
            alpha_beta_ratio=1.0,
            theta_alpha_ratio=0.5,
            confidence=0.1,
            trend="insufficient_data",
            recommendation="Unable to analyze focus level. Please check EEG data quality."
        )
    
    def get_focus_statistics(self) -> Dict[str, float]:
        """
        Get focus statistics from historical data
        
        Returns:
            Dictionary with focus statistics
        """
        if not self.focus_history:
            return {
                'mean_focus': 0.0,
                'std_focus': 0.0,
                'max_focus': 0.0,
                'min_focus': 0.0,
                'focus_stability': 0.0
            }
        
        focus_array = np.array(self.focus_history)
        
        return {
            'mean_focus': float(np.mean(focus_array)),
            'std_focus': float(np.std(focus_array)),
            'max_focus': float(np.max(focus_array)),
            'min_focus': float(np.min(focus_array)),
            'focus_stability': float(1.0 - np.std(focus_array))  # Higher = more stable
        }
    
    def reset_history(self):
        """Reset historical data"""
        self.focus_history = []
        self.attention_history = []
        logger.info("Focus detection history reset")
