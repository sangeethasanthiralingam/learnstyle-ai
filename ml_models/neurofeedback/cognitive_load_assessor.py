"""
Cognitive Load Assessment Module

This module provides real-time cognitive load assessment and optimization:
- Multi-dimensional cognitive load measurement
- Working memory capacity assessment
- Content difficulty optimization
- Learning efficiency monitoring

Author: LearnStyle AI Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CognitiveLoadType(Enum):
    """Types of cognitive load"""
    INTRINSIC = "intrinsic"      # Content complexity
    EXTRINSIC = "extrinsic"      # Presentation/interface issues
    GERMANE = "germane"          # Learning-relevant processing

class LoadLevel(Enum):
    """Cognitive load level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    OVERLOAD = "overload"

@dataclass
class CognitiveLoadMetrics:
    """Cognitive load measurement metrics"""
    total_load: float
    intrinsic_load: float
    extrinsic_load: float
    germane_load: float
    load_classification: LoadLevel
    working_memory_usage: float
    processing_efficiency: float
    content_difficulty_score: float
    interface_complexity_score: float
    learning_efficiency: float
    confidence: float
    recommendation: str

class CognitiveLoadAssessor:
    """
    Real-time cognitive load assessment and optimization system
    """
    
    def __init__(self, 
                 load_threshold: float = 0.7,
                 assessment_window: int = 20):
        """
        Initialize cognitive load assessor
        
        Args:
            load_threshold: Threshold for high cognitive load
            assessment_window: Window size for load assessment (seconds)
        """
        self.load_threshold = load_threshold
        self.assessment_window = assessment_window
        
        # Load level thresholds
        self.load_thresholds = {
            LoadLevel.VERY_LOW: 0.2,
            LoadLevel.LOW: 0.4,
            LoadLevel.MEDIUM: 0.6,
            LoadLevel.HIGH: 0.8,
            LoadLevel.VERY_HIGH: 0.9,
            LoadLevel.OVERLOAD: 0.95
        }
        
        # Historical data for trend analysis
        self.load_history = []
        self.efficiency_history = []
        
        logger.info("Cognitive Load Assessor initialized")
    
    def assess_cognitive_load(self, eeg_features: Dict[str, float],
                            content_metrics: Optional[Dict] = None,
                            user_behavior: Optional[Dict] = None) -> CognitiveLoadMetrics:
        """
        Assess cognitive load from multiple sources
        
        Args:
            eeg_features: EEG features dictionary
            content_metrics: Content complexity metrics
            user_behavior: User interaction behavior metrics
            
        Returns:
            CognitiveLoadMetrics object with load analysis
        """
        try:
            # Calculate different types of cognitive load
            intrinsic_load = self._assess_intrinsic_load(eeg_features, content_metrics)
            extrinsic_load = self._assess_extrinsic_load(eeg_features, user_behavior)
            germane_load = self._assess_germane_load(eeg_features, content_metrics)
            
            # Calculate total cognitive load
            total_load = self._calculate_total_load(intrinsic_load, extrinsic_load, germane_load)
            
            # Classify load level
            load_classification = self._classify_load_level(total_load)
            
            # Assess working memory usage
            working_memory_usage = self._assess_working_memory_usage(eeg_features)
            
            # Calculate processing efficiency
            processing_efficiency = self._calculate_processing_efficiency(eeg_features, total_load)
            
            # Calculate content difficulty score
            content_difficulty_score = self._calculate_content_difficulty(content_metrics)
            
            # Calculate interface complexity score
            interface_complexity_score = self._calculate_interface_complexity(user_behavior)
            
            # Calculate learning efficiency
            learning_efficiency = self._calculate_learning_efficiency(
                total_load, processing_efficiency, content_difficulty_score
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(eeg_features, content_metrics, user_behavior)
            
            # Generate recommendation
            recommendation = self._generate_load_recommendation(
                load_classification, total_load, learning_efficiency
            )
            
            # Update history
            self._update_history(total_load, learning_efficiency)
            
            return CognitiveLoadMetrics(
                total_load=total_load,
                intrinsic_load=intrinsic_load,
                extrinsic_load=extrinsic_load,
                germane_load=germane_load,
                load_classification=load_classification,
                working_memory_usage=working_memory_usage,
                processing_efficiency=processing_efficiency,
                content_difficulty_score=content_difficulty_score,
                interface_complexity_score=interface_complexity_score,
                learning_efficiency=learning_efficiency,
                confidence=confidence,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error assessing cognitive load: {str(e)}")
            return self._get_default_metrics()
    
    def _assess_intrinsic_load(self, eeg_features: Dict[str, float],
                             content_metrics: Optional[Dict]) -> float:
        """
        Assess intrinsic cognitive load (content complexity)
        
        Args:
            eeg_features: EEG features dictionary
            content_metrics: Content complexity metrics
            
        Returns:
            Intrinsic load level (0-1)
        """
        # Gamma power indicates high-level cognitive processing
        gamma_power = eeg_features.get('gamma_power_raw', 0)
        
        # Spectral centroid indicates overall brain activity
        spectral_centroid = eeg_features.get('spectral_centroid', 0)
        
        # Signal complexity indicates cognitive engagement
        complexity = eeg_features.get('complexity', 0)
        
        # Normalize EEG-based metrics
        gamma_score = min(1.0, gamma_power / 100.0)
        spectral_score = min(1.0, spectral_centroid / 30.0)
        complexity_score = min(1.0, complexity / 2.0)
        
        # Combine EEG metrics
        eeg_intrinsic_load = (gamma_score * 0.4 + spectral_score * 0.3 + complexity_score * 0.3)
        
        # Add content-based metrics if available
        if content_metrics:
            content_complexity = content_metrics.get('complexity_score', 0.5)
            content_difficulty = content_metrics.get('difficulty_level', 0.5)
            content_length = content_metrics.get('content_length', 0.5)
            
            content_intrinsic_load = (content_complexity * 0.4 + 
                                    content_difficulty * 0.4 + 
                                    content_length * 0.2)
            
            # Weighted combination
            intrinsic_load = (eeg_intrinsic_load * 0.6 + content_intrinsic_load * 0.4)
        else:
            intrinsic_load = eeg_intrinsic_load
        
        return max(0, min(1, intrinsic_load))
    
    def _assess_extrinsic_load(self, eeg_features: Dict[str, float],
                             user_behavior: Optional[Dict]) -> float:
        """
        Assess extrinsic cognitive load (interface/presentation issues)
        
        Args:
            eeg_features: EEG features dictionary
            user_behavior: User interaction behavior metrics
            
        Returns:
            Extrinsic load level (0-1)
        """
        # Alpha power indicates relaxed state (lower extrinsic load)
        alpha_power = eeg_features.get('alpha_power_raw', 0)
        
        # Beta power indicates active processing (higher extrinsic load)
        beta_power = eeg_features.get('beta_power_raw', 0)
        
        # Calculate alpha/beta ratio (higher ratio = lower extrinsic load)
        if beta_power > 0:
            alpha_beta_ratio = alpha_power / beta_power
        else:
            alpha_beta_ratio = 1.0
        
        # Normalize ratio (inverse relationship with extrinsic load)
        eeg_extrinsic_load = max(0, 1 - (alpha_beta_ratio / 2.0))
        
        # Add behavior-based metrics if available
        if user_behavior:
            click_frequency = user_behavior.get('click_frequency', 0)
            scroll_frequency = user_behavior.get('scroll_frequency', 0)
            error_rate = user_behavior.get('error_rate', 0)
            time_on_task = user_behavior.get('time_on_task', 0)
            
            # Higher interaction frequency and error rate indicate higher extrinsic load
            behavior_extrinsic_load = (
                min(1.0, click_frequency / 10.0) * 0.3 +
                min(1.0, scroll_frequency / 20.0) * 0.2 +
                min(1.0, error_rate) * 0.3 +
                min(1.0, time_on_task / 300.0) * 0.2  # 5 minutes baseline
            )
            
            # Weighted combination
            extrinsic_load = (eeg_extrinsic_load * 0.4 + behavior_extrinsic_load * 0.6)
        else:
            extrinsic_load = eeg_extrinsic_load
        
        return max(0, min(1, extrinsic_load))
    
    def _assess_germane_load(self, eeg_features: Dict[str, float],
                           content_metrics: Optional[Dict]) -> float:
        """
        Assess germane cognitive load (learning-relevant processing)
        
        Args:
            eeg_features: EEG features dictionary
            content_metrics: Content metrics
            
        Returns:
            Germane load level (0-1)
        """
        # Theta power indicates learning and memory formation
        theta_power = eeg_features.get('theta_power_raw', 0)
        
        # Alpha power indicates relaxed awareness (good for learning)
        alpha_power = eeg_features.get('alpha_power_raw', 0)
        
        # Calculate theta/alpha ratio (optimal for learning)
        if alpha_power > 0:
            theta_alpha_ratio = theta_power / alpha_power
        else:
            theta_alpha_ratio = 0.5
        
        # Optimal theta/alpha ratio for learning is around 0.5-1.0
        optimal_ratio = 0.75
        ratio_score = 1.0 - abs(theta_alpha_ratio - optimal_ratio) / optimal_ratio
        
        # Signal complexity indicates active learning
        complexity = eeg_features.get('complexity', 0)
        complexity_score = min(1.0, complexity / 2.0)
        
        # Combine metrics
        germane_load = (ratio_score * 0.6 + complexity_score * 0.4)
        
        return max(0, min(1, germane_load))
    
    def _calculate_total_load(self, intrinsic_load: float, 
                            extrinsic_load: float, germane_load: float) -> float:
        """
        Calculate total cognitive load
        
        Args:
            intrinsic_load: Intrinsic load level
            extrinsic_load: Extrinsic load level
            germane_load: Germane load level
            
        Returns:
            Total cognitive load (0-1)
        """
        # Total load is primarily intrinsic + extrinsic
        # Germane load is beneficial, so we subtract it
        total_load = intrinsic_load + extrinsic_load - (germane_load * 0.3)
        
        return max(0, min(1, total_load))
    
    def _classify_load_level(self, total_load: float) -> LoadLevel:
        """
        Classify cognitive load level
        
        Args:
            total_load: Total cognitive load (0-1)
            
        Returns:
            LoadLevel enum value
        """
        if total_load >= self.load_thresholds[LoadLevel.OVERLOAD]:
            return LoadLevel.OVERLOAD
        elif total_load >= self.load_thresholds[LoadLevel.VERY_HIGH]:
            return LoadLevel.VERY_HIGH
        elif total_load >= self.load_thresholds[LoadLevel.HIGH]:
            return LoadLevel.HIGH
        elif total_load >= self.load_thresholds[LoadLevel.MEDIUM]:
            return LoadLevel.MEDIUM
        elif total_load >= self.load_thresholds[LoadLevel.LOW]:
            return LoadLevel.LOW
        else:
            return LoadLevel.VERY_LOW
    
    def _assess_working_memory_usage(self, eeg_features: Dict[str, float]) -> float:
        """
        Assess working memory usage
        
        Args:
            eeg_features: EEG features dictionary
            
        Returns:
            Working memory usage level (0-1)
        """
        # Gamma power indicates high-level cognitive processing
        gamma_power = eeg_features.get('gamma_power_raw', 0)
        
        # Beta power indicates active processing
        beta_power = eeg_features.get('beta_power_raw', 0)
        
        # Working memory usage is related to gamma and beta activity
        total_high_freq_power = gamma_power + beta_power
        working_memory_usage = min(1.0, total_high_freq_power / 200.0)
        
        return working_memory_usage
    
    def _calculate_processing_efficiency(self, eeg_features: Dict[str, float],
                                       total_load: float) -> float:
        """
        Calculate cognitive processing efficiency
        
        Args:
            eeg_features: EEG features dictionary
            total_load: Total cognitive load
            
        Returns:
            Processing efficiency (0-1)
        """
        # Signal complexity indicates efficient processing
        complexity = eeg_features.get('complexity', 0)
        complexity_score = min(1.0, complexity / 2.0)
        
        # Lower load with higher complexity indicates efficiency
        efficiency = complexity_score * (1 - total_load)
        
        return max(0, min(1, efficiency))
    
    def _calculate_content_difficulty(self, content_metrics: Optional[Dict]) -> float:
        """
        Calculate content difficulty score
        
        Args:
            content_metrics: Content metrics dictionary
            
        Returns:
            Content difficulty score (0-1)
        """
        if not content_metrics:
            return 0.5  # Default moderate difficulty
        
        complexity = content_metrics.get('complexity_score', 0.5)
        difficulty = content_metrics.get('difficulty_level', 0.5)
        length = content_metrics.get('content_length', 0.5)
        
        # Weighted combination
        difficulty_score = (complexity * 0.4 + difficulty * 0.4 + length * 0.2)
        
        return max(0, min(1, difficulty_score))
    
    def _calculate_interface_complexity(self, user_behavior: Optional[Dict]) -> float:
        """
        Calculate interface complexity score
        
        Args:
            user_behavior: User behavior metrics
            
        Returns:
            Interface complexity score (0-1)
        """
        if not user_behavior:
            return 0.3  # Default low complexity
        
        click_frequency = user_behavior.get('click_frequency', 0)
        scroll_frequency = user_behavior.get('scroll_frequency', 0)
        error_rate = user_behavior.get('error_rate', 0)
        
        # Higher interaction frequency and error rate indicate complexity
        complexity_score = (
            min(1.0, click_frequency / 10.0) * 0.3 +
            min(1.0, scroll_frequency / 20.0) * 0.2 +
            min(1.0, error_rate) * 0.5
        )
        
        return max(0, min(1, complexity_score))
    
    def _calculate_learning_efficiency(self, total_load: float,
                                     processing_efficiency: float,
                                     content_difficulty: float) -> float:
        """
        Calculate learning efficiency
        
        Args:
            total_load: Total cognitive load
            processing_efficiency: Processing efficiency
            content_difficulty: Content difficulty score
            
        Returns:
            Learning efficiency (0-1)
        """
        # Learning efficiency is optimal when:
        # - Load is moderate (not too high, not too low)
        # - Processing efficiency is high
        # - Content difficulty matches user capacity
        
        # Optimal load range is 0.4-0.7
        optimal_load_score = 1.0 - abs(total_load - 0.55) / 0.55
        
        # Combine metrics
        learning_efficiency = (
            optimal_load_score * 0.4 +
            processing_efficiency * 0.4 +
            (1 - abs(content_difficulty - 0.6)) * 0.2  # Optimal difficulty around 0.6
        )
        
        return max(0, min(1, learning_efficiency))
    
    def _calculate_confidence(self, eeg_features: Dict[str, float],
                            content_metrics: Optional[Dict],
                            user_behavior: Optional[Dict]) -> float:
        """
        Calculate confidence in cognitive load assessment
        
        Args:
            eeg_features: EEG features dictionary
            content_metrics: Content metrics
            user_behavior: User behavior metrics
            
        Returns:
            Confidence level (0-1)
        """
        # Base confidence from signal quality
        complexity = eeg_features.get('complexity', 0)
        spectral_centroid = eeg_features.get('spectral_centroid', 0)
        
        signal_quality = min(1.0, complexity / 2.0)
        spectral_quality = 1.0 if 5 <= spectral_centroid <= 30 else 0.5
        
        base_confidence = (signal_quality + spectral_quality) / 2.0
        
        # Increase confidence with more data sources
        data_sources = 1  # EEG always available
        if content_metrics:
            data_sources += 1
        if user_behavior:
            data_sources += 1
        
        # Confidence increases with more data sources
        confidence_multiplier = min(1.0, data_sources / 3.0)
        
        return max(0.1, min(1.0, base_confidence * confidence_multiplier))
    
    def _generate_load_recommendation(self, load_classification: LoadLevel,
                                    total_load: float,
                                    learning_efficiency: float) -> str:
        """
        Generate cognitive load recommendations
        
        Args:
            load_classification: Load level classification
            total_load: Total cognitive load
            learning_efficiency: Learning efficiency
            
        Returns:
            Recommendation string
        """
        recommendations = {
            LoadLevel.VERY_LOW: "Very low cognitive load. You can handle more challenging content.",
            LoadLevel.LOW: "Low cognitive load. Consider increasing content difficulty or adding more material.",
            LoadLevel.MEDIUM: "Optimal cognitive load. You're in the ideal learning zone.",
            LoadLevel.HIGH: "High cognitive load detected. Consider simplifying content or taking a break.",
            LoadLevel.VERY_HIGH: "Very high cognitive load. Take a break and reduce content complexity.",
            LoadLevel.OVERLOAD: "Cognitive overload! Stop learning and take a substantial break."
        }
        
        base_recommendation = recommendations.get(load_classification, "Monitor your cognitive load.")
        
        # Add efficiency-based advice
        if learning_efficiency > 0.8:
            base_recommendation += " Your learning efficiency is excellent!"
        elif learning_efficiency > 0.6:
            base_recommendation += " Your learning efficiency is good."
        elif learning_efficiency > 0.4:
            base_recommendation += " Consider optimizing your learning approach."
        else:
            base_recommendation += " Your learning efficiency could be improved."
        
        return base_recommendation
    
    def _update_history(self, total_load: float, learning_efficiency: float):
        """Update historical data for trend analysis"""
        self.load_history.append(total_load)
        self.efficiency_history.append(learning_efficiency)
        
        # Keep only recent history
        max_history = self.assessment_window * 2
        if len(self.load_history) > max_history:
            self.load_history = self.load_history[-max_history:]
            self.efficiency_history = self.efficiency_history[-max_history:]
    
    def _get_default_metrics(self) -> CognitiveLoadMetrics:
        """Return default metrics when error occurs"""
        return CognitiveLoadMetrics(
            total_load=0.5,
            intrinsic_load=0.5,
            extrinsic_load=0.3,
            germane_load=0.6,
            load_classification=LoadLevel.MEDIUM,
            working_memory_usage=0.5,
            processing_efficiency=0.5,
            content_difficulty_score=0.5,
            interface_complexity_score=0.3,
            learning_efficiency=0.5,
            confidence=0.1,
            recommendation="Unable to assess cognitive load. Please check data quality."
        )
    
    def get_load_statistics(self) -> Dict[str, float]:
        """
        Get cognitive load statistics from historical data
        
        Returns:
            Dictionary with load statistics
        """
        if not self.load_history:
            return {
                'mean_load': 0.0,
                'max_load': 0.0,
                'load_trend': 0.0,
                'mean_efficiency': 0.0
            }
        
        load_array = np.array(self.load_history)
        efficiency_array = np.array(self.efficiency_history)
        
        # Calculate load trend
        if len(load_array) > 1:
            load_trend = np.polyfit(range(len(load_array)), load_array, 1)[0]
        else:
            load_trend = 0.0
        
        return {
            'mean_load': float(np.mean(load_array)),
            'max_load': float(np.max(load_array)),
            'load_trend': float(load_trend),
            'mean_efficiency': float(np.mean(efficiency_array))
        }
    
    def reset_history(self):
        """Reset historical data"""
        self.load_history = []
        self.efficiency_history = []
        logger.info("Cognitive load assessment history reset")
