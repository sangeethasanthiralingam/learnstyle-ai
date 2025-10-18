"""
Learning Effectiveness Analysis Module

This module provides comprehensive learning effectiveness measurement including:
- Learning outcome assessment
- Effectiveness metric calculation
- Learning efficiency analysis
- Retention and transfer measurement
- Long-term learning impact evaluation

Author: LearnStyle AI Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class EffectivenessMetric(Enum):
    """Types of learning effectiveness metrics"""
    KNOWLEDGE_RETENTION = "knowledge_retention"
    SKILL_ACQUISITION = "skill_acquisition"
    TRANSFER_ABILITY = "transfer_ability"
    ENGAGEMENT_LEVEL = "engagement_level"
    LEARNING_EFFICIENCY = "learning_efficiency"
    SATISFACTION_SCORE = "satisfaction_score"

class LearningOutcome(Enum):
    """Types of learning outcomes"""
    KNOWLEDGE_GAIN = "knowledge_gain"
    SKILL_DEVELOPMENT = "skill_development"
    BEHAVIOR_CHANGE = "behavior_change"
    ATTITUDE_SHIFT = "attitude_shift"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"

@dataclass
class LearningEffectivenessMetrics:
    """Comprehensive learning effectiveness metrics"""
    overall_effectiveness: float
    knowledge_retention_rate: float
    skill_acquisition_rate: float
    transfer_effectiveness: float
    engagement_effectiveness: float
    learning_efficiency: float
    satisfaction_effectiveness: float
    learning_acceleration: float
    retention_decay_rate: float
    transfer_success_rate: float
    confidence_interval: Tuple[float, float]
    statistical_significance: float
    effect_size: float
    recommendations: List[str]

@dataclass
class LearningOutcomeData:
    """Learning outcome measurement data"""
    user_id: str
    learning_session_id: str
    pre_test_score: float
    post_test_score: float
    retention_test_score: float
    transfer_test_score: float
    engagement_score: float
    satisfaction_score: float
    learning_time: float
    completion_rate: float
    timestamp: datetime

class LearningEffectivenessAnalyzer:
    """
    Advanced learning effectiveness analysis system
    """
    
    def __init__(self, 
                 significance_level: float = 0.05,
                 effect_size_threshold: float = 0.2,
                 retention_period: int = 30):
        """
        Initialize learning effectiveness analyzer
        
        Args:
            significance_level: Statistical significance level
            effect_size_threshold: Minimum effect size threshold
            retention_period: Days for retention measurement
        """
        self.significance_level = significance_level
        self.effect_size_threshold = effect_size_threshold
        self.retention_period = retention_period
        
        # Historical data for trend analysis
        self.learning_outcomes = []
        self.effectiveness_history = []
        
        logger.info("Learning Effectiveness Analyzer initialized")
    
    def analyze_learning_effectiveness(self, learning_data: List[LearningOutcomeData],
                                     baseline_data: Optional[List[LearningOutcomeData]] = None) -> LearningEffectivenessMetrics:
        """
        Analyze learning effectiveness from outcome data
        
        Args:
            learning_data: Learning outcome data
            baseline_data: Optional baseline data for comparison
            
        Returns:
            LearningEffectivenessMetrics object with comprehensive analysis
        """
        try:
            if not learning_data:
                return self._get_default_metrics()
            
            # Calculate individual effectiveness metrics
            knowledge_retention_rate = self._calculate_knowledge_retention_rate(learning_data)
            skill_acquisition_rate = self._calculate_skill_acquisition_rate(learning_data)
            transfer_effectiveness = self._calculate_transfer_effectiveness(learning_data)
            engagement_effectiveness = self._calculate_engagement_effectiveness(learning_data)
            learning_efficiency = self._calculate_learning_efficiency(learning_data)
            satisfaction_effectiveness = self._calculate_satisfaction_effectiveness(learning_data)
            
            # Calculate overall effectiveness
            overall_effectiveness = self._calculate_overall_effectiveness(
                knowledge_retention_rate, skill_acquisition_rate, transfer_effectiveness,
                engagement_effectiveness, learning_efficiency, satisfaction_effectiveness
            )
            
            # Calculate additional metrics
            learning_acceleration = self._calculate_learning_acceleration(learning_data)
            retention_decay_rate = self._calculate_retention_decay_rate(learning_data)
            transfer_success_rate = self._calculate_transfer_success_rate(learning_data)
            
            # Statistical analysis
            confidence_interval = self._calculate_confidence_interval(learning_data)
            statistical_significance = self._calculate_statistical_significance(learning_data, baseline_data)
            effect_size = self._calculate_effect_size(learning_data, baseline_data)
            
            # Generate recommendations
            recommendations = self._generate_effectiveness_recommendations(
                overall_effectiveness, knowledge_retention_rate, skill_acquisition_rate,
                transfer_effectiveness, engagement_effectiveness, learning_efficiency
            )
            
            # Create effectiveness entry
            effectiveness_entry = {
                'timestamp': datetime.now().isoformat(),
                'overall_effectiveness': overall_effectiveness,
                'knowledge_retention_rate': knowledge_retention_rate,
                'skill_acquisition_rate': skill_acquisition_rate,
                'transfer_effectiveness': transfer_effectiveness,
                'engagement_effectiveness': engagement_effectiveness,
                'learning_efficiency': learning_efficiency,
                'statistical_significance': statistical_significance,
                'effect_size': effect_size
            }
            
            # Update history
            self._update_effectiveness_history(effectiveness_entry)
            
            return LearningEffectivenessMetrics(
                overall_effectiveness=overall_effectiveness,
                knowledge_retention_rate=knowledge_retention_rate,
                skill_acquisition_rate=skill_acquisition_rate,
                transfer_effectiveness=transfer_effectiveness,
                engagement_effectiveness=engagement_effectiveness,
                learning_efficiency=learning_efficiency,
                satisfaction_effectiveness=satisfaction_effectiveness,
                learning_acceleration=learning_acceleration,
                retention_decay_rate=retention_decay_rate,
                transfer_success_rate=transfer_success_rate,
                confidence_interval=confidence_interval,
                statistical_significance=statistical_significance,
                effect_size=effect_size,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error analyzing learning effectiveness: {str(e)}")
            return self._get_default_metrics()
    
    def _calculate_knowledge_retention_rate(self, learning_data: List[LearningOutcomeData]) -> float:
        """Calculate knowledge retention rate"""
        try:
            if not learning_data:
                return 0.0
            
            retention_scores = []
            for data in learning_data:
                if data.pre_test_score > 0 and data.retention_test_score >= 0:
                    retention_rate = data.retention_test_score / data.pre_test_score
                    retention_scores.append(min(1.0, retention_rate))
            
            return np.mean(retention_scores) if retention_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating knowledge retention rate: {str(e)}")
            return 0.0
    
    def _calculate_skill_acquisition_rate(self, learning_data: List[LearningOutcomeData]) -> float:
        """Calculate skill acquisition rate"""
        try:
            if not learning_data:
                return 0.0
            
            acquisition_rates = []
            for data in learning_data:
                if data.pre_test_score >= 0 and data.post_test_score >= 0:
                    acquisition_rate = (data.post_test_score - data.pre_test_score) / max(1.0, data.pre_test_score)
                    acquisition_rates.append(max(0.0, acquisition_rate))
            
            return np.mean(acquisition_rates) if acquisition_rates else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating skill acquisition rate: {str(e)}")
            return 0.0
    
    def _calculate_transfer_effectiveness(self, learning_data: List[LearningOutcomeData]) -> float:
        """Calculate transfer effectiveness"""
        try:
            if not learning_data:
                return 0.0
            
            transfer_scores = []
            for data in learning_data:
                if data.post_test_score > 0 and data.transfer_test_score >= 0:
                    transfer_rate = data.transfer_test_score / data.post_test_score
                    transfer_scores.append(min(1.0, transfer_rate))
            
            return np.mean(transfer_scores) if transfer_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating transfer effectiveness: {str(e)}")
            return 0.0
    
    def _calculate_engagement_effectiveness(self, learning_data: List[LearningOutcomeData]) -> float:
        """Calculate engagement effectiveness"""
        try:
            if not learning_data:
                return 0.0
            
            engagement_scores = [data.engagement_score for data in learning_data if data.engagement_score >= 0]
            return np.mean(engagement_scores) if engagement_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating engagement effectiveness: {str(e)}")
            return 0.0
    
    def _calculate_learning_efficiency(self, learning_data: List[LearningOutcomeData]) -> float:
        """Calculate learning efficiency"""
        try:
            if not learning_data:
                return 0.0
            
            efficiency_scores = []
            for data in learning_data:
                if data.learning_time > 0 and data.post_test_score >= 0:
                    # Efficiency = learning gain per unit time
                    learning_gain = max(0, data.post_test_score - data.pre_test_score)
                    efficiency = learning_gain / data.learning_time
                    efficiency_scores.append(efficiency)
            
            return np.mean(efficiency_scores) if efficiency_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating learning efficiency: {str(e)}")
            return 0.0
    
    def _calculate_satisfaction_effectiveness(self, learning_data: List[LearningOutcomeData]) -> float:
        """Calculate satisfaction effectiveness"""
        try:
            if not learning_data:
                return 0.0
            
            satisfaction_scores = [data.satisfaction_score for data in learning_data if data.satisfaction_score >= 0]
            return np.mean(satisfaction_scores) if satisfaction_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating satisfaction effectiveness: {str(e)}")
            return 0.0
    
    def _calculate_overall_effectiveness(self, knowledge_retention: float, skill_acquisition: float,
                                       transfer_effectiveness: float, engagement_effectiveness: float,
                                       learning_efficiency: float, satisfaction_effectiveness: float) -> float:
        """Calculate overall learning effectiveness"""
        try:
            # Weighted combination of effectiveness metrics
            weights = {
                'knowledge_retention': 0.25,
                'skill_acquisition': 0.25,
                'transfer_effectiveness': 0.20,
                'engagement_effectiveness': 0.15,
                'learning_efficiency': 0.10,
                'satisfaction_effectiveness': 0.05
            }
            
            overall_effectiveness = (
                knowledge_retention * weights['knowledge_retention'] +
                skill_acquisition * weights['skill_acquisition'] +
                transfer_effectiveness * weights['transfer_effectiveness'] +
                engagement_effectiveness * weights['engagement_effectiveness'] +
                learning_efficiency * weights['learning_efficiency'] +
                satisfaction_effectiveness * weights['satisfaction_effectiveness']
            )
            
            return max(0.0, min(1.0, overall_effectiveness))
            
        except Exception as e:
            logger.error(f"Error calculating overall effectiveness: {str(e)}")
            return 0.0
    
    def _calculate_learning_acceleration(self, learning_data: List[LearningOutcomeData]) -> float:
        """Calculate learning acceleration over time"""
        try:
            if len(learning_data) < 2:
                return 0.0
            
            # Sort by timestamp
            sorted_data = sorted(learning_data, key=lambda x: x.timestamp)
            
            # Calculate learning gains over time
            learning_gains = []
            for i in range(1, len(sorted_data)):
                prev_data = sorted_data[i-1]
                curr_data = sorted_data[i]
                
                if prev_data.pre_test_score >= 0 and curr_data.pre_test_score >= 0:
                    gain = curr_data.post_test_score - prev_data.post_test_score
                    time_diff = (curr_data.timestamp - prev_data.timestamp).total_seconds() / 3600  # hours
                    
                    if time_diff > 0:
                        acceleration = gain / time_diff
                        learning_gains.append(acceleration)
            
            return np.mean(learning_gains) if learning_gains else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating learning acceleration: {str(e)}")
            return 0.0
    
    def _calculate_retention_decay_rate(self, learning_data: List[LearningOutcomeData]) -> float:
        """Calculate knowledge retention decay rate"""
        try:
            if not learning_data:
                return 0.0
            
            decay_rates = []
            for data in learning_data:
                if data.post_test_score > 0 and data.retention_test_score >= 0:
                    # Calculate decay rate
                    decay = (data.post_test_score - data.retention_test_score) / data.post_test_score
                    decay_rates.append(max(0.0, decay))
            
            return np.mean(decay_rates) if decay_rates else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating retention decay rate: {str(e)}")
            return 0.0
    
    def _calculate_transfer_success_rate(self, learning_data: List[LearningOutcomeData]) -> float:
        """Calculate transfer success rate"""
        try:
            if not learning_data:
                return 0.0
            
            transfer_successes = 0
            total_transfer_attempts = 0
            
            for data in learning_data:
                if data.post_test_score > 0 and data.transfer_test_score >= 0:
                    total_transfer_attempts += 1
                    # Consider transfer successful if transfer score is at least 70% of post-test score
                    if data.transfer_test_score >= 0.7 * data.post_test_score:
                        transfer_successes += 1
            
            return transfer_successes / total_transfer_attempts if total_transfer_attempts > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating transfer success rate: {str(e)}")
            return 0.0
    
    def _calculate_confidence_interval(self, learning_data: List[LearningOutcomeData], 
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for overall effectiveness"""
        try:
            if not learning_data:
                return (0.0, 0.0)
            
            # Calculate effectiveness scores for each data point
            effectiveness_scores = []
            for data in learning_data:
                # Simple effectiveness score calculation
                if data.pre_test_score >= 0 and data.post_test_score >= 0:
                    effectiveness = (data.post_test_score - data.pre_test_score) / max(1.0, data.pre_test_score)
                    effectiveness_scores.append(max(0.0, effectiveness))
            
            if not effectiveness_scores:
                return (0.0, 0.0)
            
            # Calculate confidence interval
            mean_score = np.mean(effectiveness_scores)
            std_score = np.std(effectiveness_scores, ddof=1)
            n = len(effectiveness_scores)
            
            # t-distribution critical value
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, n - 1)
            
            margin_error = t_critical * (std_score / np.sqrt(n))
            
            lower_bound = max(0.0, mean_score - margin_error)
            upper_bound = min(1.0, mean_score + margin_error)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {str(e)}")
            return (0.0, 0.0)
    
    def _calculate_statistical_significance(self, learning_data: List[LearningOutcomeData],
                                          baseline_data: Optional[List[LearningOutcomeData]]) -> float:
        """Calculate statistical significance of learning effectiveness"""
        try:
            if not learning_data or not baseline_data:
                return 0.0
            
            # Extract effectiveness scores
            treatment_scores = []
            for data in learning_data:
                if data.pre_test_score >= 0 and data.post_test_score >= 0:
                    effectiveness = (data.post_test_score - data.pre_test_score) / max(1.0, data.pre_test_score)
                    treatment_scores.append(max(0.0, effectiveness))
            
            baseline_scores = []
            for data in baseline_data:
                if data.pre_test_score >= 0 and data.post_test_score >= 0:
                    effectiveness = (data.post_test_score - data.pre_test_score) / max(1.0, data.pre_test_score)
                    baseline_scores.append(max(0.0, effectiveness))
            
            if not treatment_scores or not baseline_scores:
                return 0.0
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(treatment_scores, baseline_scores)
            
            return p_value
            
        except Exception as e:
            logger.error(f"Error calculating statistical significance: {str(e)}")
            return 0.0
    
    def _calculate_effect_size(self, learning_data: List[LearningOutcomeData],
                             baseline_data: Optional[List[LearningOutcomeData]]) -> float:
        """Calculate effect size (Cohen's d)"""
        try:
            if not learning_data or not baseline_data:
                return 0.0
            
            # Extract effectiveness scores
            treatment_scores = []
            for data in learning_data:
                if data.pre_test_score >= 0 and data.post_test_score >= 0:
                    effectiveness = (data.post_test_score - data.pre_test_score) / max(1.0, data.pre_test_score)
                    treatment_scores.append(max(0.0, effectiveness))
            
            baseline_scores = []
            for data in baseline_data:
                if data.pre_test_score >= 0 and data.post_test_score >= 0:
                    effectiveness = (data.post_test_score - data.pre_test_score) / max(1.0, data.pre_test_score)
                    baseline_scores.append(max(0.0, effectiveness))
            
            if not treatment_scores or not baseline_scores:
                return 0.0
            
            # Calculate Cohen's d
            mean_treatment = np.mean(treatment_scores)
            mean_baseline = np.mean(baseline_scores)
            
            # Pooled standard deviation
            n1, n2 = len(treatment_scores), len(baseline_scores)
            var1, var2 = np.var(treatment_scores, ddof=1), np.var(baseline_scores, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            
            if pooled_std == 0:
                return 0.0
            
            cohens_d = (mean_treatment - mean_baseline) / pooled_std
            
            return cohens_d
            
        except Exception as e:
            logger.error(f"Error calculating effect size: {str(e)}")
            return 0.0
    
    def _generate_effectiveness_recommendations(self, overall_effectiveness: float,
                                              knowledge_retention: float, skill_acquisition: float,
                                              transfer_effectiveness: float, engagement_effectiveness: float,
                                              learning_efficiency: float) -> List[str]:
        """Generate recommendations for improving learning effectiveness"""
        try:
            recommendations = []
            
            # Overall effectiveness recommendations
            if overall_effectiveness < 0.5:
                recommendations.append("Implement comprehensive learning effectiveness improvement strategies")
                recommendations.append("Conduct detailed analysis of learning barriers and challenges")
            
            # Knowledge retention recommendations
            if knowledge_retention < 0.6:
                recommendations.append("Implement spaced repetition and retrieval practice techniques")
                recommendations.append("Add regular review sessions and knowledge reinforcement activities")
            
            # Skill acquisition recommendations
            if skill_acquisition < 0.5:
                recommendations.append("Increase hands-on practice and skill-building activities")
                recommendations.append("Provide more immediate feedback and skill assessment")
            
            # Transfer effectiveness recommendations
            if transfer_effectiveness < 0.4:
                recommendations.append("Design learning activities that promote knowledge transfer")
                recommendations.append("Include real-world applications and case studies")
            
            # Engagement recommendations
            if engagement_effectiveness < 0.6:
                recommendations.append("Enhance interactive and engaging learning experiences")
                recommendations.append("Implement gamification and social learning elements")
            
            # Efficiency recommendations
            if learning_efficiency < 0.5:
                recommendations.append("Optimize learning content and delivery methods")
                recommendations.append("Implement adaptive learning paths based on individual progress")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Continue monitoring learning effectiveness metrics")
                recommendations.append("Implement continuous improvement processes")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating effectiveness recommendations: {str(e)}")
            return ["Monitor learning effectiveness and implement improvements as needed"]
    
    def _update_effectiveness_history(self, effectiveness_entry: Dict):
        """Update effectiveness history for trend analysis"""
        self.effectiveness_history.append(effectiveness_entry)
        
        # Keep only recent history
        if len(self.effectiveness_history) > 100:
            self.effectiveness_history = self.effectiveness_history[-100:]
    
    def _get_default_metrics(self) -> LearningEffectivenessMetrics:
        """Return default metrics when analysis fails"""
        return LearningEffectivenessMetrics(
            overall_effectiveness=0.0,
            knowledge_retention_rate=0.0,
            skill_acquisition_rate=0.0,
            transfer_effectiveness=0.0,
            engagement_effectiveness=0.0,
            learning_efficiency=0.0,
            satisfaction_effectiveness=0.0,
            learning_acceleration=0.0,
            retention_decay_rate=0.0,
            transfer_success_rate=0.0,
            confidence_interval=(0.0, 0.0),
            statistical_significance=0.0,
            effect_size=0.0,
            recommendations=["Insufficient data for analysis"]
        )
    
    def get_effectiveness_trends(self) -> Dict[str, float]:
        """Get learning effectiveness trends from historical data"""
        try:
            if len(self.effectiveness_history) < 2:
                return {
                    'trend_direction': 0.0,
                    'trend_strength': 0.0,
                    'volatility': 0.0
                }
            
            # Extract effectiveness scores over time
            effectiveness_scores = [entry['overall_effectiveness'] for entry in self.effectiveness_history]
            
            # Calculate trend direction
            x = np.arange(len(effectiveness_scores))
            trend_slope = np.polyfit(x, effectiveness_scores, 1)[0]
            
            # Calculate trend strength (R-squared)
            y_pred = np.polyval(np.polyfit(x, effectiveness_scores, 1), x)
            ss_res = np.sum((effectiveness_scores - y_pred) ** 2)
            ss_tot = np.sum((effectiveness_scores - np.mean(effectiveness_scores)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate volatility (standard deviation)
            volatility = np.std(effectiveness_scores)
            
            return {
                'trend_direction': float(trend_slope),
                'trend_strength': float(r_squared),
                'volatility': float(volatility)
            }
            
        except Exception as e:
            logger.error(f"Error calculating effectiveness trends: {str(e)}")
            return {
                'trend_direction': 0.0,
                'trend_strength': 0.0,
                'volatility': 0.0
            }
    
    def generate_effectiveness_report(self, learning_data: List[LearningOutcomeData]) -> Dict:
        """Generate comprehensive learning effectiveness report"""
        try:
            # Analyze effectiveness
            metrics = self.analyze_learning_effectiveness(learning_data)
            
            # Get trends
            trends = self.get_effectiveness_trends()
            
            # Generate report
            report = {
                'analysis_timestamp': datetime.now().isoformat(),
                'data_points': len(learning_data),
                'effectiveness_metrics': {
                    'overall_effectiveness': metrics.overall_effectiveness,
                    'knowledge_retention_rate': metrics.knowledge_retention_rate,
                    'skill_acquisition_rate': metrics.skill_acquisition_rate,
                    'transfer_effectiveness': metrics.transfer_effectiveness,
                    'engagement_effectiveness': metrics.engagement_effectiveness,
                    'learning_efficiency': metrics.learning_efficiency,
                    'satisfaction_effectiveness': metrics.satisfaction_effectiveness
                },
                'statistical_analysis': {
                    'confidence_interval': metrics.confidence_interval,
                    'statistical_significance': metrics.statistical_significance,
                    'effect_size': metrics.effect_size
                },
                'trends': trends,
                'recommendations': metrics.recommendations
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating effectiveness report: {str(e)}")
            return {'error': str(e)}
