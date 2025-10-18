"""
Engagement Optimization Module

This module provides real-time engagement optimization including:
- Dynamic content adaptation based on emotions
- Engagement intervention strategies
- Learning path optimization
- Real-time feedback adjustment
- Personalized engagement enhancement

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

class InterventionType(Enum):
    """Types of engagement interventions"""
    CONTENT_ADJUSTMENT = "content_adjustment"
    DIFFICULTY_MODIFICATION = "difficulty_modification"
    PACE_CHANGE = "pace_change"
    INTERACTION_ENHANCEMENT = "interaction_enhancement"
    MOTIVATION_BOOST = "motivation_boost"
    BREAK_RECOMMENDATION = "break_recommendation"
    SUPPORT_ACTIVATION = "support_activation"

class OptimizationStrategy(Enum):
    """Engagement optimization strategies"""
    PROACTIVE = "proactive"
    REACTIVE = "reactive"
    ADAPTIVE = "adaptive"
    PREVENTIVE = "preventive"

class EngagementLevel(Enum):
    """Engagement level classifications"""
    CRITICAL = "critical"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    OPTIMAL = "optimal"

@dataclass
class EngagementIntervention:
    """Individual engagement intervention"""
    type: InterventionType
    priority: int
    confidence: float
    expected_improvement: float
    description: str
    parameters: Dict[str, any]
    implementation_time: float  # seconds

@dataclass
class EngagementOptimizationResult:
    """Complete engagement optimization result"""
    interventions: List[EngagementIntervention]
    overall_strategy: OptimizationStrategy
    expected_engagement_improvement: float
    implementation_priority: List[InterventionType]
    personalized_recommendations: List[str]
    monitoring_metrics: Dict[str, float]
    confidence: float

class EngagementOptimizer:
    """
    Advanced engagement optimization system
    """
    
    def __init__(self, 
                 intervention_threshold: float = 0.3,
                 optimization_window: int = 20,
                 max_interventions: int = 5):
        """
        Initialize engagement optimizer
        
        Args:
            intervention_threshold: Threshold for triggering interventions
            optimization_window: Window size for optimization analysis
            max_interventions: Maximum number of interventions per optimization
        """
        self.intervention_threshold = intervention_threshold
        self.optimization_window = optimization_window
        self.max_interventions = max_interventions
        
        # Intervention effectiveness tracking
        self.intervention_history = []
        self.effectiveness_metrics = {}
        
        # Engagement level thresholds
        self.engagement_thresholds = {
            EngagementLevel.CRITICAL: 0.2,
            EngagementLevel.LOW: 0.4,
            EngagementLevel.MEDIUM: 0.6,
            EngagementLevel.HIGH: 0.8,
            EngagementLevel.OPTIMAL: 0.9
        }
        
        logger.info("Engagement Optimizer initialized")
    
    def optimize_engagement(self, emotion_data: Dict, attention_data: Dict, 
                          learning_context: Dict) -> EngagementOptimizationResult:
        """
        Optimize engagement based on emotion and attention data
        
        Args:
            emotion_data: Fused emotion analysis data
            attention_data: Attention and engagement data
            learning_context: Current learning context
            
        Returns:
            EngagementOptimizationResult with optimization recommendations
        """
        try:
            # Analyze current engagement state
            current_engagement = self._analyze_current_engagement(emotion_data, attention_data)
            
            # Determine optimization strategy
            strategy = self._determine_optimization_strategy(current_engagement, learning_context)
            
            # Generate interventions
            interventions = self._generate_interventions(
                emotion_data, attention_data, learning_context, current_engagement, strategy
            )
            
            # Prioritize interventions
            prioritized_interventions = self._prioritize_interventions(interventions)
            
            # Calculate expected improvement
            expected_improvement = self._calculate_expected_improvement(prioritized_interventions)
            
            # Generate personalized recommendations
            personalized_recommendations = self._generate_personalized_recommendations(
                emotion_data, attention_data, learning_context, prioritized_interventions
            )
            
            # Define monitoring metrics
            monitoring_metrics = self._define_monitoring_metrics(prioritized_interventions)
            
            # Calculate overall confidence
            confidence = self._calculate_optimization_confidence(
                emotion_data, attention_data, prioritized_interventions
            )
            
            return EngagementOptimizationResult(
                interventions=prioritized_interventions[:self.max_interventions],
                overall_strategy=strategy,
                expected_engagement_improvement=expected_improvement,
                implementation_priority=[interv.type for interv in prioritized_interventions],
                personalized_recommendations=personalized_recommendations,
                monitoring_metrics=monitoring_metrics,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error optimizing engagement: {str(e)}")
            return self._get_default_result()
    
    def _analyze_current_engagement(self, emotion_data: Dict, attention_data: Dict) -> Dict:
        """Analyze current engagement state"""
        try:
            # Extract engagement metrics
            emotional_engagement = emotion_data.get('emotional_engagement', 0.5)
            attention_score = attention_data.get('attention_score', 0.5)
            engagement_score = attention_data.get('engagement_score', 0.5)
            learning_readiness = emotion_data.get('learning_readiness', 0.5)
            
            # Calculate overall engagement
            overall_engagement = (
                emotional_engagement * 0.3 +
                attention_score * 0.3 +
                engagement_score * 0.3 +
                learning_readiness * 0.1
            )
            
            # Classify engagement level
            if overall_engagement >= self.engagement_thresholds[EngagementLevel.OPTIMAL]:
                engagement_level = EngagementLevel.OPTIMAL
            elif overall_engagement >= self.engagement_thresholds[EngagementLevel.HIGH]:
                engagement_level = EngagementLevel.HIGH
            elif overall_engagement >= self.engagement_thresholds[EngagementLevel.MEDIUM]:
                engagement_level = EngagementLevel.MEDIUM
            elif overall_engagement >= self.engagement_thresholds[EngagementLevel.LOW]:
                engagement_level = EngagementLevel.LOW
            else:
                engagement_level = EngagementLevel.CRITICAL
            
            # Identify engagement issues
            issues = []
            if emotional_engagement < 0.4:
                issues.append('low_emotional_engagement')
            if attention_score < 0.4:
                issues.append('low_attention')
            if engagement_score < 0.4:
                issues.append('low_engagement')
            if learning_readiness < 0.4:
                issues.append('low_learning_readiness')
            
            return {
                'overall_engagement': overall_engagement,
                'engagement_level': engagement_level,
                'emotional_engagement': emotional_engagement,
                'attention_score': attention_score,
                'engagement_score': engagement_score,
                'learning_readiness': learning_readiness,
                'issues': issues
            }
            
        except Exception as e:
            logger.error(f"Error analyzing current engagement: {str(e)}")
            return {
                'overall_engagement': 0.5,
                'engagement_level': EngagementLevel.MEDIUM,
                'emotional_engagement': 0.5,
                'attention_score': 0.5,
                'engagement_score': 0.5,
                'learning_readiness': 0.5,
                'issues': []
            }
    
    def _determine_optimization_strategy(self, current_engagement: Dict, 
                                       learning_context: Dict) -> OptimizationStrategy:
        """Determine optimization strategy based on current state"""
        try:
            engagement_level = current_engagement['engagement_level']
            issues = current_engagement['issues']
            
            # Critical engagement requires proactive intervention
            if engagement_level == EngagementLevel.CRITICAL:
                return OptimizationStrategy.PROACTIVE
            
            # Multiple issues require adaptive approach
            if len(issues) > 2:
                return OptimizationStrategy.ADAPTIVE
            
            # Single issue can be handled reactively
            if len(issues) == 1:
                return OptimizationStrategy.REACTIVE
            
            # High engagement requires preventive maintenance
            if engagement_level in [EngagementLevel.HIGH, EngagementLevel.OPTIMAL]:
                return OptimizationStrategy.PREVENTIVE
            
            # Default to adaptive
            return OptimizationStrategy.ADAPTIVE
            
        except Exception as e:
            logger.error(f"Error determining optimization strategy: {str(e)}")
            return OptimizationStrategy.ADAPTIVE
    
    def _generate_interventions(self, emotion_data: Dict, attention_data: Dict, 
                              learning_context: Dict, current_engagement: Dict, 
                              strategy: OptimizationStrategy) -> List[EngagementIntervention]:
        """Generate engagement interventions based on current state"""
        try:
            interventions = []
            issues = current_engagement['issues']
            engagement_level = current_engagement['engagement_level']
            
            # Content adjustment interventions
            if 'low_emotional_engagement' in issues or 'low_engagement' in issues:
                content_intervention = self._create_content_adjustment_intervention(
                    emotion_data, learning_context
                )
                if content_intervention:
                    interventions.append(content_intervention)
            
            # Difficulty modification interventions
            if 'low_learning_readiness' in issues or engagement_level == EngagementLevel.CRITICAL:
                difficulty_intervention = self._create_difficulty_modification_intervention(
                    attention_data, learning_context
                )
                if difficulty_intervention:
                    interventions.append(difficulty_intervention)
            
            # Pace change interventions
            if 'low_attention' in issues or 'low_engagement' in issues:
                pace_intervention = self._create_pace_change_intervention(
                    attention_data, learning_context
                )
                if pace_intervention:
                    interventions.append(pace_intervention)
            
            # Interaction enhancement interventions
            if 'low_emotional_engagement' in issues:
                interaction_intervention = self._create_interaction_enhancement_intervention(
                    emotion_data, learning_context
                )
                if interaction_intervention:
                    interventions.append(interaction_intervention)
            
            # Motivation boost interventions
            if engagement_level in [EngagementLevel.CRITICAL, EngagementLevel.LOW]:
                motivation_intervention = self._create_motivation_boost_intervention(
                    emotion_data, learning_context
                )
                if motivation_intervention:
                    interventions.append(motivation_intervention)
            
            # Break recommendation interventions
            if 'low_attention' in issues and attention_data.get('fatigue_score', 0) > 0.7:
                break_intervention = self._create_break_recommendation_intervention(
                    attention_data, learning_context
                )
                if break_intervention:
                    interventions.append(break_intervention)
            
            # Support activation interventions
            if engagement_level == EngagementLevel.CRITICAL:
                support_intervention = self._create_support_activation_intervention(
                    emotion_data, attention_data, learning_context
                )
                if support_intervention:
                    interventions.append(support_intervention)
            
            return interventions
            
        except Exception as e:
            logger.error(f"Error generating interventions: {str(e)}")
            return []
    
    def _create_content_adjustment_intervention(self, emotion_data: Dict, 
                                              learning_context: Dict) -> Optional[EngagementIntervention]:
        """Create content adjustment intervention"""
        try:
            emotion_state = emotion_data.get('overall_emotion_state', 'neutral')
            learning_emotion_state = emotion_data.get('learning_emotion_state', 'motivated')
            
            # Determine content adjustments based on emotions
            adjustments = []
            
            if emotion_state in ['negative', 'very_negative']:
                adjustments.append('increase_positive_content')
                adjustments.append('add_encouragement_messages')
            
            if learning_emotion_state in ['confused', 'overwhelmed']:
                adjustments.append('simplify_explanations')
                adjustments.append('add_step_by_step_guidance')
            
            if learning_emotion_state == 'bored':
                adjustments.append('add_interactive_elements')
                adjustments.append('increase_content_variety')
            
            if not adjustments:
                return None
            
            return EngagementIntervention(
                type=InterventionType.CONTENT_ADJUSTMENT,
                priority=3,
                confidence=0.8,
                expected_improvement=0.2,
                description=f"Adjust content based on emotional state: {', '.join(adjustments)}",
                parameters={'adjustments': adjustments, 'emotion_state': emotion_state},
                implementation_time=5.0
            )
            
        except Exception as e:
            logger.error(f"Error creating content adjustment intervention: {str(e)}")
            return None
    
    def _create_difficulty_modification_intervention(self, attention_data: Dict, 
                                                   learning_context: Dict) -> Optional[EngagementIntervention]:
        """Create difficulty modification intervention"""
        try:
            attention_score = attention_data.get('attention_score', 0.5)
            cognitive_load = attention_data.get('cognitive_load', 0.5)
            
            # Determine difficulty adjustments
            if attention_score < 0.4 or cognitive_load > 0.8:
                difficulty_change = 'decrease'
                expected_improvement = 0.25
            elif attention_score > 0.8 and cognitive_load < 0.3:
                difficulty_change = 'increase'
                expected_improvement = 0.15
            else:
                return None
            
            return EngagementIntervention(
                type=InterventionType.DIFFICULTY_MODIFICATION,
                priority=4,
                confidence=0.7,
                expected_improvement=expected_improvement,
                description=f"Modify difficulty level: {difficulty_change} complexity",
                parameters={'difficulty_change': difficulty_change, 'attention_score': attention_score},
                implementation_time=3.0
            )
            
        except Exception as e:
            logger.error(f"Error creating difficulty modification intervention: {str(e)}")
            return None
    
    def _create_pace_change_intervention(self, attention_data: Dict, 
                                       learning_context: Dict) -> Optional[EngagementIntervention]:
        """Create pace change intervention"""
        try:
            attention_score = attention_data.get('attention_score', 0.5)
            engagement_score = attention_data.get('engagement_score', 0.5)
            
            # Determine pace adjustments
            if attention_score < 0.4 or engagement_score < 0.4:
                pace_change = 'slow_down'
                expected_improvement = 0.2
            elif attention_score > 0.8 and engagement_score > 0.8:
                pace_change = 'speed_up'
                expected_improvement = 0.1
            else:
                return None
            
            return EngagementIntervention(
                type=InterventionType.PACE_CHANGE,
                priority=3,
                confidence=0.6,
                expected_improvement=expected_improvement,
                description=f"Adjust learning pace: {pace_change}",
                parameters={'pace_change': pace_change, 'attention_score': attention_score},
                implementation_time=2.0
            )
            
        except Exception as e:
            logger.error(f"Error creating pace change intervention: {str(e)}")
            return None
    
    def _create_interaction_enhancement_intervention(self, emotion_data: Dict, 
                                                   learning_context: Dict) -> Optional[EngagementIntervention]:
        """Create interaction enhancement intervention"""
        try:
            emotional_engagement = emotion_data.get('emotional_engagement', 0.5)
            
            if emotional_engagement < 0.5:
                interaction_types = ['add_quizzes', 'enable_chat', 'add_polls', 'create_discussions']
                
                return EngagementIntervention(
                    type=InterventionType.INTERACTION_ENHANCEMENT,
                    priority=2,
                    confidence=0.7,
                    expected_improvement=0.15,
                    description="Enhance interactions to boost emotional engagement",
                    parameters={'interaction_types': interaction_types, 'emotional_engagement': emotional_engagement},
                    implementation_time=10.0
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating interaction enhancement intervention: {str(e)}")
            return None
    
    def _create_motivation_boost_intervention(self, emotion_data: Dict, 
                                            learning_context: Dict) -> Optional[EngagementIntervention]:
        """Create motivation boost intervention"""
        try:
            learning_emotion_state = emotion_data.get('learning_emotion_state', 'motivated')
            
            if learning_emotion_state in ['bored', 'frustrated', 'overwhelmed']:
                motivation_strategies = []
                
                if learning_emotion_state == 'bored':
                    motivation_strategies.extend(['add_gamification', 'create_challenges', 'show_progress'])
                elif learning_emotion_state == 'frustrated':
                    motivation_strategies.extend(['provide_encouragement', 'offer_help', 'break_down_tasks'])
                elif learning_emotion_state == 'overwhelmed':
                    motivation_strategies.extend(['simplify_goals', 'provide_guidance', 'offer_support'])
                
                return EngagementIntervention(
                    type=InterventionType.MOTIVATION_BOOST,
                    priority=5,
                    confidence=0.8,
                    expected_improvement=0.3,
                    description=f"Boost motivation using: {', '.join(motivation_strategies)}",
                    parameters={'strategies': motivation_strategies, 'emotion_state': learning_emotion_state},
                    implementation_time=15.0
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating motivation boost intervention: {str(e)}")
            return None
    
    def _create_break_recommendation_intervention(self, attention_data: Dict, 
                                                learning_context: Dict) -> Optional[EngagementIntervention]:
        """Create break recommendation intervention"""
        try:
            fatigue_score = attention_data.get('fatigue_score', 0.5)
            attention_score = attention_data.get('attention_score', 0.5)
            
            if fatigue_score > 0.7 or attention_score < 0.3:
                break_duration = min(15, int(fatigue_score * 20))  # 5-15 minutes
                
                return EngagementIntervention(
                    type=InterventionType.BREAK_RECOMMENDATION,
                    priority=6,
                    confidence=0.9,
                    expected_improvement=0.4,
                    description=f"Recommend {break_duration}-minute break to restore focus",
                    parameters={'break_duration': break_duration, 'fatigue_score': fatigue_score},
                    implementation_time=1.0
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating break recommendation intervention: {str(e)}")
            return None
    
    def _create_support_activation_intervention(self, emotion_data: Dict, attention_data: Dict, 
                                              learning_context: Dict) -> Optional[EngagementIntervention]:
        """Create support activation intervention"""
        try:
            engagement_level = attention_data.get('engagement_level', 'medium')
            
            if engagement_level in ['critical', 'low']:
                support_types = ['human_tutor', 'peer_support', 'ai_assistant', 'resource_links']
                
                return EngagementIntervention(
                    type=InterventionType.SUPPORT_ACTIVATION,
                    priority=7,
                    confidence=0.9,
                    expected_improvement=0.5,
                    description="Activate support systems to help struggling learner",
                    parameters={'support_types': support_types, 'engagement_level': engagement_level},
                    implementation_time=30.0
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating support activation intervention: {str(e)}")
            return None
    
    def _prioritize_interventions(self, interventions: List[EngagementIntervention]) -> List[EngagementIntervention]:
        """Prioritize interventions based on priority and expected improvement"""
        try:
            # Sort by priority (higher number = higher priority) and expected improvement
            prioritized = sorted(
                interventions,
                key=lambda x: (x.priority, x.expected_improvement),
                reverse=True
            )
            
            return prioritized
            
        except Exception as e:
            logger.error(f"Error prioritizing interventions: {str(e)}")
            return interventions
    
    def _calculate_expected_improvement(self, interventions: List[EngagementIntervention]) -> float:
        """Calculate expected overall improvement from interventions"""
        try:
            if not interventions:
                return 0.0
            
            # Calculate weighted improvement based on confidence
            total_improvement = 0.0
            total_weight = 0.0
            
            for intervention in interventions:
                weight = intervention.confidence
                total_improvement += intervention.expected_improvement * weight
                total_weight += weight
            
            if total_weight > 0:
                return total_improvement / total_weight
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating expected improvement: {str(e)}")
            return 0.0
    
    def _generate_personalized_recommendations(self, emotion_data: Dict, attention_data: Dict, 
                                             learning_context: Dict, 
                                             interventions: List[EngagementIntervention]) -> List[str]:
        """Generate personalized recommendations based on interventions"""
        try:
            recommendations = []
            
            for intervention in interventions:
                if intervention.type == InterventionType.CONTENT_ADJUSTMENT:
                    recommendations.append("Consider adjusting content presentation to better match your current emotional state")
                elif intervention.type == InterventionType.DIFFICULTY_MODIFICATION:
                    recommendations.append("The current difficulty level may not be optimal for your attention level")
                elif intervention.type == InterventionType.PACE_CHANGE:
                    recommendations.append("Adjusting the learning pace could improve your engagement")
                elif intervention.type == InterventionType.INTERACTION_ENHANCEMENT:
                    recommendations.append("More interactive elements could boost your emotional engagement")
                elif intervention.type == InterventionType.MOTIVATION_BOOST:
                    recommendations.append("Consider taking a moment to reflect on your learning goals and progress")
                elif intervention.type == InterventionType.BREAK_RECOMMENDATION:
                    recommendations.append("A short break might help restore your focus and energy")
                elif intervention.type == InterventionType.SUPPORT_ACTIVATION:
                    recommendations.append("Don't hesitate to reach out for help - support is available")
            
            # Add general recommendations based on emotion state
            emotion_state = emotion_data.get('overall_emotion_state', 'neutral')
            if emotion_state in ['negative', 'very_negative']:
                recommendations.append("Remember that learning is a process - it's okay to struggle sometimes")
            elif emotion_state == 'confused':
                recommendations.append("Take your time to understand each concept before moving forward")
            elif emotion_state == 'overwhelmed':
                recommendations.append("Break down complex topics into smaller, manageable parts")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating personalized recommendations: {str(e)}")
            return ["Focus on maintaining a positive learning attitude"]
    
    def _define_monitoring_metrics(self, interventions: List[EngagementIntervention]) -> Dict[str, float]:
        """Define metrics to monitor intervention effectiveness"""
        try:
            metrics = {
                'engagement_score': 0.0,
                'attention_score': 0.0,
                'emotional_engagement': 0.0,
                'learning_readiness': 0.0,
                'intervention_effectiveness': 0.0
            }
            
            # Set monitoring intervals based on intervention types
            for intervention in interventions:
                if intervention.type in [InterventionType.CONTENT_ADJUSTMENT, InterventionType.INTERACTION_ENHANCEMENT]:
                    metrics['engagement_score'] = 0.3
                    metrics['emotional_engagement'] = 0.3
                elif intervention.type in [InterventionType.DIFFICULTY_MODIFICATION, InterventionType.PACE_CHANGE]:
                    metrics['attention_score'] = 0.4
                    metrics['learning_readiness'] = 0.3
                elif intervention.type == InterventionType.MOTIVATION_BOOST:
                    metrics['emotional_engagement'] = 0.5
                    metrics['learning_readiness'] = 0.4
                elif intervention.type == InterventionType.BREAK_RECOMMENDATION:
                    metrics['attention_score'] = 0.6
                elif intervention.type == InterventionType.SUPPORT_ACTIVATION:
                    metrics['learning_readiness'] = 0.5
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error defining monitoring metrics: {str(e)}")
            return {'engagement_score': 0.5, 'attention_score': 0.5, 'emotional_engagement': 0.5}
    
    def _calculate_optimization_confidence(self, emotion_data: Dict, attention_data: Dict, 
                                         interventions: List[EngagementIntervention]) -> float:
        """Calculate confidence in optimization recommendations"""
        try:
            # Base confidence on data quality
            emotion_confidence = emotion_data.get('confidence', 0.5)
            attention_confidence = attention_data.get('confidence', 0.5)
            
            # Factor in intervention confidence
            if interventions:
                avg_intervention_confidence = np.mean([interv.confidence for interv in interventions])
            else:
                avg_intervention_confidence = 0.5
            
            # Calculate overall confidence
            overall_confidence = (
                emotion_confidence * 0.4 +
                attention_confidence * 0.4 +
                avg_intervention_confidence * 0.2
            )
            
            return max(0.1, min(1.0, overall_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating optimization confidence: {str(e)}")
            return 0.1
    
    def _get_default_result(self) -> EngagementOptimizationResult:
        """Return default result when optimization fails"""
        return EngagementOptimizationResult(
            interventions=[],
            overall_strategy=OptimizationStrategy.ADAPTIVE,
            expected_engagement_improvement=0.0,
            implementation_priority=[],
            personalized_recommendations=["Focus on maintaining consistent engagement"],
            monitoring_metrics={'engagement_score': 0.5},
            confidence=0.1
        )
    
    def track_intervention_effectiveness(self, intervention: EngagementIntervention, 
                                       before_metrics: Dict, after_metrics: Dict):
        """Track effectiveness of implemented interventions"""
        try:
            # Calculate effectiveness
            improvement = {}
            for metric in before_metrics:
                if metric in after_metrics:
                    improvement[metric] = after_metrics[metric] - before_metrics[metric]
            
            # Store effectiveness data
            effectiveness_entry = {
                'intervention_type': intervention.type.value,
                'timestamp': datetime.now().isoformat(),
                'before_metrics': before_metrics,
                'after_metrics': after_metrics,
                'improvement': improvement,
                'expected_improvement': intervention.expected_improvement
            }
            
            self.intervention_history.append(effectiveness_entry)
            
            # Update effectiveness metrics
            intervention_type = intervention.type.value
            if intervention_type not in self.effectiveness_metrics:
                self.effectiveness_metrics[intervention_type] = []
            
            avg_improvement = np.mean(list(improvement.values())) if improvement else 0.0
            self.effectiveness_metrics[intervention_type].append(avg_improvement)
            
        except Exception as e:
            logger.error(f"Error tracking intervention effectiveness: {str(e)}")
    
    def get_optimization_statistics(self) -> Dict[str, float]:
        """
        Get optimization statistics from historical data
        
        Returns:
            Dictionary with optimization statistics
        """
        if not self.intervention_history:
            return {
                'average_effectiveness': 0.0,
                'most_effective_intervention': 'none',
                'intervention_success_rate': 0.0
            }
        
        # Calculate statistics
        all_improvements = []
        intervention_effectiveness = {}
        
        for entry in self.intervention_history:
            intervention_type = entry['intervention_type']
            improvement = entry['improvement']
            
            if improvement:
                avg_improvement = np.mean(list(improvement.values()))
                all_improvements.append(avg_improvement)
                
                if intervention_type not in intervention_effectiveness:
                    intervention_effectiveness[intervention_type] = []
                intervention_effectiveness[intervention_type].append(avg_improvement)
        
        # Calculate metrics
        average_effectiveness = np.mean(all_improvements) if all_improvements else 0.0
        
        # Most effective intervention
        most_effective = 'none'
        if intervention_effectiveness:
            avg_by_type = {k: np.mean(v) for k, v in intervention_effectiveness.items()}
            most_effective = max(avg_by_type.items(), key=lambda x: x[1])[0]
        
        # Success rate (improvements > 0)
        success_rate = np.mean([imp > 0 for imp in all_improvements]) if all_improvements else 0.0
        
        return {
            'average_effectiveness': float(average_effectiveness),
            'most_effective_intervention': most_effective,
            'intervention_success_rate': float(success_rate)
        }
