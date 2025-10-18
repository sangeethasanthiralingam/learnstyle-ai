"""
Group Dynamics Analysis Module

This module provides comprehensive group dynamics analysis including:
- Group formation and composition analysis
- Collaboration effectiveness measurement
- Group communication pattern analysis
- Leadership emergence detection
- Group performance optimization

Author: LearnStyle AI Team
Version: 1.0.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

class GroupRole(Enum):
    """Group role classifications"""
    LEADER = "leader"
    FACILITATOR = "facilitator"
    CONTRIBUTOR = "contributor"
    OBSERVER = "observer"
    CHALLENGER = "challenger"
    SUPPORTER = "supporter"

class GroupCohesion(Enum):
    """Group cohesion levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class CollaborationEffectiveness(Enum):
    """Collaboration effectiveness levels"""
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"

@dataclass
class GroupMember:
    """Individual group member data"""
    user_id: str
    learning_style: str
    expertise_level: float
    participation_score: float
    communication_frequency: float
    contribution_quality: float
    leadership_tendency: float
    collaboration_style: str

@dataclass
class GroupDynamicsMetrics:
    """Comprehensive group dynamics analysis metrics"""
    group_id: str
    group_size: int
    group_cohesion: GroupCohesion
    collaboration_effectiveness: CollaborationEffectiveness
    communication_balance: float
    leadership_distribution: Dict[str, float]
    role_distribution: Dict[GroupRole, int]
    participation_equality: float
    knowledge_diversity: float
    group_synergy: float
    conflict_level: float
    group_satisfaction: float
    learning_outcomes: float
    recommendations: List[str]
    confidence: float

class GroupDynamicsAnalyzer:
    """
    Advanced group dynamics analysis system
    """
    
    def __init__(self, 
                 cohesion_threshold: float = 0.6,
                 effectiveness_threshold: float = 0.7,
                 participation_threshold: float = 0.5):
        """
        Initialize group dynamics analyzer
        
        Args:
            cohesion_threshold: Threshold for group cohesion classification
            effectiveness_threshold: Threshold for collaboration effectiveness
            participation_threshold: Threshold for participation equality
        """
        self.cohesion_threshold = cohesion_threshold
        self.effectiveness_threshold = effectiveness_threshold
        self.participation_threshold = participation_threshold
        
        # Group dynamics thresholds
        self.cohesion_thresholds = {
            GroupCohesion.VERY_LOW: 0.2,
            GroupCohesion.LOW: 0.4,
            GroupCohesion.MEDIUM: 0.6,
            GroupCohesion.HIGH: 0.8,
            GroupCohesion.VERY_HIGH: 0.9
        }
        
        # Historical data for trend analysis
        self.group_history = []
        self.member_interactions = defaultdict(list)
        
        logger.info("Group Dynamics Analyzer initialized")
    
    def analyze_group_dynamics(self, group_members: List[GroupMember], 
                             interaction_data: Dict, group_id: str) -> GroupDynamicsMetrics:
        """
        Analyze group dynamics from member data and interactions
        
        Args:
            group_members: List of group members
            interaction_data: Group interaction data
            group_id: Unique group identifier
            
        Returns:
            GroupDynamicsMetrics object with comprehensive analysis
        """
        try:
            if not group_members:
                return self._get_default_metrics(group_id)
            
            group_size = len(group_members)
            
            # Calculate group cohesion
            group_cohesion, cohesion_score = self._calculate_group_cohesion(
                group_members, interaction_data
            )
            
            # Calculate collaboration effectiveness
            collaboration_effectiveness, effectiveness_score = self._calculate_collaboration_effectiveness(
                group_members, interaction_data
            )
            
            # Analyze communication patterns
            communication_balance = self._analyze_communication_balance(
                group_members, interaction_data
            )
            
            # Analyze leadership distribution
            leadership_distribution = self._analyze_leadership_distribution(
                group_members, interaction_data
            )
            
            # Determine role distribution
            role_distribution = self._determine_role_distribution(group_members)
            
            # Calculate participation equality
            participation_equality = self._calculate_participation_equality(group_members)
            
            # Calculate knowledge diversity
            knowledge_diversity = self._calculate_knowledge_diversity(group_members)
            
            # Calculate group synergy
            group_synergy = self._calculate_group_synergy(
                group_members, interaction_data, knowledge_diversity
            )
            
            # Assess conflict level
            conflict_level = self._assess_conflict_level(group_members, interaction_data)
            
            # Calculate group satisfaction
            group_satisfaction = self._calculate_group_satisfaction(
                group_members, interaction_data
            )
            
            # Calculate learning outcomes
            learning_outcomes = self._calculate_learning_outcomes(
                group_members, interaction_data
            )
            
            # Generate recommendations
            recommendations = self._generate_group_recommendations(
                group_cohesion, collaboration_effectiveness, participation_equality,
                knowledge_diversity, conflict_level, group_satisfaction
            )
            
            # Calculate confidence
            confidence = self._calculate_analysis_confidence(
                group_members, interaction_data
            )
            
            # Create group dynamics entry
            dynamics_entry = {
                'group_id': group_id,
                'timestamp': datetime.now().isoformat(),
                'group_size': group_size,
                'cohesion_score': cohesion_score,
                'effectiveness_score': effectiveness_score,
                'participation_equality': participation_equality,
                'knowledge_diversity': knowledge_diversity,
                'group_synergy': group_synergy,
                'conflict_level': conflict_level,
                'group_satisfaction': group_satisfaction,
                'learning_outcomes': learning_outcomes
            }
            
            # Update history
            self._update_group_history(dynamics_entry)
            
            return GroupDynamicsMetrics(
                group_id=group_id,
                group_size=group_size,
                group_cohesion=group_cohesion,
                collaboration_effectiveness=collaboration_effectiveness,
                communication_balance=communication_balance,
                leadership_distribution=leadership_distribution,
                role_distribution=role_distribution,
                participation_equality=participation_equality,
                knowledge_diversity=knowledge_diversity,
                group_synergy=group_synergy,
                conflict_level=conflict_level,
                group_satisfaction=group_satisfaction,
                learning_outcomes=learning_outcomes,
                recommendations=recommendations,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing group dynamics: {str(e)}")
            return self._get_default_metrics(group_id)
    
    def _calculate_group_cohesion(self, group_members: List[GroupMember], 
                                interaction_data: Dict) -> Tuple[GroupCohesion, float]:
        """Calculate group cohesion from member data and interactions"""
        try:
            # Factors contributing to cohesion
            participation_scores = [member.participation_score for member in group_members]
            avg_participation = np.mean(participation_scores)
            
            # Communication frequency
            communication_frequencies = [member.communication_frequency for member in group_members]
            avg_communication = np.mean(communication_frequencies)
            
            # Contribution quality
            contribution_qualities = [member.contribution_quality for member in group_members]
            avg_contribution = np.mean(contribution_qualities)
            
            # Interaction balance
            interaction_balance = self._calculate_interaction_balance(group_members, interaction_data)
            
            # Calculate cohesion score
            cohesion_score = (
                avg_participation * 0.3 +
                avg_communication * 0.3 +
                avg_contribution * 0.2 +
                interaction_balance * 0.2
            )
            
            # Classify cohesion level
            if cohesion_score >= self.cohesion_thresholds[GroupCohesion.VERY_HIGH]:
                cohesion_level = GroupCohesion.VERY_HIGH
            elif cohesion_score >= self.cohesion_thresholds[GroupCohesion.HIGH]:
                cohesion_level = GroupCohesion.HIGH
            elif cohesion_score >= self.cohesion_thresholds[GroupCohesion.MEDIUM]:
                cohesion_level = GroupCohesion.MEDIUM
            elif cohesion_score >= self.cohesion_thresholds[GroupCohesion.LOW]:
                cohesion_level = GroupCohesion.LOW
            else:
                cohesion_level = GroupCohesion.VERY_LOW
            
            return cohesion_level, cohesion_score
            
        except Exception as e:
            logger.error(f"Error calculating group cohesion: {str(e)}")
            return GroupCohesion.MEDIUM, 0.5
    
    def _calculate_collaboration_effectiveness(self, group_members: List[GroupMember], 
                                            interaction_data: Dict) -> Tuple[CollaborationEffectiveness, float]:
        """Calculate collaboration effectiveness"""
        try:
            # Factors contributing to effectiveness
            participation_equality = self._calculate_participation_equality(group_members)
            knowledge_diversity = self._calculate_knowledge_diversity(group_members)
            communication_balance = self._analyze_communication_balance(group_members, interaction_data)
            
            # Task completion rate
            task_completion = interaction_data.get('task_completion_rate', 0.5)
            
            # Quality of outcomes
            outcome_quality = interaction_data.get('outcome_quality', 0.5)
            
            # Calculate effectiveness score
            effectiveness_score = (
                participation_equality * 0.25 +
                knowledge_diversity * 0.25 +
                communication_balance * 0.2 +
                task_completion * 0.15 +
                outcome_quality * 0.15
            )
            
            # Classify effectiveness level
            if effectiveness_score >= 0.8:
                effectiveness_level = CollaborationEffectiveness.EXCELLENT
            elif effectiveness_score >= 0.6:
                effectiveness_level = CollaborationEffectiveness.GOOD
            elif effectiveness_score >= 0.4:
                effectiveness_level = CollaborationEffectiveness.FAIR
            else:
                effectiveness_level = CollaborationEffectiveness.POOR
            
            return effectiveness_level, effectiveness_score
            
        except Exception as e:
            logger.error(f"Error calculating collaboration effectiveness: {str(e)}")
            return CollaborationEffectiveness.FAIR, 0.5
    
    def _analyze_communication_balance(self, group_members: List[GroupMember], 
                                     interaction_data: Dict) -> float:
        """Analyze communication balance within the group"""
        try:
            # Calculate communication distribution
            communication_scores = [member.communication_frequency for member in group_members]
            
            if not communication_scores:
                return 0.5
            
            # Balance is inverse of variance (more equal = better balance)
            communication_variance = np.var(communication_scores)
            max_possible_variance = 1.0  # Assuming scores are 0-1
            
            balance_score = 1.0 - (communication_variance / max_possible_variance)
            
            return max(0, min(1, balance_score))
            
        except Exception as e:
            logger.error(f"Error analyzing communication balance: {str(e)}")
            return 0.5
    
    def _analyze_leadership_distribution(self, group_members: List[GroupMember], 
                                       interaction_data: Dict) -> Dict[str, float]:
        """Analyze leadership distribution within the group"""
        try:
            leadership_scores = {}
            
            for member in group_members:
                # Calculate leadership score based on multiple factors
                leadership_score = (
                    member.leadership_tendency * 0.4 +
                    member.participation_score * 0.3 +
                    member.contribution_quality * 0.3
                )
                
                leadership_scores[member.user_id] = leadership_score
            
            # Normalize scores
            total_leadership = sum(leadership_scores.values())
            if total_leadership > 0:
                for user_id in leadership_scores:
                    leadership_scores[user_id] = leadership_scores[user_id] / total_leadership
            
            return leadership_scores
            
        except Exception as e:
            logger.error(f"Error analyzing leadership distribution: {str(e)}")
            return {}
    
    def _determine_role_distribution(self, group_members: List[GroupMember]) -> Dict[GroupRole, int]:
        """Determine role distribution within the group"""
        try:
            role_counts = {role: 0 for role in GroupRole}
            
            for member in group_members:
                # Determine role based on member characteristics
                if member.leadership_tendency > 0.8 and member.participation_score > 0.7:
                    role = GroupRole.LEADER
                elif member.leadership_tendency > 0.6 and member.contribution_quality > 0.7:
                    role = GroupRole.FACILITATOR
                elif member.participation_score > 0.6 and member.contribution_quality > 0.6:
                    role = GroupRole.CONTRIBUTOR
                elif member.participation_score < 0.3:
                    role = GroupRole.OBSERVER
                elif member.leadership_tendency > 0.5 and member.contribution_quality < 0.4:
                    role = GroupRole.CHALLENGER
                else:
                    role = GroupRole.SUPPORTER
                
                role_counts[role] += 1
            
            return role_counts
            
        except Exception as e:
            logger.error(f"Error determining role distribution: {str(e)}")
            return {role: 0 for role in GroupRole}
    
    def _calculate_participation_equality(self, group_members: List[GroupMember]) -> float:
        """Calculate participation equality within the group"""
        try:
            participation_scores = [member.participation_score for member in group_members]
            
            if not participation_scores:
                return 0.5
            
            # Calculate Gini coefficient for participation equality
            participation_scores = sorted(participation_scores)
            n = len(participation_scores)
            
            if n == 0:
                return 0.5
            
            # Calculate Gini coefficient
            cumsum = np.cumsum(participation_scores)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
            
            # Convert to equality score (1 - gini)
            equality_score = 1 - gini
            
            return max(0, min(1, equality_score))
            
        except Exception as e:
            logger.error(f"Error calculating participation equality: {str(e)}")
            return 0.5
    
    def _calculate_knowledge_diversity(self, group_members: List[GroupMember]) -> float:
        """Calculate knowledge diversity within the group"""
        try:
            # Extract learning styles and expertise levels
            learning_styles = [member.learning_style for member in group_members]
            expertise_levels = [member.expertise_level for member in group_members]
            
            # Calculate learning style diversity
            style_counts = {}
            for style in learning_styles:
                style_counts[style] = style_counts.get(style, 0) + 1
            
            # Shannon entropy for learning style diversity
            total_styles = len(learning_styles)
            style_entropy = 0
            for count in style_counts.values():
                if count > 0:
                    p = count / total_styles
                    style_entropy -= p * np.log2(p)
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(set(learning_styles))) if len(set(learning_styles)) > 1 else 1
            style_diversity = style_entropy / max_entropy if max_entropy > 0 else 0
            
            # Calculate expertise diversity
            expertise_variance = np.var(expertise_levels) if len(expertise_levels) > 1 else 0
            expertise_diversity = min(1.0, expertise_variance)
            
            # Combined diversity score
            diversity_score = (style_diversity * 0.6 + expertise_diversity * 0.4)
            
            return max(0, min(1, diversity_score))
            
        except Exception as e:
            logger.error(f"Error calculating knowledge diversity: {str(e)}")
            return 0.5
    
    def _calculate_group_synergy(self, group_members: List[GroupMember], 
                               interaction_data: Dict, knowledge_diversity: float) -> float:
        """Calculate group synergy"""
        try:
            # Individual performance
            individual_scores = [member.contribution_quality for member in group_members]
            avg_individual_performance = np.mean(individual_scores)
            
            # Group performance (from interaction data)
            group_performance = interaction_data.get('group_performance', avg_individual_performance)
            
            # Calculate synergy (group performance vs sum of individual performances)
            individual_sum = sum(individual_scores)
            synergy_ratio = group_performance / individual_sum if individual_sum > 0 else 1.0
            
            # Factor in knowledge diversity (diversity can enhance synergy)
            diversity_factor = 1.0 + (knowledge_diversity * 0.2)
            
            synergy_score = min(2.0, synergy_ratio * diversity_factor)
            
            return max(0, min(1, synergy_score))
            
        except Exception as e:
            logger.error(f"Error calculating group synergy: {str(e)}")
            return 0.5
    
    def _assess_conflict_level(self, group_members: List[GroupMember], 
                             interaction_data: Dict) -> float:
        """Assess conflict level within the group"""
        try:
            # Factors indicating conflict
            participation_variance = np.var([member.participation_score for member in group_members])
            communication_imbalance = 1.0 - self._analyze_communication_balance(group_members, interaction_data)
            
            # Direct conflict indicators from interaction data
            conflict_indicators = interaction_data.get('conflict_indicators', {})
            disagreement_frequency = conflict_indicators.get('disagreement_frequency', 0.0)
            negative_sentiment = conflict_indicators.get('negative_sentiment', 0.0)
            
            # Calculate conflict score
            conflict_score = (
                participation_variance * 0.3 +
                communication_imbalance * 0.3 +
                disagreement_frequency * 0.2 +
                negative_sentiment * 0.2
            )
            
            return max(0, min(1, conflict_score))
            
        except Exception as e:
            logger.error(f"Error assessing conflict level: {str(e)}")
            return 0.0
    
    def _calculate_group_satisfaction(self, group_members: List[GroupMember], 
                                    interaction_data: Dict) -> float:
        """Calculate group satisfaction"""
        try:
            # Member satisfaction indicators
            satisfaction_indicators = []
            
            for member in group_members:
                # Satisfaction based on participation and contribution quality
                member_satisfaction = (
                    member.participation_score * 0.5 +
                    member.contribution_quality * 0.5
                )
                satisfaction_indicators.append(member_satisfaction)
            
            # Group-level satisfaction from interaction data
            group_satisfaction = interaction_data.get('group_satisfaction', 0.5)
            
            # Combined satisfaction score
            avg_member_satisfaction = np.mean(satisfaction_indicators) if satisfaction_indicators else 0.5
            combined_satisfaction = (avg_member_satisfaction * 0.7 + group_satisfaction * 0.3)
            
            return max(0, min(1, combined_satisfaction))
            
        except Exception as e:
            logger.error(f"Error calculating group satisfaction: {str(e)}")
            return 0.5
    
    def _calculate_learning_outcomes(self, group_members: List[GroupMember], 
                                   interaction_data: Dict) -> float:
        """Calculate learning outcomes for the group"""
        try:
            # Individual learning outcomes
            individual_outcomes = []
            for member in group_members:
                # Learning outcome based on contribution quality and participation
                member_outcome = (
                    member.contribution_quality * 0.6 +
                    member.participation_score * 0.4
                )
                individual_outcomes.append(member_outcome)
            
            # Group learning outcomes from interaction data
            group_outcomes = interaction_data.get('learning_outcomes', {})
            knowledge_gain = group_outcomes.get('knowledge_gain', 0.5)
            skill_development = group_outcomes.get('skill_development', 0.5)
            collaboration_skills = group_outcomes.get('collaboration_skills', 0.5)
            
            # Combined learning outcomes
            avg_individual_outcomes = np.mean(individual_outcomes) if individual_outcomes else 0.5
            group_learning_score = (knowledge_gain + skill_development + collaboration_skills) / 3.0
            
            combined_outcomes = (avg_individual_outcomes * 0.6 + group_learning_score * 0.4)
            
            return max(0, min(1, combined_outcomes))
            
        except Exception as e:
            logger.error(f"Error calculating learning outcomes: {str(e)}")
            return 0.5
    
    def _calculate_interaction_balance(self, group_members: List[GroupMember], 
                                     interaction_data: Dict) -> float:
        """Calculate interaction balance within the group"""
        try:
            # Get interaction matrix from interaction data
            interaction_matrix = interaction_data.get('interaction_matrix', {})
            
            if not interaction_matrix:
                return 0.5
            
            # Calculate interaction balance
            interactions = []
            for member in group_members:
                member_interactions = interaction_matrix.get(member.user_id, {})
                total_interactions = sum(member_interactions.values())
                interactions.append(total_interactions)
            
            if not interactions:
                return 0.5
            
            # Balance is inverse of variance
            interaction_variance = np.var(interactions)
            max_variance = max(interactions) - min(interactions) if max(interactions) > min(interactions) else 1
            
            balance_score = 1.0 - (interaction_variance / max_variance) if max_variance > 0 else 0.5
            
            return max(0, min(1, balance_score))
            
        except Exception as e:
            logger.error(f"Error calculating interaction balance: {str(e)}")
            return 0.5
    
    def _generate_group_recommendations(self, group_cohesion: GroupCohesion, 
                                      collaboration_effectiveness: CollaborationEffectiveness,
                                      participation_equality: float, knowledge_diversity: float,
                                      conflict_level: float, group_satisfaction: float) -> List[str]:
        """Generate recommendations for group improvement"""
        try:
            recommendations = []
            
            # Cohesion recommendations
            if group_cohesion in [GroupCohesion.VERY_LOW, GroupCohesion.LOW]:
                recommendations.append("Encourage more frequent communication and interaction among group members")
                recommendations.append("Organize team-building activities to improve group cohesion")
            
            # Effectiveness recommendations
            if collaboration_effectiveness in [CollaborationEffectiveness.POOR, CollaborationEffectiveness.FAIR]:
                recommendations.append("Improve task distribution and role clarity within the group")
                recommendations.append("Provide training on effective collaboration techniques")
            
            # Participation equality recommendations
            if participation_equality < 0.5:
                recommendations.append("Encourage quieter members to participate more actively")
                recommendations.append("Implement structured participation protocols")
            
            # Knowledge diversity recommendations
            if knowledge_diversity < 0.3:
                recommendations.append("Encourage knowledge sharing and cross-training among members")
                recommendations.append("Assign diverse roles to leverage different expertise")
            
            # Conflict recommendations
            if conflict_level > 0.6:
                recommendations.append("Address conflicts through mediation and open communication")
                recommendations.append("Establish clear group norms and conflict resolution procedures")
            
            # Satisfaction recommendations
            if group_satisfaction < 0.5:
                recommendations.append("Gather feedback and address member concerns")
                recommendations.append("Recognize and celebrate group achievements")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Continue current collaborative practices")
                recommendations.append("Monitor group dynamics and adjust as needed")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating group recommendations: {str(e)}")
            return ["Monitor group dynamics and provide support as needed"]
    
    def _calculate_analysis_confidence(self, group_members: List[GroupMember], 
                                     interaction_data: Dict) -> float:
        """Calculate confidence in group dynamics analysis"""
        try:
            # Factors affecting confidence
            group_size = len(group_members)
            data_completeness = len(interaction_data) / 10.0  # Normalize by expected data points
            
            # Member data quality
            member_data_quality = np.mean([
                member.participation_score + member.contribution_quality + member.communication_frequency
                for member in group_members
            ]) / 3.0
            
            # Calculate confidence
            confidence = (
                min(1.0, group_size / 5.0) * 0.3 +  # Optimal group size around 5
                min(1.0, data_completeness) * 0.4 +
                member_data_quality * 0.3
            )
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating analysis confidence: {str(e)}")
            return 0.1
    
    def _update_group_history(self, dynamics_entry: Dict):
        """Update group history for trend analysis"""
        self.group_history.append(dynamics_entry)
        
        # Keep only recent history
        if len(self.group_history) > 100:
            self.group_history = self.group_history[-100:]
    
    def _get_default_metrics(self, group_id: str) -> GroupDynamicsMetrics:
        """Return default metrics when analysis fails"""
        return GroupDynamicsMetrics(
            group_id=group_id,
            group_size=0,
            group_cohesion=GroupCohesion.MEDIUM,
            collaboration_effectiveness=CollaborationEffectiveness.FAIR,
            communication_balance=0.5,
            leadership_distribution={},
            role_distribution={role: 0 for role in GroupRole},
            participation_equality=0.5,
            knowledge_diversity=0.5,
            group_synergy=0.5,
            conflict_level=0.0,
            group_satisfaction=0.5,
            learning_outcomes=0.5,
            recommendations=["Insufficient data for analysis"],
            confidence=0.1
        )
    
    def get_group_statistics(self) -> Dict[str, float]:
        """
        Get group dynamics statistics from historical data
        
        Returns:
            Dictionary with group statistics
        """
        if not self.group_history:
            return {
                'average_cohesion': 0.0,
                'average_effectiveness': 0.0,
                'average_participation_equality': 0.0,
                'group_performance_trend': 0.0
            }
        
        # Calculate statistics
        cohesion_scores = [entry['cohesion_score'] for entry in self.group_history]
        effectiveness_scores = [entry['effectiveness_score'] for entry in self.group_history]
        participation_equalities = [entry['participation_equality'] for entry in self.group_history]
        learning_outcomes = [entry['learning_outcomes'] for entry in self.group_history]
        
        # Average metrics
        average_cohesion = np.mean(cohesion_scores)
        average_effectiveness = np.mean(effectiveness_scores)
        average_participation_equality = np.mean(participation_equalities)
        
        # Performance trend
        if len(learning_outcomes) > 1:
            performance_trend = np.polyfit(range(len(learning_outcomes)), learning_outcomes, 1)[0]
        else:
            performance_trend = 0.0
        
        return {
            'average_cohesion': float(average_cohesion),
            'average_effectiveness': float(average_effectiveness),
            'average_participation_equality': float(average_participation_equality),
            'group_performance_trend': float(performance_trend)
        }
