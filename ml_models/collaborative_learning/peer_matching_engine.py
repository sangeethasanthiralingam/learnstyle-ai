"""
Peer Matching Engine Module

This module provides intelligent peer matching and learning partnerships including:
- Optimal peer matching based on learning styles and expertise
- Complementary skill pairing
- Learning goal alignment
- Personality compatibility assessment
- Dynamic group formation

Author: LearnStyle AI Team
Version: 1.0.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
from itertools import combinations
import random

logger = logging.getLogger(__name__)

class MatchingStrategy(Enum):
    """Peer matching strategies"""
    COMPLEMENTARY = "complementary"      # Match different strengths
    SIMILAR = "similar"                 # Match similar characteristics
    MENTORSHIP = "mentorship"           # Expert-novice pairing
    COLLABORATIVE = "collaborative"     # Equal partnership
    DIVERSITY = "diversity"             # Maximize diversity

class CompatibilityLevel(Enum):
    """Compatibility level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class LearningGoal(Enum):
    """Learning goal types"""
    SKILL_DEVELOPMENT = "skill_development"
    KNOWLEDGE_ACQUISITION = "knowledge_acquisition"
    PROJECT_COLLABORATION = "project_collaboration"
    PEER_TUTORING = "peer_tutoring"
    RESEARCH_COLLABORATION = "research_collaboration"

@dataclass
class PeerProfile:
    """Individual peer profile data"""
    user_id: str
    learning_style: str
    expertise_areas: List[str]
    expertise_levels: Dict[str, float]
    learning_goals: List[LearningGoal]
    personality_traits: Dict[str, float]
    availability: Dict[str, List[int]]  # Day of week -> hours available
    communication_preferences: List[str]
    collaboration_style: str
    preferred_group_size: int
    timezone: str

@dataclass
class PeerMatch:
    """Peer matching result"""
    peer1_id: str
    peer2_id: str
    compatibility_score: float
    compatibility_level: CompatibilityLevel
    matching_strategy: MatchingStrategy
    strengths: List[str]
    learning_opportunities: List[str]
    potential_challenges: List[str]
    recommended_activities: List[str]
    confidence: float

@dataclass
class GroupFormation:
    """Group formation result"""
    group_id: str
    members: List[str]
    group_size: int
    group_cohesion: float
    skill_diversity: float
    learning_goal_alignment: float
    recommended_roles: Dict[str, str]
    group_activities: List[str]
    success_probability: float

class PeerMatchingEngine:
    """
    Advanced peer matching and group formation system
    """
    
    def __init__(self, 
                 compatibility_threshold: float = 0.6,
                 max_group_size: int = 6,
                 min_group_size: int = 2):
        """
        Initialize peer matching engine
        
        Args:
            compatibility_threshold: Threshold for compatibility classification
            max_group_size: Maximum group size for formation
            min_group_size: Minimum group size for formation
        """
        self.compatibility_threshold = compatibility_threshold
        self.max_group_size = max_group_size
        self.min_group_size = min_group_size
        
        # Compatibility level thresholds
        self.compatibility_thresholds = {
            CompatibilityLevel.LOW: 0.3,
            CompatibilityLevel.MEDIUM: 0.5,
            CompatibilityLevel.HIGH: 0.7,
            CompatibilityLevel.VERY_HIGH: 0.9
        }
        
        # Learning style compatibility matrix
        self.learning_style_compatibility = self._initialize_learning_style_compatibility()
        
        # Historical matching data
        self.matching_history = []
        self.success_rates = {}
        
        logger.info("Peer Matching Engine initialized")
    
    def find_peer_matches(self, user_profile: PeerProfile, 
                         candidate_profiles: List[PeerProfile],
                         matching_strategy: MatchingStrategy = MatchingStrategy.COMPLEMENTARY,
                         max_matches: int = 5) -> List[PeerMatch]:
        """
        Find optimal peer matches for a user
        
        Args:
            user_profile: Profile of the user seeking matches
            candidate_profiles: List of candidate peer profiles
            matching_strategy: Strategy for matching
            max_matches: Maximum number of matches to return
            
        Returns:
            List of PeerMatch objects
        """
        try:
            matches = []
            
            for candidate in candidate_profiles:
                if candidate.user_id == user_profile.user_id:
                    continue
                
                # Calculate compatibility score
                compatibility_score = self._calculate_compatibility(
                    user_profile, candidate, matching_strategy
                )
                
                # Determine compatibility level
                compatibility_level = self._classify_compatibility(compatibility_score)
                
                # Generate match details
                strengths = self._identify_match_strengths(user_profile, candidate)
                learning_opportunities = self._identify_learning_opportunities(user_profile, candidate)
                potential_challenges = self._identify_potential_challenges(user_profile, candidate)
                recommended_activities = self._recommend_activities(
                    user_profile, candidate, matching_strategy
                )
                
                # Calculate confidence
                confidence = self._calculate_match_confidence(
                    user_profile, candidate, compatibility_score
                )
                
                match = PeerMatch(
                    peer1_id=user_profile.user_id,
                    peer2_id=candidate.user_id,
                    compatibility_score=compatibility_score,
                    compatibility_level=compatibility_level,
                    matching_strategy=matching_strategy,
                    strengths=strengths,
                    learning_opportunities=learning_opportunities,
                    potential_challenges=potential_challenges,
                    recommended_activities=recommended_activities,
                    confidence=confidence
                )
                
                matches.append(match)
            
            # Sort by compatibility score and return top matches
            matches.sort(key=lambda x: x.compatibility_score, reverse=True)
            return matches[:max_matches]
            
        except Exception as e:
            logger.error(f"Error finding peer matches: {str(e)}")
            return []
    
    def form_optimal_group(self, candidate_profiles: List[PeerProfile],
                          target_size: int = 4,
                          learning_goal: LearningGoal = LearningGoal.SKILL_DEVELOPMENT,
                          strategy: MatchingStrategy = MatchingStrategy.COLLABORATIVE) -> Optional[GroupFormation]:
        """
        Form an optimal group from candidate profiles
        
        Args:
            candidate_profiles: List of candidate profiles
            target_size: Target group size
            learning_goal: Primary learning goal for the group
            strategy: Group formation strategy
            
        Returns:
            GroupFormation object or None
        """
        try:
            if len(candidate_profiles) < self.min_group_size:
                return None
            
            # Adjust target size based on available candidates
            actual_size = min(target_size, len(candidate_profiles), self.max_group_size)
            
            # Generate group combinations
            best_group = None
            best_score = 0.0
            
            # Try different combinations (limit to avoid computational explosion)
            max_combinations = min(100, len(list(combinations(candidate_profiles, actual_size))))
            
            for i, combination in enumerate(combinations(candidate_profiles, actual_size)):
                if i >= max_combinations:
                    break
                
                group_score = self._evaluate_group_combination(
                    list(combination), learning_goal, strategy
                )
                
                if group_score > best_score:
                    best_score = group_score
                    best_group = list(combination)
            
            if not best_group:
                return None
            
            # Create group formation result
            group_id = f"group_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            member_ids = [member.user_id for member in best_group]
            
            # Calculate group metrics
            group_cohesion = self._calculate_group_cohesion(best_group)
            skill_diversity = self._calculate_skill_diversity(best_group)
            learning_goal_alignment = self._calculate_learning_goal_alignment(best_group, learning_goal)
            
            # Assign recommended roles
            recommended_roles = self._assign_group_roles(best_group)
            
            # Recommend group activities
            group_activities = self._recommend_group_activities(best_group, learning_goal)
            
            # Calculate success probability
            success_probability = self._calculate_group_success_probability(
                best_group, group_cohesion, skill_diversity, learning_goal_alignment
            )
            
            return GroupFormation(
                group_id=group_id,
                members=member_ids,
                group_size=len(best_group),
                group_cohesion=group_cohesion,
                skill_diversity=skill_diversity,
                learning_goal_alignment=learning_goal_alignment,
                recommended_roles=recommended_roles,
                group_activities=group_activities,
                success_probability=success_probability
            )
            
        except Exception as e:
            logger.error(f"Error forming optimal group: {str(e)}")
            return None
    
    def _calculate_compatibility(self, profile1: PeerProfile, profile2: PeerProfile, 
                               strategy: MatchingStrategy) -> float:
        """Calculate compatibility score between two profiles"""
        try:
            compatibility_factors = []
            
            # Learning style compatibility
            learning_style_score = self._calculate_learning_style_compatibility(
                profile1.learning_style, profile2.learning_style, strategy
            )
            compatibility_factors.append(learning_style_score)
            
            # Expertise compatibility
            expertise_score = self._calculate_expertise_compatibility(
                profile1.expertise_areas, profile1.expertise_levels,
                profile2.expertise_areas, profile2.expertise_levels, strategy
            )
            compatibility_factors.append(expertise_score)
            
            # Learning goal compatibility
            goal_score = self._calculate_goal_compatibility(
                profile1.learning_goals, profile2.learning_goals
            )
            compatibility_factors.append(goal_score)
            
            # Personality compatibility
            personality_score = self._calculate_personality_compatibility(
                profile1.personality_traits, profile2.personality_traits
            )
            compatibility_factors.append(personality_score)
            
            # Communication compatibility
            communication_score = self._calculate_communication_compatibility(
                profile1.communication_preferences, profile2.communication_preferences
            )
            compatibility_factors.append(communication_score)
            
            # Collaboration style compatibility
            collaboration_score = self._calculate_collaboration_compatibility(
                profile1.collaboration_style, profile2.collaboration_style
            )
            compatibility_factors.append(collaboration_score)
            
            # Calculate weighted average
            weights = [0.25, 0.2, 0.15, 0.15, 0.15, 0.1]
            compatibility_score = sum(score * weight for score, weight in zip(compatibility_factors, weights))
            
            return max(0, min(1, compatibility_score))
            
        except Exception as e:
            logger.error(f"Error calculating compatibility: {str(e)}")
            return 0.5
    
    def _calculate_learning_style_compatibility(self, style1: str, style2: str, 
                                             strategy: MatchingStrategy) -> float:
        """Calculate learning style compatibility"""
        try:
            if strategy == MatchingStrategy.SIMILAR:
                # Similar styles are better
                return 1.0 if style1 == style2 else 0.3
            elif strategy == MatchingStrategy.COMPLEMENTARY:
                # Complementary styles are better
                return self.learning_style_compatibility.get((style1, style2), 0.5)
            elif strategy == MatchingStrategy.MENTORSHIP:
                # Different styles can work for mentorship
                return 0.7 if style1 != style2 else 0.5
            else:
                # Default to complementary
                return self.learning_style_compatibility.get((style1, style2), 0.5)
                
        except Exception as e:
            logger.error(f"Error calculating learning style compatibility: {str(e)}")
            return 0.5
    
    def _calculate_expertise_compatibility(self, areas1: List[str], levels1: Dict[str, float],
                                        areas2: List[str], levels2: Dict[str, float],
                                        strategy: MatchingStrategy) -> float:
        """Calculate expertise compatibility"""
        try:
            if strategy == MatchingStrategy.MENTORSHIP:
                # Look for expert-novice pairing
                expertise_gap = 0
                common_areas = set(areas1) & set(areas2)
                
                for area in common_areas:
                    level1 = levels1.get(area, 0)
                    level2 = levels2.get(area, 0)
                    expertise_gap += abs(level1 - level2)
                
                # Higher gap is better for mentorship
                return min(1.0, expertise_gap / len(common_areas)) if common_areas else 0.3
                
            elif strategy == MatchingStrategy.COMPLEMENTARY:
                # Look for complementary expertise
                unique_areas1 = set(areas1) - set(areas2)
                unique_areas2 = set(areas2) - set(areas1)
                complementarity = len(unique_areas1) + len(unique_areas2)
                
                return min(1.0, complementarity / 5.0)  # Normalize
                
            else:
                # Look for similar expertise levels
                common_areas = set(areas1) & set(areas2)
                if not common_areas:
                    return 0.3
                
                level_similarity = 0
                for area in common_areas:
                    level1 = levels1.get(area, 0)
                    level2 = levels2.get(area, 0)
                    similarity = 1.0 - abs(level1 - level2)
                    level_similarity += similarity
                
                return level_similarity / len(common_areas)
                
        except Exception as e:
            logger.error(f"Error calculating expertise compatibility: {str(e)}")
            return 0.5
    
    def _calculate_goal_compatibility(self, goals1: List[LearningGoal], 
                                   goals2: List[LearningGoal]) -> float:
        """Calculate learning goal compatibility"""
        try:
            if not goals1 or not goals2:
                return 0.5
            
            # Calculate overlap in learning goals
            common_goals = set(goals1) & set(goals2)
            total_goals = len(set(goals1) | set(goals2))
            
            if total_goals == 0:
                return 0.5
            
            overlap_ratio = len(common_goals) / total_goals
            return overlap_ratio
            
        except Exception as e:
            logger.error(f"Error calculating goal compatibility: {str(e)}")
            return 0.5
    
    def _calculate_personality_compatibility(self, traits1: Dict[str, float], 
                                          traits2: Dict[str, float]) -> float:
        """Calculate personality compatibility"""
        try:
            if not traits1 or not traits2:
                return 0.5
            
            # Calculate compatibility for each trait
            compatibility_scores = []
            
            for trait in traits1:
                if trait in traits2:
                    value1 = traits1[trait]
                    value2 = traits2[trait]
                    
                    # For most traits, similarity is good
                    # For some traits like dominance, complementarity might be better
                    if trait in ['dominance', 'assertiveness']:
                        # Some complementarity is good
                        compatibility = 1.0 - abs(value1 - value2) * 0.5
                    else:
                        # Similarity is generally good
                        compatibility = 1.0 - abs(value1 - value2)
                    
                    compatibility_scores.append(compatibility)
            
            return np.mean(compatibility_scores) if compatibility_scores else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating personality compatibility: {str(e)}")
            return 0.5
    
    def _calculate_communication_compatibility(self, prefs1: List[str], 
                                            prefs2: List[str]) -> float:
        """Calculate communication preference compatibility"""
        try:
            if not prefs1 or not prefs2:
                return 0.5
            
            # Calculate overlap in communication preferences
            common_prefs = set(prefs1) & set(prefs2)
            total_prefs = len(set(prefs1) | set(prefs2))
            
            if total_prefs == 0:
                return 0.5
            
            overlap_ratio = len(common_prefs) / total_prefs
            return overlap_ratio
            
        except Exception as e:
            logger.error(f"Error calculating communication compatibility: {str(e)}")
            return 0.5
    
    def _calculate_collaboration_compatibility(self, style1: str, style2: str) -> float:
        """Calculate collaboration style compatibility"""
        try:
            # Define collaboration style compatibility matrix
            style_compatibility = {
                ('leader', 'follower'): 0.9,
                ('follower', 'leader'): 0.9,
                ('collaborator', 'collaborator'): 0.8,
                ('independent', 'independent'): 0.7,
                ('leader', 'leader'): 0.4,
                ('follower', 'follower'): 0.6,
                ('collaborator', 'independent'): 0.5,
                ('independent', 'collaborator'): 0.5,
            }
            
            # Check both directions
            compatibility = style_compatibility.get((style1, style2), 0.5)
            reverse_compatibility = style_compatibility.get((style2, style1), 0.5)
            
            return max(compatibility, reverse_compatibility)
            
        except Exception as e:
            logger.error(f"Error calculating collaboration compatibility: {str(e)}")
            return 0.5
    
    def _classify_compatibility(self, score: float) -> CompatibilityLevel:
        """Classify compatibility level from score"""
        if score >= self.compatibility_thresholds[CompatibilityLevel.VERY_HIGH]:
            return CompatibilityLevel.VERY_HIGH
        elif score >= self.compatibility_thresholds[CompatibilityLevel.HIGH]:
            return CompatibilityLevel.HIGH
        elif score >= self.compatibility_thresholds[CompatibilityLevel.MEDIUM]:
            return CompatibilityLevel.MEDIUM
        else:
            return CompatibilityLevel.LOW
    
    def _identify_match_strengths(self, profile1: PeerProfile, profile2: PeerProfile) -> List[str]:
        """Identify strengths of a peer match"""
        try:
            strengths = []
            
            # Learning style strengths
            if profile1.learning_style != profile2.learning_style:
                strengths.append(f"Complementary learning styles: {profile1.learning_style} and {profile2.learning_style}")
            
            # Expertise strengths
            unique_areas1 = set(profile1.expertise_areas) - set(profile2.expertise_areas)
            unique_areas2 = set(profile2.expertise_areas) - set(profile1.expertise_areas)
            
            if unique_areas1:
                strengths.append(f"{profile1.user_id} brings expertise in: {', '.join(unique_areas1)}")
            if unique_areas2:
                strengths.append(f"{profile2.user_id} brings expertise in: {', '.join(unique_areas2)}")
            
            # Goal alignment
            common_goals = set(profile1.learning_goals) & set(profile2.learning_goals)
            if common_goals:
                strengths.append(f"Shared learning goals: {', '.join([goal.value for goal in common_goals])}")
            
            # Communication compatibility
            common_comm = set(profile1.communication_preferences) & set(profile2.communication_preferences)
            if common_comm:
                strengths.append(f"Compatible communication preferences: {', '.join(common_comm)}")
            
            return strengths
            
        except Exception as e:
            logger.error(f"Error identifying match strengths: {str(e)}")
            return ["Potential for mutual learning"]
    
    def _identify_learning_opportunities(self, profile1: PeerProfile, profile2: PeerProfile) -> List[str]:
        """Identify learning opportunities in a peer match"""
        try:
            opportunities = []
            
            # Skill exchange opportunities
            for area in profile1.expertise_areas:
                if area not in profile2.expertise_areas:
                    level1 = profile1.expertise_levels.get(area, 0)
                    if level1 > 0.6:  # High expertise
                        opportunities.append(f"{profile1.user_id} can teach {area} to {profile2.user_id}")
            
            for area in profile2.expertise_areas:
                if area not in profile1.expertise_areas:
                    level2 = profile2.expertise_levels.get(area, 0)
                    if level2 > 0.6:  # High expertise
                        opportunities.append(f"{profile2.user_id} can teach {area} to {profile1.user_id}")
            
            # Collaborative learning opportunities
            common_areas = set(profile1.expertise_areas) & set(profile2.expertise_areas)
            for area in common_areas:
                level1 = profile1.expertise_levels.get(area, 0)
                level2 = profile2.expertise_levels.get(area, 0)
                
                if abs(level1 - level2) < 0.3:  # Similar levels
                    opportunities.append(f"Collaborative learning in {area}")
                elif level1 > level2 + 0.3:  # Profile1 is more expert
                    opportunities.append(f"Mentorship opportunity: {profile1.user_id} can mentor {profile2.user_id} in {area}")
                elif level2 > level1 + 0.3:  # Profile2 is more expert
                    opportunities.append(f"Mentorship opportunity: {profile2.user_id} can mentor {profile1.user_id} in {area}")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying learning opportunities: {str(e)}")
            return ["Mutual learning and skill development"]
    
    def _identify_potential_challenges(self, profile1: PeerProfile, profile2: PeerProfile) -> List[str]:
        """Identify potential challenges in a peer match"""
        try:
            challenges = []
            
            # Communication challenges
            if not (set(profile1.communication_preferences) & set(profile2.communication_preferences)):
                challenges.append("Different communication preferences may require adaptation")
            
            # Collaboration style challenges
            if profile1.collaboration_style == 'independent' and profile2.collaboration_style == 'collaborator':
                challenges.append("Different collaboration styles may need alignment")
            
            # Availability challenges
            if not self._check_availability_overlap(profile1.availability, profile2.availability):
                challenges.append("Limited overlapping availability may affect collaboration")
            
            # Timezone challenges
            if profile1.timezone != profile2.timezone:
                challenges.append("Different timezones may require scheduling coordination")
            
            return challenges
            
        except Exception as e:
            logger.error(f"Error identifying potential challenges: {str(e)}")
            return ["May require initial adjustment period"]
    
    def _recommend_activities(self, profile1: PeerProfile, profile2: PeerProfile, 
                           strategy: MatchingStrategy) -> List[str]:
        """Recommend activities for a peer match"""
        try:
            activities = []
            
            if strategy == MatchingStrategy.MENTORSHIP:
                activities.extend([
                    "One-on-one mentoring sessions",
                    "Skill demonstration and practice",
                    "Project-based learning with guidance",
                    "Regular check-ins and progress reviews"
                ])
            elif strategy == MatchingStrategy.COLLABORATIVE:
                activities.extend([
                    "Joint project development",
                    "Peer review and feedback sessions",
                    "Collaborative problem-solving",
                    "Knowledge sharing sessions"
                ])
            elif strategy == MatchingStrategy.COMPLEMENTARY:
                activities.extend([
                    "Cross-training in different skills",
                    "Complementary role assignments",
                    "Skill exchange workshops",
                    "Diverse perspective discussions"
                ])
            else:
                activities.extend([
                    "Study groups and discussions",
                    "Peer learning activities",
                    "Collaborative assignments",
                    "Regular check-ins and support"
                ])
            
            return activities
            
        except Exception as e:
            logger.error(f"Error recommending activities: {str(e)}")
            return ["Collaborative learning activities"]
    
    def _calculate_match_confidence(self, profile1: PeerProfile, profile2: PeerProfile, 
                                  compatibility_score: float) -> float:
        """Calculate confidence in peer match"""
        try:
            # Factors affecting confidence
            profile_completeness1 = self._calculate_profile_completeness(profile1)
            profile_completeness2 = self._calculate_profile_completeness(profile2)
            
            # Data quality
            data_quality = (profile_completeness1 + profile_completeness2) / 2.0
            
            # Compatibility strength
            compatibility_strength = compatibility_score
            
            # Combined confidence
            confidence = (data_quality * 0.6 + compatibility_strength * 0.4)
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating match confidence: {str(e)}")
            return 0.5
    
    def _calculate_profile_completeness(self, profile: PeerProfile) -> float:
        """Calculate profile completeness score"""
        try:
            completeness_factors = []
            
            # Basic information
            if profile.learning_style:
                completeness_factors.append(1.0)
            if profile.expertise_areas:
                completeness_factors.append(1.0)
            if profile.learning_goals:
                completeness_factors.append(1.0)
            if profile.personality_traits:
                completeness_factors.append(1.0)
            if profile.availability:
                completeness_factors.append(1.0)
            if profile.communication_preferences:
                completeness_factors.append(1.0)
            
            return np.mean(completeness_factors) if completeness_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating profile completeness: {str(e)}")
            return 0.0
    
    def _check_availability_overlap(self, availability1: Dict[str, List[int]], 
                                  availability2: Dict[str, List[int]]) -> bool:
        """Check if two profiles have overlapping availability"""
        try:
            for day in availability1:
                if day in availability2:
                    hours1 = set(availability1[day])
                    hours2 = set(availability2[day])
                    if hours1 & hours2:  # Intersection is not empty
                        return True
            return False
            
        except Exception as e:
            logger.error(f"Error checking availability overlap: {str(e)}")
            return True  # Assume overlap if error
    
    def _evaluate_group_combination(self, group: List[PeerProfile], 
                                  learning_goal: LearningGoal, 
                                  strategy: MatchingStrategy) -> float:
        """Evaluate a group combination"""
        try:
            # Calculate group metrics
            cohesion = self._calculate_group_cohesion(group)
            diversity = self._calculate_skill_diversity(group)
            goal_alignment = self._calculate_learning_goal_alignment(group, learning_goal)
            
            # Calculate overall group score
            group_score = (cohesion * 0.4 + diversity * 0.3 + goal_alignment * 0.3)
            
            return group_score
            
        except Exception as e:
            logger.error(f"Error evaluating group combination: {str(e)}")
            return 0.0
    
    def _calculate_group_cohesion(self, group: List[PeerProfile]) -> float:
        """Calculate group cohesion"""
        try:
            if len(group) < 2:
                return 0.5
            
            # Calculate average compatibility between all pairs
            compatibilities = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    compatibility = self._calculate_compatibility(
                        group[i], group[j], MatchingStrategy.COLLABORATIVE
                    )
                    compatibilities.append(compatibility)
            
            return np.mean(compatibilities) if compatibilities else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating group cohesion: {str(e)}")
            return 0.5
    
    def _calculate_skill_diversity(self, group: List[PeerProfile]) -> float:
        """Calculate skill diversity in group"""
        try:
            all_areas = set()
            for member in group:
                all_areas.update(member.expertise_areas)
            
            # Diversity is higher with more unique areas
            diversity = len(all_areas) / (len(group) * 3)  # Normalize by expected areas per person
            return min(1.0, diversity)
            
        except Exception as e:
            logger.error(f"Error calculating skill diversity: {str(e)}")
            return 0.5
    
    def _calculate_learning_goal_alignment(self, group: List[PeerProfile], 
                                         learning_goal: LearningGoal) -> float:
        """Calculate learning goal alignment in group"""
        try:
            goal_counts = {}
            for member in group:
                for goal in member.learning_goals:
                    goal_counts[goal] = goal_counts.get(goal, 0) + 1
            
            # Check alignment with target goal
            target_goal_count = goal_counts.get(learning_goal, 0)
            alignment = target_goal_count / len(group)
            
            return alignment
            
        except Exception as e:
            logger.error(f"Error calculating learning goal alignment: {str(e)}")
            return 0.5
    
    def _assign_group_roles(self, group: List[PeerProfile]) -> Dict[str, str]:
        """Assign recommended roles to group members"""
        try:
            roles = {}
            
            # Sort by leadership tendency and expertise
            sorted_members = sorted(
                group, 
                key=lambda x: (x.personality_traits.get('leadership', 0), 
                             max(x.expertise_levels.values()) if x.expertise_levels else 0),
                reverse=True
            )
            
            role_assignments = ['leader', 'facilitator', 'contributor', 'supporter']
            
            for i, member in enumerate(sorted_members):
                if i < len(role_assignments):
                    roles[member.user_id] = role_assignments[i]
                else:
                    roles[member.user_id] = 'contributor'
            
            return roles
            
        except Exception as e:
            logger.error(f"Error assigning group roles: {str(e)}")
            return {member.user_id: 'contributor' for member in group}
    
    def _recommend_group_activities(self, group: List[PeerProfile], 
                                  learning_goal: LearningGoal) -> List[str]:
        """Recommend activities for group"""
        try:
            activities = []
            
            if learning_goal == LearningGoal.SKILL_DEVELOPMENT:
                activities.extend([
                    "Skill-building workshops",
                    "Peer teaching sessions",
                    "Practice projects",
                    "Skill assessment and feedback"
                ])
            elif learning_goal == LearningGoal.PROJECT_COLLABORATION:
                activities.extend([
                    "Project planning sessions",
                    "Collaborative development",
                    "Regular progress reviews",
                    "Final presentation preparation"
                ])
            elif learning_goal == LearningGoal.RESEARCH_COLLABORATION:
                activities.extend([
                    "Research planning and design",
                    "Data collection and analysis",
                    "Literature review sessions",
                    "Research presentation preparation"
                ])
            else:
                activities.extend([
                    "Group discussions",
                    "Knowledge sharing sessions",
                    "Collaborative learning activities",
                    "Peer support and mentoring"
                ])
            
            return activities
            
        except Exception as e:
            logger.error(f"Error recommending group activities: {str(e)}")
            return ["Collaborative learning activities"]
    
    def _calculate_group_success_probability(self, group: List[PeerProfile], 
                                           cohesion: float, diversity: float, 
                                           goal_alignment: float) -> float:
        """Calculate group success probability"""
        try:
            # Base success factors
            success_factors = [cohesion, diversity, goal_alignment]
            
            # Group size factor (optimal around 4-5 members)
            group_size = len(group)
            size_factor = 1.0 - abs(group_size - 4.5) / 4.5
            success_factors.append(size_factor)
            
            # Calculate success probability
            success_probability = np.mean(success_factors)
            
            return max(0.0, min(1.0, success_probability))
            
        except Exception as e:
            logger.error(f"Error calculating group success probability: {str(e)}")
            return 0.5
    
    def _initialize_learning_style_compatibility(self) -> Dict[Tuple[str, str], float]:
        """Initialize learning style compatibility matrix"""
        styles = ['visual', 'auditory', 'kinesthetic', 'reading_writing']
        
        compatibility = {}
        for style1 in styles:
            for style2 in styles:
                if style1 == style2:
                    compatibility[(style1, style2)] = 0.6  # Similar but not identical
                else:
                    # Define compatibility based on learning theory
                    if (style1, style2) in [('visual', 'reading_writing'), ('reading_writing', 'visual')]:
                        compatibility[(style1, style2)] = 0.8  # High compatibility
                    elif (style1, style2) in [('auditory', 'kinesthetic'), ('kinesthetic', 'auditory')]:
                        compatibility[(style1, style2)] = 0.7  # Good compatibility
                    else:
                        compatibility[(style1, style2)] = 0.5  # Moderate compatibility
        
        return compatibility
    
    def track_match_success(self, match: PeerMatch, success_metrics: Dict[str, float]):
        """Track success of peer matches for learning"""
        try:
            success_entry = {
                'match_id': f"{match.peer1_id}_{match.peer2_id}",
                'timestamp': datetime.now().isoformat(),
                'compatibility_score': match.compatibility_score,
                'strategy': match.matching_strategy.value,
                'success_metrics': success_metrics
            }
            
            self.matching_history.append(success_entry)
            
            # Update success rates by strategy
            strategy = match.matching_strategy.value
            if strategy not in self.success_rates:
                self.success_rates[strategy] = []
            
            avg_success = np.mean(list(success_metrics.values()))
            self.success_rates[strategy].append(avg_success)
            
        except Exception as e:
            logger.error(f"Error tracking match success: {str(e)}")
    
    def get_matching_statistics(self) -> Dict[str, float]:
        """Get matching statistics from historical data"""
        if not self.matching_history:
            return {
                'average_compatibility': 0.0,
                'most_effective_strategy': 'complementary',
                'success_rate': 0.0
            }
        
        # Calculate statistics
        compatibility_scores = [entry['compatibility_score'] for entry in self.matching_history]
        avg_compatibility = np.mean(compatibility_scores)
        
        # Most effective strategy
        most_effective = 'complementary'
        if self.success_rates:
            strategy_avg_success = {k: np.mean(v) for k, v in self.success_rates.items()}
            most_effective = max(strategy_avg_success.items(), key=lambda x: x[1])[0]
        
        # Overall success rate
        all_success_rates = []
        for rates in self.success_rates.values():
            all_success_rates.extend(rates)
        success_rate = np.mean(all_success_rates) if all_success_rates else 0.0
        
        return {
            'average_compatibility': float(avg_compatibility),
            'most_effective_strategy': most_effective,
            'success_rate': float(success_rate)
        }
