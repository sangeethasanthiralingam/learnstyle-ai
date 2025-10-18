"""
Skill Gap Analyzer Module

This module provides comprehensive skill gap analysis and recommendations.

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import joblib

logger = logging.getLogger(__name__)

class SkillCategory(Enum):
    """Skill categories"""
    TECHNICAL = "technical"
    SOFT_SKILLS = "soft_skills"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    TOOLS_AND_TECHNOLOGIES = "tools_and_technologies"
    LEADERSHIP = "leadership"
    COMMUNICATION = "communication"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"

class SkillLevel(Enum):
    """Skill proficiency levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class LearningMethod(Enum):
    """Learning methods for skill development"""
    ONLINE_COURSES = "online_courses"
    BOOTCAMPS = "bootcamps"
    CERTIFICATIONS = "certifications"
    PROJECTS = "projects"
    MENTORSHIP = "mentorship"
    WORKSHOPS = "workshops"
    SELF_STUDY = "self_study"
    PRACTICAL_EXPERIENCE = "practical_experience"

@dataclass
class SkillGap:
    """Skill gap data structure"""
    skill_name: str
    category: SkillCategory
    current_level: SkillLevel
    target_level: SkillLevel
    importance_score: float
    difficulty_score: float
    time_to_develop: int  # weeks
    learning_methods: List[LearningMethod]
    resources: List[str]
    prerequisites: List[str]
    related_skills: List[str]
    market_demand: float
    salary_impact: float

@dataclass
class SkillDevelopmentPlan:
    """Skill development plan"""
    plan_id: str
    user_id: str
    target_role: str
    total_skills: int
    skills_to_develop: int
    estimated_duration: int  # weeks
    priority_skills: List[str]
    learning_path: List[Dict[str, Any]]
    milestones: List[Dict[str, Any]]
    success_metrics: List[str]
    created_at: datetime

class SkillGapAnalyzer:
    """
    Advanced skill gap analysis system
    """
    
    def __init__(self):
        """Initialize skill gap analyzer"""
        self.skill_database = {}
        self.skill_relationships = {}
        self.learning_resources = {}
        self.market_data = {}
        
        # Initialize with sample data
        self._initialize_skill_database()
        self._initialize_learning_resources()
        self._initialize_market_data()
        
        logger.info("Skill Gap Analyzer initialized")
    
    def analyze_skill_gaps(self, current_skills: Dict[str, SkillLevel], 
                          target_skills: Dict[str, SkillLevel],
                          target_role: str = "Software Developer") -> List[SkillGap]:
        """
        Analyze skill gaps between current and target skill sets
        
        Args:
            current_skills: Current skill levels
            target_skills: Target skill levels
            target_role: Target role for context
            
        Returns:
            List of skill gaps with recommendations
        """
        try:
            skill_gaps = []
            
            # Find missing skills and skill level gaps
            all_skills = set(current_skills.keys()) | set(target_skills.keys())
            
            for skill in all_skills:
                current_level = current_skills.get(skill, SkillLevel.BEGINNER)
                target_level = target_skills.get(skill, SkillLevel.BEGINNER)
                
                # Skip if already at target level or above
                if self._is_level_sufficient(current_level, target_level):
                    continue
                
                # Get skill information
                skill_info = self.skill_database.get(skill, {})
                
                # Calculate gap metrics
                importance_score = self._calculate_importance_score(skill, target_role, skill_info)
                difficulty_score = self._calculate_difficulty_score(skill, current_level, target_level, skill_info)
                time_to_develop = self._calculate_development_time(skill, current_level, target_level, skill_info)
                
                # Get learning recommendations
                learning_methods = self._recommend_learning_methods(skill, current_level, target_level, skill_info)
                resources = self._recommend_resources(skill, learning_methods)
                prerequisites = self._identify_prerequisites(skill, skill_info)
                related_skills = self._find_related_skills(skill)
                
                # Get market data
                market_demand = self._get_market_demand(skill)
                salary_impact = self._calculate_salary_impact(skill, target_level)
                
                # Create skill gap
                skill_gap = SkillGap(
                    skill_name=skill,
                    category=SkillCategory(skill_info.get('category', 'technical')),
                    current_level=current_level,
                    target_level=target_level,
                    importance_score=importance_score,
                    difficulty_score=difficulty_score,
                    time_to_develop=time_to_develop,
                    learning_methods=learning_methods,
                    resources=resources,
                    prerequisites=prerequisites,
                    related_skills=related_skills,
                    market_demand=market_demand,
                    salary_impact=salary_impact
                )
                
                skill_gaps.append(skill_gap)
            
            # Sort by importance and difficulty
            skill_gaps.sort(key=lambda x: (x.importance_score, -x.difficulty_score), reverse=True)
            
            return skill_gaps
            
        except Exception as e:
            logger.error(f"Error analyzing skill gaps: {str(e)}")
            return []
    
    def create_development_plan(self, user_id: str, skill_gaps: List[SkillGap], 
                              target_role: str, max_duration_weeks: int = 52) -> SkillDevelopmentPlan:
        """
        Create personalized skill development plan
        
        Args:
            user_id: User identifier
            skill_gaps: List of skill gaps to address
            target_role: Target role
            max_duration_weeks: Maximum plan duration
            
        Returns:
            Personalized development plan
        """
        try:
            plan_id = f"plan_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prioritize skills based on importance and feasibility
            prioritized_skills = self._prioritize_skills(skill_gaps, max_duration_weeks)
            
            # Create learning path
            learning_path = self._create_learning_path(prioritized_skills, max_duration_weeks)
            
            # Create milestones
            milestones = self._create_milestones(learning_path, prioritized_skills)
            
            # Define success metrics
            success_metrics = self._define_success_metrics(prioritized_skills, target_role)
            
            # Calculate total duration
            total_duration = sum(phase['duration_weeks'] for phase in learning_path)
            
            plan = SkillDevelopmentPlan(
                plan_id=plan_id,
                user_id=user_id,
                target_role=target_role,
                total_skills=len(skill_gaps),
                skills_to_develop=len(prioritized_skills),
                estimated_duration=min(total_duration, max_duration_weeks),
                priority_skills=[gap.skill_name for gap in prioritized_skills],
                learning_path=learning_path,
                milestones=milestones,
                success_metrics=success_metrics,
                created_at=datetime.now()
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating development plan: {str(e)}")
            return None
    
    def recommend_skill_priorities(self, skill_gaps: List[SkillGap], 
                                 learning_style: str = "visual",
                                 available_time_hours: int = 10) -> List[SkillGap]:
        """
        Recommend skill development priorities
        
        Args:
            skill_gaps: List of skill gaps
            learning_style: User's learning style
            available_time_hours: Available learning time per week
            
        Returns:
            Prioritized list of skill gaps
        """
        try:
            # Calculate priority scores for each skill gap
            for gap in skill_gaps:
                # Base priority from importance and market demand
                base_priority = gap.importance_score * 0.6 + gap.market_demand * 0.4
                
                # Adjust for learning style compatibility
                style_factor = self._calculate_learning_style_compatibility(
                    gap.skill_name, learning_style
                )
                
                # Adjust for time feasibility
                time_factor = self._calculate_time_feasibility(
                    gap.time_to_develop, available_time_hours
                )
                
                # Adjust for difficulty
                difficulty_factor = 1.0 - (gap.difficulty_score * 0.3)
                
                # Calculate final priority score
                gap.priority_score = base_priority * style_factor * time_factor * difficulty_factor
            
            # Sort by priority score
            prioritized_gaps = sorted(skill_gaps, key=lambda x: x.priority_score, reverse=True)
            
            return prioritized_gaps
            
        except Exception as e:
            logger.error(f"Error recommending skill priorities: {str(e)}")
            return skill_gaps
    
    def track_skill_progress(self, user_id: str, skill_name: str, 
                           current_level: SkillLevel, 
                           target_level: SkillLevel) -> Dict[str, Any]:
        """
        Track skill development progress
        
        Args:
            user_id: User identifier
            skill_name: Skill being developed
            current_level: Current skill level
            target_level: Target skill level
            
        Returns:
            Progress tracking data
        """
        try:
            # Calculate progress percentage
            level_values = {
                SkillLevel.BEGINNER: 1,
                SkillLevel.INTERMEDIATE: 2,
                SkillLevel.ADVANCED: 3,
                SkillLevel.EXPERT: 4
            }
            
            current_value = level_values[current_level]
            target_value = level_values[target_level]
            progress_percentage = ((current_value - 1) / (target_value - 1)) * 100
            
            # Estimate time remaining
            skill_info = self.skill_database.get(skill_name, {})
            total_time = self._calculate_development_time(skill_name, SkillLevel.BEGINNER, target_level, skill_info)
            remaining_time = total_time * (1 - progress_percentage / 100)
            
            # Get next milestones
            next_milestones = self._get_next_milestones(skill_name, current_level, target_level)
            
            # Calculate confidence score
            confidence = self._calculate_progress_confidence(progress_percentage, skill_name)
            
            return {
                'skill_name': skill_name,
                'current_level': current_level.value,
                'target_level': target_level.value,
                'progress_percentage': min(100, max(0, progress_percentage)),
                'estimated_remaining_weeks': max(1, int(remaining_time)),
                'next_milestones': next_milestones,
                'confidence_score': confidence,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error tracking skill progress: {str(e)}")
            return {}
    
    def _initialize_skill_database(self):
        """Initialize skill database with sample data"""
        try:
            self.skill_database = {
                'python': {
                    'category': 'technical',
                    'difficulty': 0.6,
                    'prerequisites': ['programming_basics'],
                    'related_skills': ['data_analysis', 'web_development', 'machine_learning'],
                    'learning_time': {'beginner': 8, 'intermediate': 16, 'advanced': 24, 'expert': 40}
                },
                'machine_learning': {
                    'category': 'technical',
                    'difficulty': 0.8,
                    'prerequisites': ['python', 'statistics', 'linear_algebra'],
                    'related_skills': ['data_science', 'deep_learning', 'ai'],
                    'learning_time': {'beginner': 12, 'intermediate': 24, 'advanced': 36, 'expert': 52}
                },
                'project_management': {
                    'category': 'soft_skills',
                    'difficulty': 0.5,
                    'prerequisites': ['communication', 'organization'],
                    'related_skills': ['leadership', 'agile', 'scrum'],
                    'learning_time': {'beginner': 6, 'intermediate': 12, 'advanced': 18, 'expert': 24}
                },
                'data_analysis': {
                    'category': 'analytical',
                    'difficulty': 0.7,
                    'prerequisites': ['statistics', 'excel'],
                    'related_skills': ['python', 'sql', 'visualization'],
                    'learning_time': {'beginner': 10, 'intermediate': 20, 'advanced': 30, 'expert': 40}
                },
                'leadership': {
                    'category': 'leadership',
                    'difficulty': 0.6,
                    'prerequisites': ['communication', 'emotional_intelligence'],
                    'related_skills': ['team_management', 'strategic_thinking', 'decision_making'],
                    'learning_time': {'beginner': 8, 'intermediate': 16, 'advanced': 24, 'expert': 32}
                }
            }
            
            logger.info("Skill database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing skill database: {str(e)}")
    
    def _initialize_learning_resources(self):
        """Initialize learning resources database"""
        try:
            self.learning_resources = {
                'python': {
                    'online_courses': [
                        'Python for Everybody (Coursera)',
                        'Complete Python Bootcamp (Udemy)',
                        'Python Data Science Handbook (O\'Reilly)'
                    ],
                    'projects': [
                        'Build a web scraper',
                        'Create a data analysis dashboard',
                        'Develop a machine learning model'
                    ],
                    'certifications': [
                        'PCAP - Certified Associate in Python Programming',
                        'PCPP - Certified Professional in Python Programming'
                    ]
                },
                'machine_learning': {
                    'online_courses': [
                        'Machine Learning (Coursera)',
                        'Deep Learning Specialization (Coursera)',
                        'Machine Learning A-Z (Udemy)'
                    ],
                    'projects': [
                        'Image classification project',
                        'Natural language processing project',
                        'Recommendation system project'
                    ],
                    'certifications': [
                        'Google Cloud Professional ML Engineer',
                        'AWS Certified Machine Learning - Specialty'
                    ]
                }
            }
            
            logger.info("Learning resources initialized")
            
        except Exception as e:
            logger.error(f"Error initializing learning resources: {str(e)}")
    
    def _initialize_market_data(self):
        """Initialize market data for skills"""
        try:
            self.market_data = {
                'python': {'demand': 0.9, 'salary_impact': 0.8, 'growth_rate': 0.15},
                'machine_learning': {'demand': 0.95, 'salary_impact': 0.9, 'growth_rate': 0.25},
                'data_analysis': {'demand': 0.85, 'salary_impact': 0.7, 'growth_rate': 0.12},
                'project_management': {'demand': 0.8, 'salary_impact': 0.6, 'growth_rate': 0.08},
                'leadership': {'demand': 0.9, 'salary_impact': 0.85, 'growth_rate': 0.10}
            }
            
            logger.info("Market data initialized")
            
        except Exception as e:
            logger.error(f"Error initializing market data: {str(e)}")
    
    def _is_level_sufficient(self, current: SkillLevel, target: SkillLevel) -> bool:
        """Check if current level meets or exceeds target level"""
        level_order = [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, SkillLevel.ADVANCED, SkillLevel.EXPERT]
        return level_order.index(current) >= level_order.index(target)
    
    def _calculate_importance_score(self, skill: str, target_role: str, skill_info: Dict[str, Any]) -> float:
        """Calculate importance score for skill in target role"""
        try:
            # Base importance from skill database
            base_importance = skill_info.get('importance', 0.5)
            
            # Role-specific importance mapping
            role_importance = {
                'Software Developer': {'python': 0.9, 'machine_learning': 0.7, 'project_management': 0.5},
                'Data Scientist': {'python': 0.95, 'machine_learning': 0.9, 'data_analysis': 0.9},
                'Product Manager': {'project_management': 0.9, 'leadership': 0.8, 'data_analysis': 0.6}
            }
            
            role_factor = role_importance.get(target_role, {}).get(skill, 0.5)
            
            return (base_importance + role_factor) / 2
            
        except Exception as e:
            logger.error(f"Error calculating importance score: {str(e)}")
            return 0.5
    
    def _calculate_difficulty_score(self, skill: str, current_level: SkillLevel, 
                                  target_level: SkillLevel, skill_info: Dict[str, Any]) -> float:
        """Calculate difficulty score for skill development"""
        try:
            # Base difficulty from skill database
            base_difficulty = skill_info.get('difficulty', 0.5)
            
            # Level gap difficulty
            level_values = {
                SkillLevel.BEGINNER: 1,
                SkillLevel.INTERMEDIATE: 2,
                SkillLevel.ADVANCED: 3,
                SkillLevel.EXPERT: 4
            }
            
            current_value = level_values[current_level]
            target_value = level_values[target_level]
            level_gap = target_value - current_value
            
            # Difficulty increases exponentially with level gap
            gap_difficulty = min(1.0, level_gap / 3.0)
            
            return (base_difficulty + gap_difficulty) / 2
            
        except Exception as e:
            logger.error(f"Error calculating difficulty score: {str(e)}")
            return 0.5
    
    def _calculate_development_time(self, skill: str, current_level: SkillLevel, 
                                 target_level: SkillLevel, skill_info: Dict[str, Any]) -> int:
        """Calculate time to develop skill to target level"""
        try:
            learning_times = skill_info.get('learning_time', {})
            
            if current_level == target_level:
                return 0
            
            # Calculate time based on level progression
            level_values = {
                SkillLevel.BEGINNER: 1,
                SkillLevel.INTERMEDIATE: 2,
                SkillLevel.ADVANCED: 3,
                SkillLevel.EXPERT: 4
            }
            
            current_value = level_values[current_level]
            target_value = level_values[target_level]
            
            total_time = 0
            for level in range(current_value, target_value):
                level_name = list(level_values.keys())[level - 1]
                time_for_level = learning_times.get(level_name, 8)
                total_time += time_for_level
            
            return max(1, total_time)
            
        except Exception as e:
            logger.error(f"Error calculating development time: {str(e)}")
            return 8
    
    def _recommend_learning_methods(self, skill: str, current_level: SkillLevel, 
                                  target_level: SkillLevel, skill_info: Dict[str, Any]) -> List[LearningMethod]:
        """Recommend learning methods for skill development"""
        try:
            methods = []
            
            # Base methods for all skills
            methods.extend([LearningMethod.ONLINE_COURSES, LearningMethod.PROJECTS])
            
            # Add methods based on skill category
            category = skill_info.get('category', 'technical')
            if category == 'technical':
                methods.extend([LearningMethod.BOOTCAMPS, LearningMethod.CERTIFICATIONS])
            elif category == 'soft_skills':
                methods.extend([LearningMethod.MENTORSHIP, LearningMethod.WORKSHOPS])
            elif category == 'leadership':
                methods.extend([LearningMethod.MENTORSHIP, LearningMethod.PRACTICAL_EXPERIENCE])
            
            # Add methods based on target level
            if target_level in [SkillLevel.ADVANCED, SkillLevel.EXPERT]:
                methods.extend([LearningMethod.MENTORSHIP, LearningMethod.PRACTICAL_EXPERIENCE])
            
            return list(set(methods))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error recommending learning methods: {str(e)}")
            return [LearningMethod.ONLINE_COURSES, LearningMethod.PROJECTS]
    
    def _recommend_resources(self, skill: str, learning_methods: List[LearningMethod]) -> List[str]:
        """Recommend specific learning resources"""
        try:
            resources = []
            skill_resources = self.learning_resources.get(skill, {})
            
            for method in learning_methods:
                method_resources = skill_resources.get(method.value, [])
                resources.extend(method_resources)
            
            return resources[:5]  # Return top 5 resources
            
        except Exception as e:
            logger.error(f"Error recommending resources: {str(e)}")
            return []
    
    def _identify_prerequisites(self, skill: str, skill_info: Dict[str, Any]) -> List[str]:
        """Identify prerequisite skills"""
        try:
            return skill_info.get('prerequisites', [])
        except Exception as e:
            logger.error(f"Error identifying prerequisites: {str(e)}")
            return []
    
    def _find_related_skills(self, skill: str) -> List[str]:
        """Find related skills"""
        try:
            skill_info = self.skill_database.get(skill, {})
            return skill_info.get('related_skills', [])
        except Exception as e:
            logger.error(f"Error finding related skills: {str(e)}")
            return []
    
    def _get_market_demand(self, skill: str) -> float:
        """Get market demand for skill"""
        try:
            return self.market_data.get(skill, {}).get('demand', 0.5)
        except Exception as e:
            logger.error(f"Error getting market demand: {str(e)}")
            return 0.5
    
    def _calculate_salary_impact(self, skill: str, target_level: SkillLevel) -> float:
        """Calculate salary impact of skill"""
        try:
            base_impact = self.market_data.get(skill, {}).get('salary_impact', 0.5)
            
            # Adjust based on skill level
            level_multiplier = {
                SkillLevel.BEGINNER: 0.5,
                SkillLevel.INTERMEDIATE: 0.75,
                SkillLevel.ADVANCED: 1.0,
                SkillLevel.EXPERT: 1.25
            }
            
            return base_impact * level_multiplier.get(target_level, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating salary impact: {str(e)}")
            return 0.5
    
    def _prioritize_skills(self, skill_gaps: List[SkillGap], max_duration_weeks: int) -> List[SkillGap]:
        """Prioritize skills based on importance and feasibility"""
        try:
            # Calculate priority scores
            for gap in skill_gaps:
                # Priority based on importance, market demand, and feasibility
                priority = (
                    gap.importance_score * 0.4 +
                    gap.market_demand * 0.3 +
                    (1 - gap.difficulty_score) * 0.2 +
                    gap.salary_impact * 0.1
                )
                gap.priority_score = priority
            
            # Sort by priority
            prioritized = sorted(skill_gaps, key=lambda x: x.priority_score, reverse=True)
            
            # Filter by time constraints
            total_time = 0
            selected_skills = []
            
            for gap in prioritized:
                if total_time + gap.time_to_develop <= max_duration_weeks:
                    selected_skills.append(gap)
                    total_time += gap.time_to_develop
                else:
                    break
            
            return selected_skills
            
        except Exception as e:
            logger.error(f"Error prioritizing skills: {str(e)}")
            return skill_gaps[:5]  # Return top 5 by default
    
    def _create_learning_path(self, prioritized_skills: List[SkillGap], 
                            max_duration_weeks: int) -> List[Dict[str, Any]]:
        """Create structured learning path"""
        try:
            learning_path = []
            current_week = 0
            
            # Group skills into phases
            phases = self._group_skills_into_phases(prioritized_skills)
            
            for phase_idx, phase_skills in enumerate(phases):
                phase_duration = sum(skill.time_to_develop for skill in phase_skills)
                
                learning_path.append({
                    'phase': phase_idx + 1,
                    'phase_name': f'Phase {phase_idx + 1}',
                    'skills': [skill.skill_name for skill in phase_skills],
                    'duration_weeks': phase_duration,
                    'start_week': current_week + 1,
                    'end_week': current_week + phase_duration,
                    'learning_methods': list(set(
                        method for skill in phase_skills 
                        for method in skill.learning_methods
                    )),
                    'focus_areas': self._identify_focus_areas(phase_skills)
                })
                
                current_week += phase_duration
                
                if current_week >= max_duration_weeks:
                    break
            
            return learning_path
            
        except Exception as e:
            logger.error(f"Error creating learning path: {str(e)}")
            return []
    
    def _group_skills_into_phases(self, skills: List[SkillGap]) -> List[List[SkillGap]]:
        """Group skills into learning phases"""
        try:
            phases = []
            current_phase = []
            current_phase_time = 0
            max_phase_time = 12  # Maximum 12 weeks per phase
            
            for skill in skills:
                if current_phase_time + skill.time_to_develop <= max_phase_time:
                    current_phase.append(skill)
                    current_phase_time += skill.time_to_develop
                else:
                    if current_phase:
                        phases.append(current_phase)
                    current_phase = [skill]
                    current_phase_time = skill.time_to_develop
            
            if current_phase:
                phases.append(current_phase)
            
            return phases
            
        except Exception as e:
            logger.error(f"Error grouping skills into phases: {str(e)}")
            return [skills]  # Return all skills in one phase
    
    def _identify_focus_areas(self, skills: List[SkillGap]) -> List[str]:
        """Identify focus areas for a phase"""
        try:
            categories = [skill.category.value for skill in skills]
            category_counts = {}
            
            for category in categories:
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Return top 2 categories
            sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            return [cat[0] for cat in sorted_categories[:2]]
            
        except Exception as e:
            logger.error(f"Error identifying focus areas: {str(e)}")
            return ['technical']
    
    def _create_milestones(self, learning_path: List[Dict[str, Any]], 
                          prioritized_skills: List[SkillGap]) -> List[Dict[str, Any]]:
        """Create learning milestones"""
        try:
            milestones = []
            
            for phase in learning_path:
                milestone = {
                    'milestone_id': f"milestone_{phase['phase']}",
                    'name': f"Complete {phase['phase_name']}",
                    'description': f"Master skills: {', '.join(phase['skills'])}",
                    'target_week': phase['end_week'],
                    'skills_completed': phase['skills'],
                    'success_criteria': self._define_success_criteria(phase['skills']),
                    'rewards': self._define_milestone_rewards(phase['phase'])
                }
                milestones.append(milestone)
            
            return milestones
            
        except Exception as e:
            logger.error(f"Error creating milestones: {str(e)}")
            return []
    
    def _define_success_criteria(self, skills: List[str]) -> List[str]:
        """Define success criteria for skills"""
        try:
            criteria = []
            for skill in skills:
                criteria.append(f"Demonstrate {skill} proficiency through practical project")
                criteria.append(f"Complete {skill} assessment with 80%+ score")
            
            return criteria[:6]  # Limit to 6 criteria
            
        except Exception as e:
            logger.error(f"Error defining success criteria: {str(e)}")
            return ["Complete skill assessment"]
    
    def _define_milestone_rewards(self, phase: int) -> List[str]:
        """Define rewards for milestones"""
        try:
            rewards = [
                f"Phase {phase} completion certificate",
                f"Unlock advanced Phase {phase + 1} content",
                "Progress tracking badge"
            ]
            
            if phase >= 2:
                rewards.append("Mentorship opportunity")
            if phase >= 3:
                rewards.append("Industry networking event access")
            
            return rewards
            
        except Exception as e:
            logger.error(f"Error defining milestone rewards: {str(e)}")
            return ["Completion certificate"]
    
    def _define_success_metrics(self, prioritized_skills: List[SkillGap], 
                              target_role: str) -> List[str]:
        """Define success metrics for the development plan"""
        try:
            metrics = [
                f"Complete {len(prioritized_skills)} priority skills",
                f"Demonstrate proficiency in {target_role} role requirements",
                "Achieve 80%+ in skill assessments",
                "Complete 3+ practical projects"
            ]
            
            # Add role-specific metrics
            if 'technical' in [skill.category.value for skill in prioritized_skills]:
                metrics.append("Build portfolio of technical projects")
            
            if 'leadership' in [skill.category.value for skill in prioritized_skills]:
                metrics.append("Lead a team project successfully")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error defining success metrics: {str(e)}")
            return ["Complete skill development plan"]
    
    def _calculate_learning_style_compatibility(self, skill: str, learning_style: str) -> float:
        """Calculate compatibility between skill and learning style"""
        try:
            # Skill-style compatibility mapping
            compatibility = {
                'python': {'visual': 0.9, 'kinesthetic': 0.8, 'analytical': 0.9},
                'machine_learning': {'visual': 0.8, 'analytical': 0.95, 'kinesthetic': 0.7},
                'project_management': {'auditory': 0.8, 'collaborative': 0.9, 'leadership': 0.85},
                'data_analysis': {'visual': 0.9, 'analytical': 0.95, 'kinesthetic': 0.6},
                'leadership': {'auditory': 0.8, 'collaborative': 0.9, 'communication': 0.85}
            }
            
            return compatibility.get(skill, {}).get(learning_style, 0.7)
            
        except Exception as e:
            logger.error(f"Error calculating learning style compatibility: {str(e)}")
            return 0.7
    
    def _calculate_time_feasibility(self, time_to_develop: int, available_time_hours: int) -> float:
        """Calculate time feasibility for skill development"""
        try:
            # Convert weeks to hours (assuming 10 hours per week)
            required_hours = time_to_develop * 10
            available_total_hours = available_time_hours * time_to_develop
            
            if available_total_hours >= required_hours:
                return 1.0
            else:
                return available_total_hours / required_hours
                
        except Exception as e:
            logger.error(f"Error calculating time feasibility: {str(e)}")
            return 0.5
    
    def _get_next_milestones(self, skill: str, current_level: SkillLevel, 
                           target_level: SkillLevel) -> List[str]:
        """Get next milestones for skill development"""
        try:
            level_values = {
                SkillLevel.BEGINNER: 1,
                SkillLevel.INTERMEDIATE: 2,
                SkillLevel.ADVANCED: 3,
                SkillLevel.EXPERT: 4
            }
            
            current_value = level_values[current_level]
            target_value = level_values[target_level]
            
            milestones = []
            
            if current_value < 2:  # Beginner to Intermediate
                milestones.append("Complete basic tutorials and exercises")
                milestones.append("Build first small project")
            
            if current_value < 3:  # Intermediate to Advanced
                milestones.append("Complete intermediate-level projects")
                milestones.append("Contribute to open source projects")
            
            if current_value < 4:  # Advanced to Expert
                milestones.append("Mentor others in the skill")
                milestones.append("Create original content or tools")
            
            return milestones[:3]  # Return top 3 milestones
            
        except Exception as e:
            logger.error(f"Error getting next milestones: {str(e)}")
            return ["Continue practicing"]
    
    def _calculate_progress_confidence(self, progress_percentage: float, skill: str) -> float:
        """Calculate confidence in progress tracking"""
        try:
            # Base confidence on progress percentage
            base_confidence = progress_percentage / 100
            
            # Adjust based on skill difficulty
            skill_info = self.skill_database.get(skill, {})
            difficulty = skill_info.get('difficulty', 0.5)
            difficulty_factor = 1.0 - (difficulty * 0.2)
            
            confidence = base_confidence * difficulty_factor
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating progress confidence: {str(e)}")
            return 0.5
    
    def get_analyzer_statistics(self) -> Dict[str, int]:
        """Get skill gap analyzer statistics"""
        try:
            return {
                'total_skills': len(self.skill_database),
                'learning_resources': sum(len(resources) for resources in self.learning_resources.values()),
                'market_data_points': len(self.market_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting analyzer statistics: {str(e)}")
            return {
                'total_skills': 0,
                'learning_resources': 0,
                'market_data_points': 0
            }
