"""
Career Progression Model Module

This module provides career progression modeling and milestone tracking.

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

class ProgressionStage(Enum):
    """Career progression stages"""
    EXPLORATION = "exploration"
    FOUNDATION = "foundation"
    GROWTH = "growth"
    MASTERY = "mastery"
    LEADERSHIP = "leadership"
    EXPERTISE = "expertise"

@dataclass
class CareerMilestone:
    """Career milestone data structure"""
    milestone_id: str
    title: str
    description: str
    stage: ProgressionStage
    required_skills: List[str]
    time_estimate: int  # months
    success_metrics: List[str]
    rewards: List[str]
    prerequisites: List[str]

@dataclass
class CareerProgressionModel:
    """Career progression model data"""
    user_id: str
    current_stage: ProgressionStage
    target_stage: ProgressionStage
    milestones: List[CareerMilestone]
    progress_percentage: float
    estimated_completion: datetime
    next_milestone: Optional[CareerMilestone]

class CareerProgressionModel:
    """
    Advanced career progression modeling system
    """
    
    def __init__(self):
        """Initialize career progression model"""
        self.progression_models = {}
        self.milestone_templates = {}
        
        # Initialize milestone templates
        self._initialize_milestone_templates()
        
        logger.info("Career Progression Model initialized")
    
    def create_progression_model(self, user_id: str, current_stage: ProgressionStage,
                               target_stage: ProgressionStage) -> CareerProgressionModel:
        """
        Create personalized career progression model
        
        Args:
            user_id: User identifier
            current_stage: Current career stage
            target_stage: Target career stage
            
        Returns:
            Career progression model
        """
        try:
            # Get milestones for progression
            milestones = self._get_milestones_for_progression(current_stage, target_stage)
            
            # Calculate progress
            progress_percentage = self._calculate_initial_progress(current_stage, target_stage)
            
            # Estimate completion time
            total_time = sum(milestone.time_estimate for milestone in milestones)
            estimated_completion = datetime.now() + timedelta(days=total_time * 30)
            
            # Get next milestone
            next_milestone = milestones[0] if milestones else None
            
            model = CareerProgressionModel(
                user_id=user_id,
                current_stage=current_stage,
                target_stage=target_stage,
                milestones=milestones,
                progress_percentage=progress_percentage,
                estimated_completion=estimated_completion,
                next_milestone=next_milestone
            )
            
            self.progression_models[user_id] = model
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating progression model: {str(e)}")
            return None
    
    def update_progress(self, user_id: str, milestone_id: str, 
                       completion_percentage: float) -> Dict[str, Any]:
        """
        Update career progression progress
        
        Args:
            user_id: User identifier
            milestone_id: Milestone identifier
            completion_percentage: Completion percentage (0-100)
            
        Returns:
            Updated progress data
        """
        try:
            if user_id not in self.progression_models:
                return {'error': 'Progression model not found'}
            
            model = self.progression_models[user_id]
            
            # Update milestone progress
            for milestone in model.milestones:
                if milestone.milestone_id == milestone_id:
                    milestone.completion_percentage = completion_percentage
                    break
            
            # Recalculate overall progress
            model.progress_percentage = self._calculate_overall_progress(model)
            
            # Update next milestone
            model.next_milestone = self._get_next_milestone(model)
            
            return {
                'user_id': user_id,
                'progress_percentage': model.progress_percentage,
                'next_milestone': model.next_milestone.title if model.next_milestone else None,
                'estimated_completion': model.estimated_completion.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating progress: {str(e)}")
            return {'error': str(e)}
    
    def _initialize_milestone_templates(self):
        """Initialize milestone templates for different career stages"""
        try:
            self.milestone_templates = {
                ProgressionStage.EXPLORATION: [
                    CareerMilestone(
                        milestone_id="explore_1",
                        title="Identify Career Interests",
                        description="Explore different career paths and identify interests",
                        stage=ProgressionStage.EXPLORATION,
                        required_skills=['self_assessment', 'research'],
                        time_estimate=2,
                        success_metrics=['Complete career assessment', 'Research 5+ career paths'],
                        rewards=['Career clarity', 'Direction setting'],
                        prerequisites=[]
                    )
                ],
                ProgressionStage.FOUNDATION: [
                    CareerMilestone(
                        milestone_id="foundation_1",
                        title="Build Core Skills",
                        description="Develop fundamental skills for chosen career path",
                        stage=ProgressionStage.FOUNDATION,
                        required_skills=['basic_programming', 'communication'],
                        time_estimate=6,
                        success_metrics=['Complete skill courses', 'Build first project'],
                        rewards=['Skill certification', 'Portfolio piece'],
                        prerequisites=['career_identification']
                    )
                ],
                ProgressionStage.GROWTH: [
                    CareerMilestone(
                        milestone_id="growth_1",
                        title="Gain Professional Experience",
                        description="Build professional experience in target field",
                        stage=ProgressionStage.GROWTH,
                        required_skills=['professional_skills', 'industry_knowledge'],
                        time_estimate=12,
                        success_metrics=['Complete internship/project', 'Network with professionals'],
                        rewards=['Professional network', 'Industry experience'],
                        prerequisites=['core_skills']
                    )
                ]
            }
            
            logger.info("Milestone templates initialized")
            
        except Exception as e:
            logger.error(f"Error initializing milestone templates: {str(e)}")
    
    def _get_milestones_for_progression(self, current_stage: ProgressionStage, 
                                      target_stage: ProgressionStage) -> List[CareerMilestone]:
        """Get milestones for career progression"""
        try:
            stage_order = [
                ProgressionStage.EXPLORATION,
                ProgressionStage.FOUNDATION,
                ProgressionStage.GROWTH,
                ProgressionStage.MASTERY,
                ProgressionStage.LEADERSHIP,
                ProgressionStage.EXPERTISE
            ]
            
            current_index = stage_order.index(current_stage)
            target_index = stage_order.index(target_stage)
            
            milestones = []
            
            for i in range(current_index, target_index + 1):
                stage = stage_order[i]
                stage_milestones = self.milestone_templates.get(stage, [])
                milestones.extend(stage_milestones)
            
            return milestones
            
        except Exception as e:
            logger.error(f"Error getting milestones for progression: {str(e)}")
            return []
    
    def _calculate_initial_progress(self, current_stage: ProgressionStage, 
                                  target_stage: ProgressionStage) -> float:
        """Calculate initial progress percentage"""
        try:
            stage_order = [
                ProgressionStage.EXPLORATION,
                ProgressionStage.FOUNDATION,
                ProgressionStage.GROWTH,
                ProgressionStage.MASTERY,
                ProgressionStage.LEADERSHIP,
                ProgressionStage.EXPERTISE
            ]
            
            current_index = stage_order.index(current_stage)
            target_index = stage_order.index(target_stage)
            
            if current_index >= target_index:
                return 100.0
            
            return (current_index / target_index) * 100
            
        except Exception as e:
            logger.error(f"Error calculating initial progress: {str(e)}")
            return 0.0
    
    def _calculate_overall_progress(self, model: CareerProgressionModel) -> float:
        """Calculate overall progress percentage"""
        try:
            if not model.milestones:
                return 0.0
            
            total_progress = sum(
                getattr(milestone, 'completion_percentage', 0) 
                for milestone in model.milestones
            )
            
            return total_progress / len(model.milestones)
            
        except Exception as e:
            logger.error(f"Error calculating overall progress: {str(e)}")
            return 0.0
    
    def _get_next_milestone(self, model: CareerProgressionModel) -> Optional[CareerMilestone]:
        """Get next milestone to work on"""
        try:
            for milestone in model.milestones:
                completion = getattr(milestone, 'completion_percentage', 0)
                if completion < 100:
                    return milestone
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting next milestone: {str(e)}")
            return None
