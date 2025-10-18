"""
Career Path Predictor Module

This module provides career path prediction based on learning styles, skills, and preferences.

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
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

logger = logging.getLogger(__name__)

class CareerStage(Enum):
    """Career development stages"""
    ENTRY_LEVEL = "entry_level"
    JUNIOR = "junior"
    MID_LEVEL = "mid_level"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"
    EXPERT = "expert"

class IndustrySector(Enum):
    """Industry sectors"""
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    EDUCATION = "education"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    CONSULTING = "consulting"
    GOVERNMENT = "government"
    NON_PROFIT = "non_profit"
    ENTERTAINMENT = "entertainment"

class WorkStyle(Enum):
    """Work style preferences"""
    INDEPENDENT = "independent"
    COLLABORATIVE = "collaborative"
    LEADERSHIP = "leadership"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    TECHNICAL = "technical"
    COMMUNICATION = "communication"

@dataclass
class CareerPath:
    """Career path data structure"""
    path_id: str
    title: str
    industry: IndustrySector
    current_stage: CareerStage
    target_stage: CareerStage
    required_skills: List[str]
    learning_style_fit: float
    skill_gaps: List[str]
    time_to_target: int  # months
    salary_range: Tuple[float, float]
    growth_potential: float
    job_market_demand: float
    work_style_match: float
    confidence_score: float

@dataclass
class IndustryTrend:
    """Industry trend data"""
    industry: IndustrySector
    growth_rate: float
    job_demand: float
    skill_demand: Dict[str, float]
    emerging_skills: List[str]
    declining_skills: List[str]
    salary_trends: Dict[CareerStage, float]
    future_outlook: str

class CareerPathPredictor:
    """
    Advanced career path prediction system
    """
    
    def __init__(self):
        """Initialize career path predictor"""
        self.career_paths = {}
        self.industry_trends = {}
        self.skill_models = {}
        self.learning_style_encoder = LabelEncoder()
        self.industry_encoder = LabelEncoder()
        self.career_stage_encoder = LabelEncoder()
        
        # Initialize with sample data
        self._initialize_sample_data()
        
        logger.info("Career Path Predictor initialized")
    
    def predict_career_paths(self, user_profile: Dict[str, Any], 
                           target_industries: Optional[List[IndustrySector]] = None,
                           max_paths: int = 5) -> List[CareerPath]:
        """
        Predict career paths for a user
        
        Args:
            user_profile: User profile with skills, learning style, preferences
            target_industries: Specific industries to focus on
            max_paths: Maximum number of paths to return
            
        Returns:
            List of predicted career paths
        """
        try:
            # Extract user characteristics
            learning_style = user_profile.get('learning_style', 'visual')
            current_skills = user_profile.get('skills', [])
            experience_years = user_profile.get('experience_years', 0)
            work_preferences = user_profile.get('work_preferences', {})
            salary_expectations = user_profile.get('salary_expectations', (50000, 100000))
            
            # Determine target industries
            if not target_industries:
                target_industries = self._recommend_industries(learning_style, work_preferences)
            
            predicted_paths = []
            
            for industry in target_industries:
                # Get industry-specific career paths
                industry_paths = self._get_industry_career_paths(industry)
                
                for path_template in industry_paths:
                    # Calculate fit scores
                    learning_style_fit = self._calculate_learning_style_fit(
                        learning_style, path_template['learning_style_requirements']
                    )
                    
                    skill_gaps = self._identify_skill_gaps(current_skills, path_template['required_skills'])
                    skill_fit = 1.0 - (len(skill_gaps) / len(path_template['required_skills']))
                    
                    work_style_match = self._calculate_work_style_match(
                        work_preferences, path_template['work_style_requirements']
                    )
                    
                    # Calculate time to target stage
                    time_to_target = self._calculate_time_to_target(
                        experience_years, path_template['current_stage'], 
                        path_template['target_stage'], skill_fit
                    )
                    
                    # Calculate confidence score
                    confidence = self._calculate_confidence_score(
                        learning_style_fit, skill_fit, work_style_match, 
                        path_template['job_market_demand']
                    )
                    
                    # Create career path
                    career_path = CareerPath(
                        path_id=f"{industry.value}_{path_template['title']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        title=path_template['title'],
                        industry=industry,
                        current_stage=CareerStage(path_template['current_stage']),
                        target_stage=CareerStage(path_template['target_stage']),
                        required_skills=path_template['required_skills'],
                        learning_style_fit=learning_style_fit,
                        skill_gaps=skill_gaps,
                        time_to_target=time_to_target,
                        salary_range=path_template['salary_range'],
                        growth_potential=path_template['growth_potential'],
                        job_market_demand=path_template['job_market_demand'],
                        work_style_match=work_style_match,
                        confidence_score=confidence
                    )
                    
                    predicted_paths.append(career_path)
            
            # Sort by confidence score and return top paths
            predicted_paths.sort(key=lambda x: x.confidence_score, reverse=True)
            return predicted_paths[:max_paths]
            
        except Exception as e:
            logger.error(f"Error predicting career paths: {str(e)}")
            return []
    
    def analyze_career_progression(self, current_path: CareerPath, 
                                 target_path: CareerPath) -> Dict[str, Any]:
        """
        Analyze progression from current to target career path
        
        Args:
            current_path: Current career path
            target_path: Target career path
            
        Returns:
            Progression analysis results
        """
        try:
            # Calculate skill transition requirements
            skill_transition = self._analyze_skill_transition(
                current_path.required_skills, target_path.required_skills
            )
            
            # Calculate industry transition difficulty
            industry_transition = self._analyze_industry_transition(
                current_path.industry, target_path.industry
            )
            
            # Calculate time and effort requirements
            transition_time = self._calculate_transition_time(
                current_path, target_path, skill_transition, industry_transition
            )
            
            # Calculate success probability
            success_probability = self._calculate_success_probability(
                current_path, target_path, skill_transition, industry_transition
            )
            
            return {
                'transition_feasibility': success_probability > 0.6,
                'success_probability': success_probability,
                'estimated_time_months': transition_time,
                'skill_transition_requirements': skill_transition,
                'industry_transition_difficulty': industry_transition,
                'recommended_steps': self._generate_transition_steps(
                    current_path, target_path, skill_transition
                ),
                'risk_factors': self._identify_risk_factors(
                    current_path, target_path, skill_transition, industry_transition
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing career progression: {str(e)}")
            return {}
    
    def get_industry_trends(self, industry: IndustrySector) -> IndustryTrend:
        """
        Get industry trend analysis
        
        Args:
            industry: Target industry
            
        Returns:
            Industry trend data
        """
        try:
            if industry in self.industry_trends:
                return self.industry_trends[industry]
            
            # Generate industry trend analysis
            trend = self._generate_industry_trend(industry)
            self.industry_trends[industry] = trend
            
            return trend
            
        except Exception as e:
            logger.error(f"Error getting industry trends: {str(e)}")
            return None
    
    def _initialize_sample_data(self):
        """Initialize with sample career path data"""
        try:
            # Technology industry career paths
            tech_paths = [
                {
                    'title': 'Software Developer',
                    'current_stage': 'entry_level',
                    'target_stage': 'senior',
                    'required_skills': ['programming', 'algorithms', 'data_structures', 'software_design', 'testing'],
                    'learning_style_requirements': ['visual', 'kinesthetic'],
                    'work_style_requirements': ['technical', 'analytical', 'independent'],
                    'salary_range': (60000, 120000),
                    'growth_potential': 0.8,
                    'job_market_demand': 0.9
                },
                {
                    'title': 'Data Scientist',
                    'current_stage': 'entry_level',
                    'target_stage': 'senior',
                    'required_skills': ['statistics', 'machine_learning', 'python', 'sql', 'data_visualization'],
                    'learning_style_requirements': ['visual', 'analytical'],
                    'work_style_requirements': ['analytical', 'independent', 'creative'],
                    'salary_range': (70000, 140000),
                    'growth_potential': 0.9,
                    'job_market_demand': 0.95
                },
                {
                    'title': 'Product Manager',
                    'current_stage': 'mid_level',
                    'target_stage': 'lead',
                    'required_skills': ['product_strategy', 'user_research', 'project_management', 'analytics', 'communication'],
                    'learning_style_requirements': ['visual', 'auditory'],
                    'work_style_requirements': ['leadership', 'collaborative', 'communication'],
                    'salary_range': (80000, 150000),
                    'growth_potential': 0.85,
                    'job_market_demand': 0.8
                }
            ]
            
            # Healthcare industry career paths
            healthcare_paths = [
                {
                    'title': 'Clinical Data Analyst',
                    'current_stage': 'entry_level',
                    'target_stage': 'senior',
                    'required_skills': ['healthcare_data', 'statistics', 'sql', 'medical_terminology', 'regulatory_compliance'],
                    'learning_style_requirements': ['visual', 'analytical'],
                    'work_style_requirements': ['analytical', 'independent', 'technical'],
                    'salary_range': (55000, 100000),
                    'growth_potential': 0.75,
                    'job_market_demand': 0.85
                }
            ]
            
            # Finance industry career paths
            finance_paths = [
                {
                    'title': 'Financial Analyst',
                    'current_stage': 'entry_level',
                    'target_stage': 'senior',
                    'required_skills': ['financial_modeling', 'excel', 'accounting', 'economics', 'risk_analysis'],
                    'learning_style_requirements': ['analytical', 'visual'],
                    'work_style_requirements': ['analytical', 'independent', 'technical'],
                    'salary_range': (50000, 110000),
                    'growth_potential': 0.7,
                    'job_market_demand': 0.75
                }
            ]
            
            # Store career paths by industry
            self.career_paths[IndustrySector.TECHNOLOGY] = tech_paths
            self.career_paths[IndustrySector.HEALTHCARE] = healthcare_paths
            self.career_paths[IndustrySector.FINANCE] = finance_paths
            
            logger.info("Sample career path data initialized")
            
        except Exception as e:
            logger.error(f"Error initializing sample data: {str(e)}")
    
    def _recommend_industries(self, learning_style: str, work_preferences: Dict[str, Any]) -> List[IndustrySector]:
        """Recommend industries based on learning style and preferences"""
        try:
            # Industry recommendations based on learning style
            style_industry_mapping = {
                'visual': [IndustrySector.TECHNOLOGY, IndustrySector.EDUCATION, IndustrySector.ENTERTAINMENT],
                'auditory': [IndustrySector.EDUCATION, IndustrySector.CONSULTING, IndustrySector.ENTERTAINMENT],
                'kinesthetic': [IndustrySector.MANUFACTURING, IndustrySector.HEALTHCARE, IndustrySector.TECHNOLOGY],
                'analytical': [IndustrySector.TECHNOLOGY, IndustrySector.FINANCE, IndustrySector.CONSULTING],
                'creative': [IndustrySector.ENTERTAINMENT, IndustrySector.EDUCATION, IndustrySector.TECHNOLOGY]
            }
            
            recommended = style_industry_mapping.get(learning_style, list(IndustrySector))
            
            # Filter based on work preferences
            if work_preferences.get('remote_friendly', False):
                recommended = [ind for ind in recommended if ind in [
                    IndustrySector.TECHNOLOGY, IndustrySector.CONSULTING, 
                    IndustrySector.EDUCATION, IndustrySector.FINANCE
                ]]
            
            return recommended[:3]  # Return top 3 recommendations
            
        except Exception as e:
            logger.error(f"Error recommending industries: {str(e)}")
            return list(IndustrySector)[:3]
    
    def _get_industry_career_paths(self, industry: IndustrySector) -> List[Dict[str, Any]]:
        """Get career paths for specific industry"""
        return self.career_paths.get(industry, [])
    
    def _calculate_learning_style_fit(self, user_style: str, required_styles: List[str]) -> float:
        """Calculate learning style fit score"""
        try:
            if user_style in required_styles:
                return 1.0
            
            # Calculate similarity based on style compatibility
            style_compatibility = {
                'visual': {'analytical': 0.8, 'kinesthetic': 0.6, 'auditory': 0.4},
                'auditory': {'communication': 0.9, 'collaborative': 0.8, 'visual': 0.4},
                'kinesthetic': {'technical': 0.8, 'visual': 0.6, 'analytical': 0.5},
                'analytical': {'technical': 0.9, 'visual': 0.8, 'independent': 0.7},
                'creative': {'communication': 0.7, 'collaborative': 0.8, 'leadership': 0.6}
            }
            
            max_compatibility = 0.0
            for required_style in required_styles:
                compatibility = style_compatibility.get(user_style, {}).get(required_style, 0.3)
                max_compatibility = max(max_compatibility, compatibility)
            
            return max_compatibility
            
        except Exception as e:
            logger.error(f"Error calculating learning style fit: {str(e)}")
            return 0.5
    
    def _identify_skill_gaps(self, current_skills: List[str], required_skills: List[str]) -> List[str]:
        """Identify missing skills"""
        try:
            return [skill for skill in required_skills if skill not in current_skills]
        except Exception as e:
            logger.error(f"Error identifying skill gaps: {str(e)}")
            return required_skills
    
    def _calculate_work_style_match(self, user_preferences: Dict[str, Any], 
                                  required_styles: List[str]) -> float:
        """Calculate work style match score"""
        try:
            if not user_preferences or not required_styles:
                return 0.5
            
            user_styles = [k for k, v in user_preferences.items() if v is True]
            if not user_styles:
                return 0.5
            
            # Calculate overlap
            overlap = len(set(user_styles) & set(required_styles))
            return overlap / len(required_styles)
            
        except Exception as e:
            logger.error(f"Error calculating work style match: {str(e)}")
            return 0.5
    
    def _calculate_time_to_target(self, experience_years: int, current_stage: str, 
                                target_stage: str, skill_fit: float) -> int:
        """Calculate time to reach target career stage"""
        try:
            stage_months = {
                'entry_level': 0,
                'junior': 12,
                'mid_level': 36,
                'senior': 60,
                'lead': 84,
                'executive': 120,
                'expert': 96
            }
            
            current_months = stage_months.get(current_stage, 0)
            target_months = stage_months.get(target_stage, 60)
            
            base_time = target_months - current_months
            
            # Adjust based on experience and skill fit
            experience_factor = max(0.5, 1.0 - (experience_years * 0.1))
            skill_factor = max(0.5, skill_fit)
            
            adjusted_time = base_time * experience_factor * (2.0 - skill_factor)
            
            return max(6, int(adjusted_time))  # Minimum 6 months
            
        except Exception as e:
            logger.error(f"Error calculating time to target: {str(e)}")
            return 24
    
    def _calculate_confidence_score(self, learning_style_fit: float, skill_fit: float,
                                  work_style_match: float, job_demand: float) -> float:
        """Calculate overall confidence score"""
        try:
            # Weighted average of all factors
            weights = {
                'learning_style': 0.25,
                'skills': 0.35,
                'work_style': 0.25,
                'job_demand': 0.15
            }
            
            confidence = (
                learning_style_fit * weights['learning_style'] +
                skill_fit * weights['skills'] +
                work_style_match * weights['work_style'] +
                job_demand * weights['job_demand']
            )
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.5
    
    def _analyze_skill_transition(self, current_skills: List[str], 
                                target_skills: List[str]) -> Dict[str, Any]:
        """Analyze skill transition requirements"""
        try:
            new_skills = [skill for skill in target_skills if skill not in current_skills]
            transferable_skills = [skill for skill in current_skills if skill in target_skills]
            obsolete_skills = [skill for skill in current_skills if skill not in target_skills]
            
            return {
                'new_skills_required': new_skills,
                'transferable_skills': transferable_skills,
                'obsolete_skills': obsolete_skills,
                'transition_difficulty': len(new_skills) / len(target_skills) if target_skills else 0,
                'skill_overlap': len(transferable_skills) / len(target_skills) if target_skills else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing skill transition: {str(e)}")
            return {}
    
    def _analyze_industry_transition(self, current_industry: IndustrySector, 
                                   target_industry: IndustrySector) -> Dict[str, Any]:
        """Analyze industry transition difficulty"""
        try:
            # Industry transition difficulty matrix
            transition_difficulty = {
                IndustrySector.TECHNOLOGY: {
                    IndustrySector.FINANCE: 0.3,
                    IndustrySector.HEALTHCARE: 0.4,
                    IndustrySector.EDUCATION: 0.2,
                    IndustrySector.CONSULTING: 0.3
                },
                IndustrySector.FINANCE: {
                    IndustrySector.TECHNOLOGY: 0.4,
                    IndustrySector.CONSULTING: 0.2,
                    IndustrySector.HEALTHCARE: 0.5
                },
                IndustrySector.HEALTHCARE: {
                    IndustrySector.TECHNOLOGY: 0.4,
                    IndustrySector.FINANCE: 0.5,
                    IndustrySector.EDUCATION: 0.3
                }
            }
            
            difficulty = transition_difficulty.get(current_industry, {}).get(target_industry, 0.6)
            
            return {
                'difficulty_score': difficulty,
                'transition_feasibility': difficulty < 0.7,
                'required_adaptation': 'high' if difficulty > 0.6 else 'medium' if difficulty > 0.4 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing industry transition: {str(e)}")
            return {'difficulty_score': 0.5, 'transition_feasibility': True, 'required_adaptation': 'medium'}
    
    def _calculate_transition_time(self, current_path: CareerPath, target_path: CareerPath,
                                 skill_transition: Dict[str, Any], 
                                 industry_transition: Dict[str, Any]) -> int:
        """Calculate transition time between career paths"""
        try:
            base_time = max(current_path.time_to_target, target_path.time_to_target)
            
            # Adjust for skill transition difficulty
            skill_factor = 1.0 + skill_transition.get('transition_difficulty', 0.5)
            
            # Adjust for industry transition difficulty
            industry_factor = 1.0 + industry_transition.get('difficulty_score', 0.5)
            
            total_time = base_time * skill_factor * industry_factor
            
            return max(6, int(total_time))
            
        except Exception as e:
            logger.error(f"Error calculating transition time: {str(e)}")
            return 24
    
    def _calculate_success_probability(self, current_path: CareerPath, target_path: CareerPath,
                                     skill_transition: Dict[str, Any], 
                                     industry_transition: Dict[str, Any]) -> float:
        """Calculate success probability for career transition"""
        try:
            # Base success probability
            base_probability = 0.5
            
            # Adjust for skill overlap
            skill_overlap = skill_transition.get('skill_overlap', 0)
            skill_factor = 0.3 + (skill_overlap * 0.7)
            
            # Adjust for industry transition difficulty
            industry_difficulty = industry_transition.get('difficulty_score', 0.5)
            industry_factor = 1.0 - (industry_difficulty * 0.5)
            
            # Adjust for current path confidence
            current_confidence = current_path.confidence_score
            target_confidence = target_path.confidence_score
            
            success_probability = base_probability * skill_factor * industry_factor * current_confidence * target_confidence
            
            return min(1.0, max(0.0, success_probability))
            
        except Exception as e:
            logger.error(f"Error calculating success probability: {str(e)}")
            return 0.5
    
    def _generate_transition_steps(self, current_path: CareerPath, target_path: CareerPath,
                                 skill_transition: Dict[str, Any]) -> List[str]:
        """Generate recommended transition steps"""
        try:
            steps = []
            
            # Skill development steps
            new_skills = skill_transition.get('new_skills_required', [])
            for skill in new_skills[:5]:  # Top 5 skills
                steps.append(f"Develop {skill} skills through courses and practice")
            
            # Industry knowledge steps
            if current_path.industry != target_path.industry:
                steps.append(f"Gain knowledge about {target_path.industry.value} industry")
                steps.append(f"Network with professionals in {target_path.industry.value} sector")
            
            # Experience building steps
            steps.append("Build relevant project portfolio")
            steps.append("Seek mentorship from target role professionals")
            steps.append("Consider lateral moves within current organization")
            
            return steps[:8]  # Return top 8 steps
            
        except Exception as e:
            logger.error(f"Error generating transition steps: {str(e)}")
            return ["Focus on skill development", "Build relevant experience"]
    
    def _identify_risk_factors(self, current_path: CareerPath, target_path: CareerPath,
                             skill_transition: Dict[str, Any], 
                             industry_transition: Dict[str, Any]) -> List[str]:
        """Identify potential risk factors"""
        try:
            risks = []
            
            # Skill gap risks
            if skill_transition.get('transition_difficulty', 0) > 0.7:
                risks.append("High skill gap may require significant learning time")
            
            # Industry transition risks
            if industry_transition.get('difficulty_score', 0) > 0.6:
                risks.append("Industry transition may require additional networking and adaptation")
            
            # Market demand risks
            if target_path.job_market_demand < 0.6:
                risks.append("Target role has limited job market demand")
            
            # Time investment risks
            if target_path.time_to_target > 36:
                risks.append("Long transition time may require sustained commitment")
            
            return risks
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {str(e)}")
            return ["General transition risks apply"]
    
    def _generate_industry_trend(self, industry: IndustrySector) -> IndustryTrend:
        """Generate industry trend analysis"""
        try:
            # Sample industry trend data
            trend_data = {
                IndustrySector.TECHNOLOGY: {
                    'growth_rate': 0.12,
                    'job_demand': 0.9,
                    'skill_demand': {'programming': 0.95, 'ai_ml': 0.9, 'cloud': 0.85, 'cybersecurity': 0.8},
                    'emerging_skills': ['artificial_intelligence', 'machine_learning', 'cloud_computing', 'cybersecurity'],
                    'declining_skills': ['legacy_systems', 'waterfall_methodology'],
                    'salary_trends': {CareerStage.ENTRY_LEVEL: 65000, CareerStage.SENIOR: 120000, CareerStage.LEAD: 150000},
                    'future_outlook': 'Strong growth expected with high demand for AI and cloud skills'
                },
                IndustrySector.HEALTHCARE: {
                    'growth_rate': 0.08,
                    'job_demand': 0.8,
                    'skill_demand': {'healthcare_data': 0.9, 'telemedicine': 0.85, 'regulatory': 0.8, 'analytics': 0.75},
                    'emerging_skills': ['telemedicine', 'healthcare_analytics', 'digital_health', 'regulatory_affairs'],
                    'declining_skills': ['paper_based_systems', 'manual_processes'],
                    'salary_trends': {CareerStage.ENTRY_LEVEL: 55000, CareerStage.SENIOR: 95000, CareerStage.LEAD: 120000},
                    'future_outlook': 'Steady growth with increasing focus on digital health and data analytics'
                },
                IndustrySector.FINANCE: {
                    'growth_rate': 0.06,
                    'job_demand': 0.75,
                    'skill_demand': {'fintech': 0.9, 'risk_management': 0.85, 'data_analytics': 0.8, 'compliance': 0.8},
                    'emerging_skills': ['fintech', 'blockchain', 'risk_analytics', 'regulatory_technology'],
                    'declining_skills': ['traditional_banking', 'manual_trading'],
                    'salary_trends': {CareerStage.ENTRY_LEVEL: 60000, CareerStage.SENIOR: 110000, CareerStage.LEAD: 140000},
                    'future_outlook': 'Moderate growth with increasing focus on fintech and digital transformation'
                }
            }
            
            data = trend_data.get(industry, trend_data[IndustrySector.TECHNOLOGY])
            
            return IndustryTrend(
                industry=industry,
                growth_rate=data['growth_rate'],
                job_demand=data['job_demand'],
                skill_demand=data['skill_demand'],
                emerging_skills=data['emerging_skills'],
                declining_skills=data['declining_skills'],
                salary_trends=data['salary_trends'],
                future_outlook=data['future_outlook']
            )
            
        except Exception as e:
            logger.error(f"Error generating industry trend: {str(e)}")
            return None
    
    def get_prediction_statistics(self) -> Dict[str, int]:
        """Get career prediction statistics"""
        try:
            total_paths = sum(len(paths) for paths in self.career_paths.values())
            total_industries = len(self.industry_trends)
            
            return {
                'total_career_paths': total_paths,
                'industries_analyzed': total_industries,
                'prediction_models': len(self.skill_models)
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction statistics: {str(e)}")
            return {
                'total_career_paths': 0,
                'industries_analyzed': 0,
                'prediction_models': 0
            }
