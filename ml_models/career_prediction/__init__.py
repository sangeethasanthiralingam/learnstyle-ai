"""
Career Prediction Engine Module

This module provides comprehensive career path prediction and skill gap analysis including:
- Career path prediction based on learning styles and skills
- Skill gap analysis and recommendations
- Industry trend analysis
- Career progression modeling
- Personalized career recommendations

Author: LearnStyle AI Team
Version: 1.0.0
"""

from .career_path_predictor import CareerPathPredictor, CareerPath, CareerStage, IndustryTrend
from .skill_gap_analyzer import SkillGapAnalyzer, SkillGap, SkillCategory, SkillLevel
from .career_recommendation_engine import CareerRecommendationEngine, CareerRecommendation, RecommendationType
from .industry_analysis import IndustryAnalyzer, IndustryInsight, MarketTrend, JobMarketAnalysis
from .career_progression_model import CareerProgressionModel, ProgressionStage, CareerMilestone

__all__ = [
    'CareerPathPredictor', 'CareerPath', 'CareerStage', 'IndustryTrend',
    'SkillGapAnalyzer', 'SkillGap', 'SkillCategory', 'SkillLevel',
    'CareerRecommendationEngine', 'CareerRecommendation', 'RecommendationType',
    'IndustryAnalyzer', 'IndustryInsight', 'MarketTrend', 'JobMarketAnalysis',
    'CareerProgressionModel', 'ProgressionStage', 'CareerMilestone'
]
