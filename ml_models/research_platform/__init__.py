"""
Learning Effectiveness Research Platform Module

This module provides comprehensive research capabilities including:
- Learning effectiveness measurement and analysis
- A/B testing framework for educational interventions
- Statistical analysis and hypothesis testing
- Research data collection and management
- Academic reporting and visualization

Author: LearnStyle AI Team
Version: 1.0.0
"""

from .learning_effectiveness_analyzer import LearningEffectivenessAnalyzer
from .ab_testing_framework import ABTestingFramework
from .statistical_analyzer import StatisticalAnalyzer
from .research_data_manager import ResearchDataManager
from .academic_report_generator import AcademicReportGenerator

__all__ = [
    'LearningEffectivenessAnalyzer',
    'ABTestingFramework',
    'StatisticalAnalyzer',
    'ResearchDataManager',
    'AcademicReportGenerator'
]
