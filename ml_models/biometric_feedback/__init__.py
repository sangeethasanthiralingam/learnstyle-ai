"""
Biometric Feedback Learning Module

This module provides comprehensive biometric feedback learning including:
- Heart Rate Variability (HRV) monitoring and analysis
- Galvanic Skin Response (GSR) monitoring
- Stress and arousal level detection
- Learning state optimization
- Biometric-based content adaptation
- Real-time physiological feedback

Author: LearnStyle AI Team
Version: 1.0.0
"""

from .hrv_analyzer import HRVAnalyzer, HRVMetrics, HRVState, StressLevel
from .gsr_monitor import GSRMonitor, GSRMetrics, ArousalLevel, EmotionalState
from .biometric_fusion import BiometricFusionEngine, BiometricState, FusedBiometricState, LearningOptimization
from .stress_detector import StressDetector, StressMetrics, StressType
from .learning_optimizer import LearningOptimizer, OptimizationStrategy, BiometricRecommendation

__all__ = [
    'HRVAnalyzer', 'HRVMetrics', 'HRVState', 'StressLevel',
    'GSRMonitor', 'GSRMetrics', 'ArousalLevel', 'EmotionalState',
    'BiometricFusionEngine', 'BiometricState', 'FusedBiometricState', 'LearningOptimization',
    'StressDetector', 'StressMetrics', 'StressType',
    'LearningOptimizer', 'OptimizationStrategy', 'BiometricRecommendation'
]
