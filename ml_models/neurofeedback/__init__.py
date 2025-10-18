"""
Neurofeedback Engine for LearnStyle AI

This module provides brain-wave adaptive learning capabilities including:
- EEG data processing and analysis
- Focus level detection
- Mental fatigue monitoring
- Cognitive load assessment
- Real-time learning optimization

Author: LearnStyle AI Team
Version: 1.0.0
"""

from .eeg_processor import EEGProcessor
from .focus_detector import FocusDetector
from .fatigue_monitor import FatigueMonitor
from .cognitive_load_assessor import CognitiveLoadAssessor

__all__ = [
    'EEGProcessor',
    'FocusDetector', 
    'FatigueMonitor',
    'CognitiveLoadAssessor'
]
