"""
Emotion and Attention AI Module

This module provides comprehensive emotion and attention detection including:
- Facial expression analysis and emotion recognition
- Voice emotion detection and sentiment analysis
- Attention state monitoring and engagement detection
- Multi-modal emotion fusion and analysis
- Real-time engagement optimization

Author: LearnStyle AI Team
Version: 1.0.0
"""

from .facial_emotion_analyzer import FacialEmotionAnalyzer
from .voice_emotion_analyzer import VoiceEmotionAnalyzer
from .attention_engagement_detector import AttentionEngagementDetector
from .emotion_fusion_engine import EmotionFusionEngine
from .engagement_optimizer import EngagementOptimizer

__all__ = [
    'FacialEmotionAnalyzer',
    'VoiceEmotionAnalyzer',
    'AttentionEngagementDetector',
    'EmotionFusionEngine',
    'EngagementOptimizer'
]
