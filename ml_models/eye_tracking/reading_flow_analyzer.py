"""
Reading Flow Analysis Module

This module provides comprehensive reading flow analysis including:
- Reading pattern recognition
- Reading speed calculation
- Regression analysis
- Reading efficiency measurement
- Text comprehension assessment

Author: LearnStyle AI Team
Version: 1.0.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ReadingPattern(Enum):
    """Types of reading patterns"""
    LINEAR = "linear"                    # Sequential left-to-right reading
    ZIGZAG = "zigzag"                   # Alternating line reading
    SCANNING = "scanning"               # Quick scanning pattern
    DETAILED = "detailed"               # Slow, careful reading
    SKIMMING = "skimming"               # Fast, surface-level reading
    RANDOM = "random"                   # No clear pattern

class ReadingEfficiency(Enum):
    """Reading efficiency levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class ReadingFlowMetrics:
    """Comprehensive reading flow metrics"""
    pattern_type: ReadingPattern
    reading_speed: float  # words per minute
    regression_count: int
    fixation_duration: float
    saccade_amplitude: float
    reading_efficiency: ReadingEfficiency
    comprehension_score: float
    attention_span: float
    text_engagement: float
    reading_rhythm: float
    confidence: float

class ReadingFlowAnalyzer:
    """
    Advanced reading flow analysis system
    """
    
    def __init__(self, 
                 min_fixation_duration: float = 50,  # milliseconds
                 max_fixation_duration: float = 600,
                 reading_speed_baseline: float = 200):  # words per minute
        """
        Initialize reading flow analyzer
        
        Args:
            min_fixation_duration: Minimum fixation duration in milliseconds
            max_fixation_duration: Maximum fixation duration in milliseconds
            reading_speed_baseline: Baseline reading speed in WPM
        """
        self.min_fixation_duration = min_fixation_duration
        self.max_fixation_duration = max_fixation_duration
        self.reading_speed_baseline = reading_speed_baseline
        
        logger.info("Reading Flow Analyzer initialized")
    
    def analyze_reading_flow(self, gaze_points: List[Dict], 
                           text_content: Optional[Dict] = None) -> ReadingFlowMetrics:
        """
        Analyze reading flow from gaze data
        
        Args:
            gaze_points: List of gaze point data
            text_content: Text content information (optional)
            
        Returns:
            ReadingFlowMetrics object with comprehensive analysis
        """
        try:
            if len(gaze_points) < 3:
                return self._get_default_metrics()
            
            # Analyze reading pattern
            pattern_type = self._analyze_reading_pattern(gaze_points)
            
            # Calculate reading speed
            reading_speed = self._calculate_reading_speed(gaze_points, text_content)
            
            # Count regressions
            regression_count = self._count_regressions(gaze_points)
            
            # Calculate fixation duration
            fixation_duration = self._calculate_average_fixation_duration(gaze_points)
            
            # Calculate saccade amplitude
            saccade_amplitude = self._calculate_average_saccade_amplitude(gaze_points)
            
            # Assess reading efficiency
            reading_efficiency = self._assess_reading_efficiency(
                reading_speed, regression_count, fixation_duration
            )
            
            # Calculate comprehension score
            comprehension_score = self._calculate_comprehension_score(
                gaze_points, reading_speed, regression_count
            )
            
            # Calculate attention span
            attention_span = self._calculate_attention_span(gaze_points)
            
            # Calculate text engagement
            text_engagement = self._calculate_text_engagement(gaze_points, text_content)
            
            # Calculate reading rhythm
            reading_rhythm = self._calculate_reading_rhythm(gaze_points)
            
            # Calculate confidence
            confidence = self._calculate_confidence(gaze_points)
            
            return ReadingFlowMetrics(
                pattern_type=pattern_type,
                reading_speed=reading_speed,
                regression_count=regression_count,
                fixation_duration=fixation_duration,
                saccade_amplitude=saccade_amplitude,
                reading_efficiency=reading_efficiency,
                comprehension_score=comprehension_score,
                attention_span=attention_span,
                text_engagement=text_engagement,
                reading_rhythm=reading_rhythm,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing reading flow: {str(e)}")
            return self._get_default_metrics()
    
    def _analyze_reading_pattern(self, gaze_points: List[Dict]) -> ReadingPattern:
        """
        Analyze reading pattern from gaze data
        
        Args:
            gaze_points: List of gaze points
            
        Returns:
            Reading pattern classification
        """
        if len(gaze_points) < 3:
            return ReadingPattern.RANDOM
        
        # Extract coordinates and timestamps
        x_coords = [p['x'] for p in gaze_points]
        y_coords = [p['y'] for p in gaze_points]
        timestamps = [p.get('timestamp', 0) for p in gaze_points]
        
        # Calculate movement patterns
        x_movements = np.diff(x_coords)
        y_movements = np.diff(y_coords)
        
        # Analyze horizontal vs vertical movement
        horizontal_dominance = np.mean(np.abs(x_movements)) > np.mean(np.abs(y_movements)) * 1.5
        vertical_dominance = np.mean(np.abs(y_movements)) > np.mean(np.abs(x_movements)) * 1.5
        
        # Analyze movement consistency
        x_consistency = 1 - np.std(x_movements) / (np.mean(np.abs(x_movements)) + 1e-6)
        y_consistency = 1 - np.std(y_movements) / (np.mean(np.abs(y_movements)) + 1e-6)
        
        # Analyze reading speed variation
        if len(timestamps) > 1:
            time_intervals = np.diff(timestamps)
            speed_variation = np.std(time_intervals) / (np.mean(time_intervals) + 1e-6)
        else:
            speed_variation = 0
        
        # Classify reading pattern
        if horizontal_dominance and x_consistency > 0.7:
            if speed_variation < 0.3:
                return ReadingPattern.LINEAR
            else:
                return ReadingPattern.SCANNING
        elif vertical_dominance and y_consistency > 0.7:
            return ReadingPattern.DETAILED
        elif speed_variation > 0.8:
            return ReadingPattern.SKIMMING
        elif self._detect_zigzag_pattern(x_coords, y_coords):
            return ReadingPattern.ZIGZAG
        else:
            return ReadingPattern.RANDOM
    
    def _detect_zigzag_pattern(self, x_coords: List[float], y_coords: List[float]) -> bool:
        """Detect zigzag reading pattern"""
        if len(x_coords) < 6:
            return False
        
        # Look for alternating horizontal movements
        x_movements = np.diff(x_coords)
        direction_changes = 0
        
        for i in range(1, len(x_movements)):
            if (x_movements[i-1] > 0 and x_movements[i] < 0) or \
               (x_movements[i-1] < 0 and x_movements[i] > 0):
                direction_changes += 1
        
        # Zigzag pattern if frequent direction changes
        change_ratio = direction_changes / len(x_movements)
        return change_ratio > 0.3
    
    def _calculate_reading_speed(self, gaze_points: List[Dict], 
                               text_content: Optional[Dict]) -> float:
        """
        Calculate reading speed in words per minute
        
        Args:
            gaze_points: List of gaze points
            text_content: Text content information
            
        Returns:
            Reading speed in WPM
        """
        if len(gaze_points) < 2:
            return 0.0
        
        # Calculate time span
        timestamps = [p.get('timestamp', 0) for p in gaze_points]
        time_span = max(timestamps) - min(timestamps)
        
        if time_span <= 0:
            return 0.0
        
        # Estimate words read based on gaze points
        if text_content and 'word_count' in text_content:
            # Use actual word count if available
            words_read = text_content['word_count']
        else:
            # Estimate based on gaze points (rough approximation)
            words_read = len(gaze_points) * 0.5  # Assume 0.5 words per fixation
        
        # Calculate WPM
        minutes = time_span / 60000.0  # Convert milliseconds to minutes
        wpm = words_read / minutes if minutes > 0 else 0.0
        
        return max(0, wpm)
    
    def _count_regressions(self, gaze_points: List[Dict]) -> int:
        """
        Count reading regressions (backward eye movements)
        
        Args:
            gaze_points: List of gaze points
            
        Returns:
            Number of regressions
        """
        if len(gaze_points) < 2:
            return 0
        
        regressions = 0
        x_coords = [p['x'] for p in gaze_points]
        
        for i in range(1, len(x_coords)):
            # Regression if moving significantly backward (left for LTR languages)
            if x_coords[i] < x_coords[i-1] - 20:  # 20 pixel threshold
                regressions += 1
        
        return regressions
    
    def _calculate_average_fixation_duration(self, gaze_points: List[Dict]) -> float:
        """Calculate average fixation duration"""
        durations = [p.get('duration', 0) for p in gaze_points]
        return np.mean(durations) if durations else 0.0
    
    def _calculate_average_saccade_amplitude(self, gaze_points: List[Dict]) -> float:
        """Calculate average saccade amplitude"""
        if len(gaze_points) < 2:
            return 0.0
        
        amplitudes = []
        for i in range(1, len(gaze_points)):
            prev_point = gaze_points[i-1]
            curr_point = gaze_points[i]
            
            amplitude = np.sqrt(
                (curr_point['x'] - prev_point['x'])**2 + 
                (curr_point['y'] - prev_point['y'])**2
            )
            amplitudes.append(amplitude)
        
        return np.mean(amplitudes) if amplitudes else 0.0
    
    def _assess_reading_efficiency(self, reading_speed: float, 
                                 regression_count: int, 
                                 fixation_duration: float) -> ReadingEfficiency:
        """
        Assess reading efficiency based on multiple factors
        
        Args:
            reading_speed: Reading speed in WPM
            regression_count: Number of regressions
            fixation_duration: Average fixation duration
            
        Returns:
            Reading efficiency classification
        """
        # Calculate efficiency score
        speed_score = min(1.0, reading_speed / self.reading_speed_baseline)
        regression_score = max(0, 1.0 - (regression_count / 10.0))  # Penalize regressions
        duration_score = 1.0 if self.min_fixation_duration <= fixation_duration <= self.max_fixation_duration else 0.5
        
        # Weighted efficiency score
        efficiency_score = (speed_score * 0.4 + regression_score * 0.3 + duration_score * 0.3)
        
        # Classify efficiency
        if efficiency_score >= 0.8:
            return ReadingEfficiency.EXCELLENT
        elif efficiency_score >= 0.6:
            return ReadingEfficiency.GOOD
        elif efficiency_score >= 0.4:
            return ReadingEfficiency.FAIR
        else:
            return ReadingEfficiency.POOR
    
    def _calculate_comprehension_score(self, gaze_points: List[Dict], 
                                     reading_speed: float, 
                                     regression_count: int) -> float:
        """
        Calculate text comprehension score
        
        Args:
            gaze_points: List of gaze points
            reading_speed: Reading speed in WPM
            regression_count: Number of regressions
            
        Returns:
            Comprehension score (0-1)
        """
        # Factors affecting comprehension
        speed_factor = min(1.0, reading_speed / self.reading_speed_baseline)
        regression_factor = max(0, 1.0 - (regression_count / 15.0))  # Some regressions are normal
        
        # Attention consistency (lower variation = better comprehension)
        if len(gaze_points) > 2:
            durations = [p.get('duration', 0) for p in gaze_points]
            duration_consistency = 1.0 - (np.std(durations) / (np.mean(durations) + 1e-6))
        else:
            duration_consistency = 0.5
        
        # Weighted comprehension score
        comprehension_score = (
            speed_factor * 0.3 +
            regression_factor * 0.4 +
            duration_consistency * 0.3
        )
        
        return max(0, min(1, comprehension_score))
    
    def _calculate_attention_span(self, gaze_points: List[Dict]) -> float:
        """
        Calculate attention span from gaze data
        
        Args:
            gaze_points: List of gaze points
            
        Returns:
            Attention span in seconds
        """
        if len(gaze_points) < 2:
            return 0.0
        
        # Calculate total reading time
        timestamps = [p.get('timestamp', 0) for p in gaze_points]
        total_time = max(timestamps) - min(timestamps)
        
        # Convert to seconds
        attention_span = total_time / 1000.0
        
        return attention_span
    
    def _calculate_text_engagement(self, gaze_points: List[Dict], 
                                 text_content: Optional[Dict]) -> float:
        """
        Calculate text engagement level
        
        Args:
            gaze_points: List of gaze points
            text_content: Text content information
            
        Returns:
            Text engagement score (0-1)
        """
        if not gaze_points:
            return 0.0
        
        # Calculate engagement based on gaze density and duration
        total_duration = sum(p.get('duration', 0) for p in gaze_points)
        avg_duration = total_duration / len(gaze_points)
        
        # Higher average duration suggests better engagement
        duration_score = min(1.0, avg_duration / 300.0)  # 300ms baseline
        
        # Calculate gaze density (points per unit area)
        if text_content and 'area' in text_content:
            area = text_content['area']
            density = len(gaze_points) / area if area > 0 else 0
            density_score = min(1.0, density * 1000)  # Normalize
        else:
            density_score = 0.5
        
        # Weighted engagement score
        engagement_score = (duration_score * 0.6 + density_score * 0.4)
        
        return max(0, min(1, engagement_score))
    
    def _calculate_reading_rhythm(self, gaze_points: List[Dict]) -> float:
        """
        Calculate reading rhythm consistency
        
        Args:
            gaze_points: List of gaze points
            
        Returns:
            Reading rhythm score (0-1)
        """
        if len(gaze_points) < 3:
            return 0.5
        
        # Calculate time intervals between fixations
        timestamps = [p.get('timestamp', 0) for p in gaze_points]
        intervals = np.diff(timestamps)
        
        if len(intervals) < 2:
            return 0.5
        
        # Calculate rhythm consistency (inverse of coefficient of variation)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval > 0:
            rhythm_consistency = 1.0 - (std_interval / mean_interval)
        else:
            rhythm_consistency = 0.0
        
        return max(0, min(1, rhythm_consistency))
    
    def _calculate_confidence(self, gaze_points: List[Dict]) -> float:
        """Calculate confidence in reading flow analysis"""
        if not gaze_points:
            return 0.0
        
        # Factors affecting confidence
        data_quantity = min(1.0, len(gaze_points) / 50.0)  # 50 points baseline
        
        # Check data quality
        valid_points = sum(1 for p in gaze_points if p.get('confidence', 1.0) > 0.5)
        data_quality = valid_points / len(gaze_points) if gaze_points else 0.0
        
        # Check temporal consistency
        timestamps = [p.get('timestamp', 0) for p in gaze_points]
        if len(timestamps) > 1:
            time_span = max(timestamps) - min(timestamps)
            temporal_consistency = 1.0 if time_span > 1000 else time_span / 1000.0
        else:
            temporal_consistency = 0.0
        
        # Weighted confidence
        confidence = (data_quantity * 0.4 + data_quality * 0.4 + temporal_consistency * 0.2)
        
        return max(0.1, min(1.0, confidence))
    
    def _get_default_metrics(self) -> ReadingFlowMetrics:
        """Return default metrics when analysis fails"""
        return ReadingFlowMetrics(
            pattern_type=ReadingPattern.RANDOM,
            reading_speed=0.0,
            regression_count=0,
            fixation_duration=0.0,
            saccade_amplitude=0.0,
            reading_efficiency=ReadingEfficiency.POOR,
            comprehension_score=0.0,
            attention_span=0.0,
            text_engagement=0.0,
            reading_rhythm=0.0,
            confidence=0.1
        )
