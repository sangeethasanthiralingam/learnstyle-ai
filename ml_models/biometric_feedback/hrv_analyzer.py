"""
Heart Rate Variability (HRV) Analyzer Module

This module provides comprehensive HRV analysis for learning optimization including:
- Real-time HRV monitoring and analysis
- Stress level detection and management
- Autonomic nervous system state assessment
- Learning readiness evaluation
- Recovery and fatigue monitoring

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
from scipy import signal
from scipy.stats import entropy
import joblib

logger = logging.getLogger(__name__)

class HRVState(Enum):
    """HRV physiological states"""
    OPTIMAL = "optimal"
    STRESSED = "stressed"
    FATIGUED = "fatigued"
    RECOVERING = "recovering"
    OVERSTIMULATED = "overstimulated"
    UNDERSTIMULATED = "understimulated"

class StressLevel(Enum):
    """Stress level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

class LearningReadiness(Enum):
    """Learning readiness states"""
    OPTIMAL = "optimal"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    NOT_RECOMMENDED = "not_recommended"

@dataclass
class HRVMetrics:
    """HRV analysis metrics"""
    timestamp: datetime
    mean_rr: float  # Mean R-R interval in ms
    rmssd: float    # Root mean square of successive differences
    sdnn: float     # Standard deviation of NN intervals
    pnn50: float    # Percentage of NN intervals > 50ms
    vlf_power: float  # Very low frequency power
    lf_power: float   # Low frequency power
    hf_power: float   # High frequency power
    lf_hf_ratio: float  # LF/HF ratio
    total_power: float  # Total spectral power
    stress_index: float  # Stress index (SDNN/RMSSD)
    autonomic_balance: float  # Autonomic nervous system balance
    recovery_index: float  # Recovery index
    learning_readiness: LearningReadiness
    hrv_state: HRVState
    stress_level: StressLevel
    confidence: float

class HRVAnalyzer:
    """
    Advanced Heart Rate Variability analysis system
    """
    
    def __init__(self):
        """Initialize HRV analyzer"""
        self.hrv_history = []
        self.baseline_metrics = None
        self.learning_thresholds = {
            'optimal_rmssd': (30, 100),
            'optimal_sdnn': (40, 80),
            'optimal_lf_hf': (0.5, 3.0),
            'stress_threshold': 0.7,
            'fatigue_threshold': 0.3
        }
        
        logger.info("HRV Analyzer initialized")
    
    def analyze_hrv(self, rr_intervals: List[float], 
                   sampling_rate: float = 1000.0) -> HRVMetrics:
        """
        Analyze HRV from R-R intervals
        
        Args:
            rr_intervals: List of R-R intervals in milliseconds
            sampling_rate: Sampling rate in Hz
            
        Returns:
            HRV analysis metrics
        """
        try:
            # Clean and validate R-R intervals
            cleaned_rr = self._clean_rr_intervals(rr_intervals)
            
            if len(cleaned_rr) < 10:
                return self._create_default_metrics()
            
            # Calculate time domain metrics
            time_domain = self._calculate_time_domain_metrics(cleaned_rr)
            
            # Calculate frequency domain metrics
            freq_domain = self._calculate_frequency_domain_metrics(cleaned_rr, sampling_rate)
            
            # Calculate derived metrics
            derived_metrics = self._calculate_derived_metrics(time_domain, freq_domain)
            
            # Assess physiological state
            hrv_state = self._assess_hrv_state(time_domain, freq_domain, derived_metrics)
            stress_level = self._assess_stress_level(derived_metrics)
            learning_readiness = self._assess_learning_readiness(hrv_state, stress_level)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(cleaned_rr, time_domain)
            
            # Create HRV metrics
            metrics = HRVMetrics(
                timestamp=datetime.now(),
                mean_rr=time_domain['mean_rr'],
                rmssd=time_domain['rmssd'],
                sdnn=time_domain['sdnn'],
                pnn50=time_domain['pnn50'],
                vlf_power=freq_domain['vlf_power'],
                lf_power=freq_domain['lf_power'],
                hf_power=freq_domain['hf_power'],
                lf_hf_ratio=freq_domain['lf_hf_ratio'],
                total_power=freq_domain['total_power'],
                stress_index=derived_metrics['stress_index'],
                autonomic_balance=derived_metrics['autonomic_balance'],
                recovery_index=derived_metrics['recovery_index'],
                learning_readiness=learning_readiness,
                hrv_state=hrv_state,
                stress_level=stress_level,
                confidence=confidence
            )
            
            # Store in history
            self.hrv_history.append(metrics)
            
            # Update baseline if needed
            self._update_baseline(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing HRV: {str(e)}")
            return self._create_default_metrics()
    
    def get_learning_recommendations(self, metrics: HRVMetrics) -> List[str]:
        """
        Get learning recommendations based on HRV metrics
        
        Args:
            metrics: HRV analysis metrics
            
        Returns:
            List of learning recommendations
        """
        try:
            recommendations = []
            
            if metrics.learning_readiness == LearningReadiness.OPTIMAL:
                recommendations.extend([
                    "Optimal learning state detected - proceed with challenging content",
                    "Consider advanced topics or complex problem-solving",
                    "Perfect time for intensive study sessions"
                ])
            elif metrics.learning_readiness == LearningReadiness.GOOD:
                recommendations.extend([
                    "Good learning state - continue with current content",
                    "Consider moderate difficulty exercises",
                    "Monitor for signs of fatigue"
                ])
            elif metrics.learning_readiness == LearningReadiness.FAIR:
                recommendations.extend([
                    "Moderate learning state - consider lighter content",
                    "Take short breaks every 20-30 minutes",
                    "Focus on review and reinforcement"
                ])
            elif metrics.learning_readiness == LearningReadiness.POOR:
                recommendations.extend([
                    "Poor learning state - reduce content difficulty",
                    "Take longer breaks (10-15 minutes)",
                    "Consider relaxation exercises before learning"
                ])
            else:  # NOT_RECOMMENDED
                recommendations.extend([
                    "Learning not recommended - take a break",
                    "Consider stress reduction techniques",
                    "Resume learning when feeling more relaxed"
                ])
            
            # Add stress-specific recommendations
            if metrics.stress_level in [StressLevel.HIGH, StressLevel.VERY_HIGH]:
                recommendations.extend([
                    "High stress detected - practice deep breathing",
                    "Consider meditation or mindfulness exercises",
                    "Avoid high-pressure learning situations"
                ])
            
            # Add fatigue-specific recommendations
            if metrics.hrv_state == HRVState.FATIGUED:
                recommendations.extend([
                    "Fatigue detected - take a rest break",
                    "Consider light physical activity",
                    "Ensure adequate sleep before next session"
                ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting learning recommendations: {str(e)}")
            return ["Monitor your physiological state and adjust learning accordingly"]
    
    def detect_stress_patterns(self, window_minutes: int = 30) -> Dict[str, Any]:
        """
        Detect stress patterns over time window
        
        Args:
            window_minutes: Time window for analysis in minutes
            
        Returns:
            Stress pattern analysis
        """
        try:
            # Get recent metrics within window
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            recent_metrics = [m for m in self.hrv_history if m.timestamp >= cutoff_time]
            
            if len(recent_metrics) < 3:
                return {'error': 'Insufficient data for pattern analysis'}
            
            # Analyze stress trends
            stress_values = [m.stress_index for m in recent_metrics]
            timestamps = [m.timestamp for m in recent_metrics]
            
            # Calculate trend
            stress_trend = np.polyfit(range(len(stress_values)), stress_values, 1)[0]
            
            # Detect stress spikes
            stress_threshold = np.mean(stress_values) + 2 * np.std(stress_values)
            stress_spikes = [i for i, val in enumerate(stress_values) if val > stress_threshold]
            
            # Calculate stress variability
            stress_variability = np.std(stress_values)
            
            # Determine pattern type
            if stress_trend > 0.1:
                pattern_type = "increasing_stress"
            elif stress_trend < -0.1:
                pattern_type = "decreasing_stress"
            elif stress_variability > 0.3:
                pattern_type = "variable_stress"
            else:
                pattern_type = "stable_stress"
            
            return {
                'pattern_type': pattern_type,
                'stress_trend': stress_trend,
                'stress_variability': stress_variability,
                'stress_spikes': len(stress_spikes),
                'average_stress': np.mean(stress_values),
                'max_stress': np.max(stress_values),
                'min_stress': np.min(stress_values),
                'recommendations': self._get_stress_pattern_recommendations(pattern_type, stress_trend)
            }
            
        except Exception as e:
            logger.error(f"Error detecting stress patterns: {str(e)}")
            return {'error': str(e)}
    
    def _clean_rr_intervals(self, rr_intervals: List[float]) -> List[float]:
        """Clean and validate R-R intervals"""
        try:
            # Remove outliers (RR intervals outside physiological range)
            cleaned = [rr for rr in rr_intervals if 300 <= rr <= 2000]  # 30-200 BPM
            
            # Remove artifacts using median filter
            if len(cleaned) > 5:
                cleaned = signal.medfilt(cleaned, kernel_size=3).tolist()
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning RR intervals: {str(e)}")
            return rr_intervals
    
    def _calculate_time_domain_metrics(self, rr_intervals: List[float]) -> Dict[str, float]:
        """Calculate time domain HRV metrics"""
        try:
            rr_array = np.array(rr_intervals)
            
            # Basic statistics
            mean_rr = np.mean(rr_array)
            
            # RMSSD - Root mean square of successive differences
            rr_diff = np.diff(rr_array)
            rmssd = np.sqrt(np.mean(rr_diff ** 2))
            
            # SDNN - Standard deviation of NN intervals
            sdnn = np.std(rr_array)
            
            # pNN50 - Percentage of NN intervals > 50ms
            pnn50 = np.sum(np.abs(rr_diff) > 50) / len(rr_diff) * 100
            
            return {
                'mean_rr': mean_rr,
                'rmssd': rmssd,
                'sdnn': sdnn,
                'pnn50': pnn50
            }
            
        except Exception as e:
            logger.error(f"Error calculating time domain metrics: {str(e)}")
            return {'mean_rr': 0, 'rmssd': 0, 'sdnn': 0, 'pnn50': 0}
    
    def _calculate_frequency_domain_metrics(self, rr_intervals: List[float], 
                                          sampling_rate: float) -> Dict[str, float]:
        """Calculate frequency domain HRV metrics"""
        try:
            # Interpolate RR intervals to regular time series
            rr_times = np.cumsum(rr_intervals) / 1000.0  # Convert to seconds
            rr_times = rr_times - rr_times[0]  # Start from 0
            
            # Create regular time grid
            time_grid = np.arange(0, rr_times[-1], 1/sampling_rate)
            
            # Interpolate RR intervals
            rr_interp = np.interp(time_grid, rr_times, rr_intervals)
            
            # Calculate power spectral density
            freqs, psd = signal.welch(rr_interp, fs=sampling_rate, nperseg=min(256, len(rr_interp)//4))
            
            # Define frequency bands
            vlf_mask = (freqs >= 0.0033) & (freqs < 0.04)
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs < 0.4)
            
            # Calculate power in each band
            vlf_power = np.trapz(psd[vlf_mask], freqs[vlf_mask])
            lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])
            hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])
            
            total_power = vlf_power + lf_power + hf_power
            
            # Calculate LF/HF ratio
            lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
            
            return {
                'vlf_power': vlf_power,
                'lf_power': lf_power,
                'hf_power': hf_power,
                'total_power': total_power,
                'lf_hf_ratio': lf_hf_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating frequency domain metrics: {str(e)}")
            return {'vlf_power': 0, 'lf_power': 0, 'hf_power': 0, 'total_power': 0, 'lf_hf_ratio': 0}
    
    def _calculate_derived_metrics(self, time_domain: Dict[str, float], 
                                 freq_domain: Dict[str, float]) -> Dict[str, float]:
        """Calculate derived HRV metrics"""
        try:
            # Stress index (SDNN/RMSSD)
            stress_index = time_domain['sdnn'] / time_domain['rmssd'] if time_domain['rmssd'] > 0 else 0
            
            # Autonomic balance (LF/HF ratio)
            autonomic_balance = freq_domain['lf_hf_ratio']
            
            # Recovery index (based on RMSSD and HF power)
            recovery_index = (time_domain['rmssd'] * freq_domain['hf_power']) / 1000
            
            return {
                'stress_index': stress_index,
                'autonomic_balance': autonomic_balance,
                'recovery_index': recovery_index
            }
            
        except Exception as e:
            logger.error(f"Error calculating derived metrics: {str(e)}")
            return {'stress_index': 0, 'autonomic_balance': 0, 'recovery_index': 0}
    
    def _assess_hrv_state(self, time_domain: Dict[str, float], 
                         freq_domain: Dict[str, float], 
                         derived_metrics: Dict[str, float]) -> HRVState:
        """Assess HRV physiological state"""
        try:
            rmssd = time_domain['rmssd']
            sdnn = time_domain['sdnn']
            lf_hf_ratio = freq_domain['lf_hf_ratio']
            stress_index = derived_metrics['stress_index']
            
            # Optimal state
            if (self.learning_thresholds['optimal_rmssd'][0] <= rmssd <= self.learning_thresholds['optimal_rmssd'][1] and
                self.learning_thresholds['optimal_sdnn'][0] <= sdnn <= self.learning_thresholds['optimal_sdnn'][1] and
                self.learning_thresholds['optimal_lf_hf'][0] <= lf_hf_ratio <= self.learning_thresholds['optimal_lf_hf'][1]):
                return HRVState.OPTIMAL
            
            # Stressed state
            if stress_index > self.learning_thresholds['stress_threshold']:
                return HRVState.STRESSED
            
            # Fatigued state
            if stress_index < self.learning_thresholds['fatigue_threshold']:
                return HRVState.FATIGUED
            
            # Overstimulated state
            if lf_hf_ratio > 4.0:
                return HRVState.OVERSTIMULATED
            
            # Understimulated state
            if lf_hf_ratio < 0.3:
                return HRVState.UNDERSTIMULATED
            
            # Recovering state
            return HRVState.RECOVERING
            
        except Exception as e:
            logger.error(f"Error assessing HRV state: {str(e)}")
            return HRVState.OPTIMAL
    
    def _assess_stress_level(self, derived_metrics: Dict[str, float]) -> StressLevel:
        """Assess stress level from derived metrics"""
        try:
            stress_index = derived_metrics['stress_index']
            
            if stress_index < 0.3:
                return StressLevel.VERY_LOW
            elif stress_index < 0.5:
                return StressLevel.LOW
            elif stress_index < 0.7:
                return StressLevel.MODERATE
            elif stress_index < 1.0:
                return StressLevel.HIGH
            else:
                return StressLevel.VERY_HIGH
                
        except Exception as e:
            logger.error(f"Error assessing stress level: {str(e)}")
            return StressLevel.MODERATE
    
    def _assess_learning_readiness(self, hrv_state: HRVState, 
                                 stress_level: StressLevel) -> LearningReadiness:
        """Assess learning readiness from HRV state and stress level"""
        try:
            # Optimal conditions
            if (hrv_state == HRVState.OPTIMAL and 
                stress_level in [StressLevel.LOW, StressLevel.MODERATE]):
                return LearningReadiness.OPTIMAL
            
            # Good conditions
            if (hrv_state in [HRVState.OPTIMAL, HRVState.RECOVERING] and 
                stress_level in [StressLevel.LOW, StressLevel.MODERATE, StressLevel.HIGH]):
                return LearningReadiness.GOOD
            
            # Fair conditions
            if (hrv_state in [HRVState.RECOVERING, HRVState.UNDERSTIMULATED] and 
                stress_level in [StressLevel.MODERATE, StressLevel.HIGH]):
                return LearningReadiness.FAIR
            
            # Poor conditions
            if (hrv_state in [HRVState.STRESSED, HRVState.OVERSTIMULATED] and 
                stress_level in [StressLevel.HIGH, StressLevel.VERY_HIGH]):
                return LearningReadiness.POOR
            
            # Not recommended
            if (hrv_state == HRVState.FATIGUED or 
                stress_level == StressLevel.VERY_HIGH):
                return LearningReadiness.NOT_RECOMMENDED
            
            return LearningReadiness.FAIR
            
        except Exception as e:
            logger.error(f"Error assessing learning readiness: {str(e)}")
            return LearningReadiness.FAIR
    
    def _calculate_confidence_score(self, rr_intervals: List[float], 
                                  time_domain: Dict[str, float]) -> float:
        """Calculate confidence score for HRV analysis"""
        try:
            # Base confidence on data quality
            data_quality = min(1.0, len(rr_intervals) / 100)  # More data = higher confidence
            
            # Confidence based on RMSSD stability
            rmssd_stability = 1.0 - min(1.0, time_domain['rmssd'] / 100)  # Lower RMSSD = more stable
            
            # Overall confidence
            confidence = (data_quality + rmssd_stability) / 2
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.5
    
    def _update_baseline(self, metrics: HRVMetrics):
        """Update baseline HRV metrics"""
        try:
            if self.baseline_metrics is None:
                self.baseline_metrics = metrics
            else:
                # Update baseline with exponential moving average
                alpha = 0.1  # Learning rate
                self.baseline_metrics.rmssd = (1 - alpha) * self.baseline_metrics.rmssd + alpha * metrics.rmssd
                self.baseline_metrics.sdnn = (1 - alpha) * self.baseline_metrics.sdnn + alpha * metrics.sdnn
                self.baseline_metrics.lf_hf_ratio = (1 - alpha) * self.baseline_metrics.lf_hf_ratio + alpha * metrics.lf_hf_ratio
                
        except Exception as e:
            logger.error(f"Error updating baseline: {str(e)}")
    
    def _get_stress_pattern_recommendations(self, pattern_type: str, 
                                          stress_trend: float) -> List[str]:
        """Get recommendations based on stress patterns"""
        try:
            recommendations = []
            
            if pattern_type == "increasing_stress":
                recommendations.extend([
                    "Stress levels are increasing - take proactive breaks",
                    "Consider stress management techniques",
                    "Reduce learning intensity temporarily"
                ])
            elif pattern_type == "decreasing_stress":
                recommendations.extend([
                    "Stress levels are decreasing - good recovery",
                    "Consider gradually increasing learning intensity",
                    "Maintain current stress management practices"
                ])
            elif pattern_type == "variable_stress":
                recommendations.extend([
                    "Variable stress patterns detected - monitor closely",
                    "Consider consistent stress management routine",
                    "Identify stress triggers and address them"
                ])
            else:  # stable_stress
                recommendations.extend([
                    "Stable stress levels - maintain current approach",
                    "Continue monitoring for changes",
                    "Consider optimizing learning schedule"
                ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting stress pattern recommendations: {str(e)}")
            return ["Monitor stress levels and adjust learning accordingly"]
    
    def _create_default_metrics(self) -> HRVMetrics:
        """Create default HRV metrics when analysis fails"""
        return HRVMetrics(
            timestamp=datetime.now(),
            mean_rr=800.0,
            rmssd=30.0,
            sdnn=40.0,
            pnn50=5.0,
            vlf_power=100.0,
            lf_power=200.0,
            hf_power=150.0,
            lf_hf_ratio=1.3,
            total_power=450.0,
            stress_index=0.5,
            autonomic_balance=1.3,
            recovery_index=4.5,
            learning_readiness=LearningReadiness.FAIR,
            hrv_state=HRVState.OPTIMAL,
            stress_level=StressLevel.MODERATE,
            confidence=0.3
        )
    
    def get_analyzer_statistics(self) -> Dict[str, int]:
        """Get HRV analyzer statistics"""
        try:
            return {
                'total_analyses': len(self.hrv_history),
                'baseline_established': 1 if self.baseline_metrics else 0,
                'learning_thresholds': len(self.learning_thresholds)
            }
            
        except Exception as e:
            logger.error(f"Error getting analyzer statistics: {str(e)}")
            return {
                'total_analyses': 0,
                'baseline_established': 0,
                'learning_thresholds': 0
            }
