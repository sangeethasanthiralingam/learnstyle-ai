"""
EEG Data Processing Module

This module handles real-time EEG data processing including:
- Band power extraction (Alpha, Beta, Theta, Gamma)
- Signal filtering and preprocessing
- Artifact detection and removal
- Feature extraction for learning optimization

Author: LearnStyle AI Team
Version: 1.0.0
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EEGProcessor:
    """
    EEG data processor for real-time brain-wave analysis
    """
    
    def __init__(self, sampling_rate: int = 256):
        """
        Initialize EEG processor
        
        Args:
            sampling_rate: EEG sampling rate in Hz (default: 256)
        """
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
        
        # Define frequency bands for learning analysis
        self.band_filters = {
            'delta': (0.5, 4),      # Deep sleep, unconscious
            'theta': (4, 8),        # Drowsiness, meditation
            'alpha': (8, 13),       # Relaxed awareness, focus
            'beta': (13, 30),       # Active concentration, alertness
            'gamma': (30, 100)      # High-level cognitive processing
        }
        
        # Initialize bandpass filters
        self._initialize_filters()
        
        logger.info(f"EEG Processor initialized with sampling rate: {sampling_rate} Hz")
    
    def _initialize_filters(self):
        """Initialize bandpass filters for each frequency band"""
        self.filters = {}
        
        for band_name, (low_freq, high_freq) in self.band_filters.items():
            # Normalize frequencies
            low_norm = low_freq / self.nyquist
            high_norm = high_freq / self.nyquist
            
            # Design Butterworth filter
            b, a = signal.butter(4, [low_norm, high_norm], btype='band')
            self.filters[band_name] = (b, a)
    
    def process_eeg_data(self, eeg_data: np.ndarray, 
                        artifact_removal: bool = True) -> Dict[str, float]:
        """
        Process raw EEG data and extract band powers
        
        Args:
            eeg_data: Raw EEG signal (1D array)
            artifact_removal: Whether to apply artifact removal
            
        Returns:
            Dictionary containing band powers and features
        """
        try:
            # Validate input data
            if len(eeg_data) < self.sampling_rate:  # Need at least 1 second of data
                logger.warning("Insufficient EEG data for processing")
                return self._get_empty_result()
            
            # Preprocess data
            processed_data = self._preprocess_data(eeg_data, artifact_removal)
            
            # Extract band powers
            band_powers = self._extract_band_powers(processed_data)
            
            # Calculate additional features
            features = self._calculate_features(processed_data, band_powers)
            
            # Combine results
            result = {**band_powers, **features}
            
            logger.debug(f"Processed EEG data: {len(eeg_data)} samples")
            return result
            
        except Exception as e:
            logger.error(f"Error processing EEG data: {str(e)}")
            return self._get_empty_result()
    
    def _preprocess_data(self, eeg_data: np.ndarray, artifact_removal: bool) -> np.ndarray:
        """
        Preprocess EEG data (detrending, artifact removal, etc.)
        
        Args:
            eeg_data: Raw EEG signal
            artifact_removal: Whether to apply artifact removal
            
        Returns:
            Preprocessed EEG data
        """
        # Detrend the signal
        processed = signal.detrend(eeg_data)
        
        # Apply artifact removal if requested
        if artifact_removal:
            processed = self._remove_artifacts(processed)
        
        # Apply notch filter to remove power line noise (50/60 Hz)
        processed = self._apply_notch_filter(processed)
        
        return processed
    
    def _remove_artifacts(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Remove artifacts from EEG data using simple thresholding
        
        Args:
            eeg_data: EEG signal
            
        Returns:
            Artifact-removed EEG data
        """
        # Calculate signal statistics
        mean_val = np.mean(eeg_data)
        std_val = np.std(eeg_data)
        
        # Remove outliers (values > 3 standard deviations)
        threshold = 3 * std_val
        mask = np.abs(eeg_data - mean_val) < threshold
        
        # Interpolate removed samples
        if not np.all(mask):
            indices = np.arange(len(eeg_data))
            processed = np.interp(indices, indices[mask], eeg_data[mask])
        else:
            processed = eeg_data
        
        return processed
    
    def _apply_notch_filter(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to remove power line noise
        
        Args:
            eeg_data: EEG signal
            
        Returns:
            Notch-filtered EEG data
        """
        # Remove 50 Hz (European) and 60 Hz (American) power line noise
        for freq in [50, 60]:
            if freq < self.nyquist:
                notch_freq = freq / self.nyquist
                b, a = signal.iirnotch(notch_freq, Q=30)
                eeg_data = signal.filtfilt(b, a, eeg_data)
        
        return eeg_data
    
    def _extract_band_powers(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        Extract power in each frequency band
        
        Args:
            eeg_data: Preprocessed EEG signal
            
        Returns:
            Dictionary of band powers
        """
        band_powers = {}
        
        for band_name, (b, a) in self.filters.items():
            # Apply bandpass filter
            filtered_signal = signal.filtfilt(b, a, eeg_data)
            
            # Calculate power (mean squared amplitude)
            power = np.mean(filtered_signal ** 2)
            
            # Convert to dB for better scaling
            power_db = 10 * np.log10(power + 1e-10)  # Add small value to avoid log(0)
            
            band_powers[f"{band_name}_power"] = power_db
            band_powers[f"{band_name}_power_raw"] = power
        
        return band_powers
    
    def _calculate_features(self, eeg_data: np.ndarray, 
                          band_powers: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate additional EEG features for learning analysis
        
        Args:
            eeg_data: Preprocessed EEG signal
            band_powers: Band power values
            
        Returns:
            Dictionary of additional features
        """
        features = {}
        
        # Spectral centroid (center of mass of spectrum)
        fft = np.fft.fft(eeg_data)
        freqs = np.fft.fftfreq(len(eeg_data), 1/self.sampling_rate)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = freqs[:len(freqs)//2]
        
        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            features['spectral_centroid'] = spectral_centroid
        else:
            features['spectral_centroid'] = 0.0
        
        # Alpha/Beta ratio (focus indicator)
        alpha_power = band_powers.get('alpha_power_raw', 0)
        beta_power = band_powers.get('beta_power_raw', 0)
        
        if beta_power > 0:
            features['alpha_beta_ratio'] = alpha_power / beta_power
        else:
            features['alpha_beta_ratio'] = 0.0
        
        # Theta/Alpha ratio (drowsiness indicator)
        theta_power = band_powers.get('theta_power_raw', 0)
        if alpha_power > 0:
            features['theta_alpha_ratio'] = theta_power / alpha_power
        else:
            features['theta_alpha_ratio'] = 0.0
        
        # Signal complexity (approximate entropy)
        features['complexity'] = self._calculate_approximate_entropy(eeg_data)
        
        return features
    
    def _calculate_approximate_entropy(self, data: np.ndarray, 
                                     m: int = 2, r: float = 0.2) -> float:
        """
        Calculate approximate entropy for signal complexity
        
        Args:
            data: Input signal
            m: Pattern length
            r: Tolerance (fraction of standard deviation)
            
        Returns:
            Approximate entropy value
        """
        try:
            N = len(data)
            if N < m + 1:
                return 0.0
            
            # Calculate standard deviation
            std_dev = np.std(data)
            r = r * std_dev
            
            def _maxdist(xi, xj, N, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _approximate_entropy(m, data, N, r):
                def _phi(m):
                    C = np.zeros(N - m + 1)
                    for i in range(N - m + 1):
                        template_i = data[i:i + m]
                        for j in range(N - m + 1):
                            template_j = data[j:j + m]
                            if _maxdist(template_i, template_j, N, m) <= r:
                                C[i] += 1.0
                    phi = np.mean(np.log(C / float(N - m + 1.0)))
                    return phi
                return _phi(m) - _phi(m + 1)
            
            return _approximate_entropy(m, data, N, r)
            
        except Exception as e:
            logger.warning(f"Error calculating approximate entropy: {str(e)}")
            return 0.0
    
    def _get_empty_result(self) -> Dict[str, float]:
        """Return empty result structure"""
        result = {}
        
        # Add empty band powers
        for band_name in self.band_filters.keys():
            result[f"{band_name}_power"] = 0.0
            result[f"{band_name}_power_raw"] = 0.0
        
        # Add empty features
        result.update({
            'spectral_centroid': 0.0,
            'alpha_beta_ratio': 0.0,
            'theta_alpha_ratio': 0.0,
            'complexity': 0.0
        })
        
        return result
    
    def get_band_info(self) -> Dict[str, Dict[str, float]]:
        """
        Get information about frequency bands
        
        Returns:
            Dictionary with band information
        """
        return {
            band_name: {
                'low_freq': freq_range[0],
                'high_freq': freq_range[1],
                'description': self._get_band_description(band_name)
            }
            for band_name, freq_range in self.band_filters.items()
        }
    
    def _get_band_description(self, band_name: str) -> str:
        """Get description of frequency band"""
        descriptions = {
            'delta': 'Deep sleep, unconscious state',
            'theta': 'Drowsiness, meditation, creativity',
            'alpha': 'Relaxed awareness, focus, calm alertness',
            'beta': 'Active concentration, alertness, problem-solving',
            'gamma': 'High-level cognitive processing, consciousness'
        }
        return descriptions.get(band_name, 'Unknown frequency band')
