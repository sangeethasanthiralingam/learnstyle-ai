"""
Statistical Analysis Module

This module provides comprehensive statistical analysis capabilities including:
- Descriptive statistics calculation
- Inferential statistical tests
- Correlation and regression analysis
- Time series analysis
- Power analysis and sample size calculation

Author: LearnStyle AI Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class StatisticalTest(Enum):
    """Types of statistical tests"""
    T_TEST_INDEPENDENT = "t_test_independent"
    T_TEST_PAIRED = "t_test_paired"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    ANOVA = "anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    PEARSON_CORRELATION = "pearson_correlation"
    SPEARMAN_CORRELATION = "spearman_correlation"

class DistributionType(Enum):
    """Types of probability distributions"""
    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    BINOMIAL = "binomial"
    POISSON = "poisson"

@dataclass
class DescriptiveStatistics:
    """Descriptive statistics results"""
    count: int
    mean: float
    median: float
    mode: float
    std: float
    variance: float
    min: float
    max: float
    range: float
    quartiles: Dict[str, float]
    skewness: float
    kurtosis: float
    outliers: List[float]

@dataclass
class CorrelationResult:
    """Correlation analysis results"""
    correlation_type: str
    correlation_coefficient: float
    p_value: float
    confidence_interval: Tuple[float, float]
    strength: str
    significance: bool

@dataclass
class RegressionResult:
    """Regression analysis results"""
    model_type: str
    r_squared: float
    adjusted_r_squared: float
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    residuals: List[float]
    predictions: List[float]

class StatisticalAnalyzer:
    """
    Advanced statistical analysis system
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistical analyzer
        
        Args:
            significance_level: Default significance level for tests
        """
        self.significance_level = significance_level
        self.analysis_history = []
        
        logger.info("Statistical Analyzer initialized")
    
    def calculate_descriptive_statistics(self, data: List[float]) -> DescriptiveStatistics:
        """Calculate comprehensive descriptive statistics"""
        try:
            if not data:
                return self._get_default_descriptive_stats()
            
            data_array = np.array(data)
            
            # Basic statistics
            count = len(data)
            mean = np.mean(data_array)
            median = np.median(data_array)
            std = np.std(data_array, ddof=1)
            variance = np.var(data_array, ddof=1)
            min_val = np.min(data_array)
            max_val = np.max(data_array)
            range_val = max_val - min_val
            
            # Mode (most frequent value)
            unique_values, counts = np.unique(data_array, return_counts=True)
            mode = unique_values[np.argmax(counts)]
            
            # Quartiles
            quartiles = {
                'q1': np.percentile(data_array, 25),
                'q2': np.percentile(data_array, 50),  # Same as median
                'q3': np.percentile(data_array, 75)
            }
            
            # Skewness and kurtosis
            skewness = stats.skew(data_array)
            kurtosis = stats.kurtosis(data_array)
            
            # Outliers (using IQR method)
            iqr = quartiles['q3'] - quartiles['q1']
            lower_bound = quartiles['q1'] - 1.5 * iqr
            upper_bound = quartiles['q3'] + 1.5 * iqr
            outliers = [x for x in data_array if x < lower_bound or x > upper_bound]
            
            return DescriptiveStatistics(
                count=count,
                mean=mean,
                median=median,
                mode=mode,
                std=std,
                variance=variance,
                min=min_val,
                max=max_val,
                range=range_val,
                quartiles=quartiles,
                skewness=skewness,
                kurtosis=kurtosis,
                outliers=outliers
            )
            
        except Exception as e:
            logger.error(f"Error calculating descriptive statistics: {str(e)}")
            return self._get_default_descriptive_stats()
    
    def perform_hypothesis_test(self, data1: List[float], data2: List[float] = None,
                              test_type: StatisticalTest = StatisticalTest.T_TEST_INDEPENDENT,
                              alternative: str = 'two-sided') -> Dict[str, Any]:
        """Perform hypothesis testing"""
        try:
            if not data1:
                return {'error': 'No data provided'}
            
            data1_array = np.array(data1)
            
            if test_type == StatisticalTest.T_TEST_INDEPENDENT:
                if data2 is None:
                    return {'error': 'Two samples required for independent t-test'}
                
                data2_array = np.array(data2)
                t_stat, p_value = stats.ttest_ind(data1_array, data2_array)
                
                return {
                    'test_type': 'independent_t_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'degrees_of_freedom': len(data1) + len(data2) - 2,
                    'significant': p_value < self.significance_level,
                    'effect_size': self._calculate_cohens_d(data1_array, data2_array)
                }
            
            elif test_type == StatisticalTest.T_TEST_PAIRED:
                if data2 is None:
                    return {'error': 'Two samples required for paired t-test'}
                
                data2_array = np.array(data2)
                t_stat, p_value = stats.ttest_rel(data1_array, data2_array)
                
                return {
                    'test_type': 'paired_t_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'degrees_of_freedom': len(data1) - 1,
                    'significant': p_value < self.significance_level,
                    'effect_size': self._calculate_cohens_d(data1_array, data2_array)
                }
            
            elif test_type == StatisticalTest.MANN_WHITNEY_U:
                if data2 is None:
                    return {'error': 'Two samples required for Mann-Whitney U test'}
                
                data2_array = np.array(data2)
                statistic, p_value = stats.mannwhitneyu(data1_array, data2_array, alternative=alternative)
                
                return {
                    'test_type': 'mann_whitney_u',
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < self.significance_level,
                    'effect_size': self._calculate_rank_biserial_correlation(data1_array, data2_array)
                }
            
            elif test_type == StatisticalTest.WILCOXON_SIGNED_RANK:
                if data2 is None:
                    return {'error': 'Two samples required for Wilcoxon signed-rank test'}
                
                data2_array = np.array(data2)
                statistic, p_value = stats.wilcoxon(data1_array, data2_array, alternative=alternative)
                
                return {
                    'test_type': 'wilcoxon_signed_rank',
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < self.significance_level,
                    'effect_size': self._calculate_rank_biserial_correlation(data1_array, data2_array)
                }
            
            elif test_type == StatisticalTest.ANOVA:
                if data2 is None:
                    return {'error': 'Multiple samples required for ANOVA'}
                
                # For simplicity, assume data2 is a list of lists
                groups = [data1_array, np.array(data2)]
                f_stat, p_value = stats.f_oneway(*groups)
                
                return {
                    'test_type': 'anova',
                    'statistic': f_stat,
                    'p_value': p_value,
                    'degrees_of_freedom_between': len(groups) - 1,
                    'degrees_of_freedom_within': sum(len(group) for group in groups) - len(groups),
                    'significant': p_value < self.significance_level,
                    'effect_size': self._calculate_eta_squared(groups)
                }
            
            else:
                return {'error': f'Unsupported test type: {test_type}'}
                
        except Exception as e:
            logger.error(f"Error performing hypothesis test: {str(e)}")
            return {'error': str(e)}
    
    def calculate_correlation(self, x: List[float], y: List[float],
                            correlation_type: str = 'pearson') -> CorrelationResult:
        """Calculate correlation between two variables"""
        try:
            if len(x) != len(y):
                return self._get_default_correlation_result()
            
            x_array = np.array(x)
            y_array = np.array(y)
            
            if correlation_type.lower() == 'pearson':
                corr_coef, p_value = pearsonr(x_array, y_array)
                corr_type = 'pearson'
            elif correlation_type.lower() == 'spearman':
                corr_coef, p_value = spearmanr(x_array, y_array)
                corr_type = 'spearman'
            elif correlation_type.lower() == 'kendall':
                corr_coef, p_value = kendalltau(x_array, y_array)
                corr_type = 'kendall'
            else:
                corr_coef, p_value = pearsonr(x_array, y_array)
                corr_type = 'pearson'
            
            # Calculate confidence interval
            n = len(x)
            if n > 2:
                z = 0.5 * np.log((1 + corr_coef) / (1 - corr_coef))
                se = 1 / np.sqrt(n - 3)
                z_lower = z - 1.96 * se
                z_upper = z + 1.96 * se
                ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
                confidence_interval = (ci_lower, ci_upper)
            else:
                confidence_interval = (0.0, 0.0)
            
            # Determine strength
            abs_corr = abs(corr_coef)
            if abs_corr < 0.1:
                strength = 'negligible'
            elif abs_corr < 0.3:
                strength = 'weak'
            elif abs_corr < 0.5:
                strength = 'moderate'
            elif abs_corr < 0.7:
                strength = 'strong'
            else:
                strength = 'very strong'
            
            return CorrelationResult(
                correlation_type=corr_type,
                correlation_coefficient=corr_coef,
                p_value=p_value,
                confidence_interval=confidence_interval,
                strength=strength,
                significance=p_value < self.significance_level
            )
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            return self._get_default_correlation_result()
    
    def perform_regression_analysis(self, x: List[float], y: List[float],
                                  regression_type: str = 'linear') -> RegressionResult:
        """Perform regression analysis"""
        try:
            if len(x) != len(y):
                return self._get_default_regression_result()
            
            x_array = np.array(x).reshape(-1, 1)
            y_array = np.array(y)
            
            if regression_type.lower() == 'linear':
                # Simple linear regression
                model = LinearRegression()
                model.fit(x_array, y_array)
                
                # Predictions
                predictions = model.predict(x_array)
                
                # Calculate R-squared
                r_squared = r2_score(y_array, predictions)
                
                # Calculate adjusted R-squared
                n = len(x)
                p = 1  # Number of predictors
                adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
                
                # Coefficients
                coefficients = {
                    'intercept': model.intercept_,
                    'slope': model.coef_[0]
                }
                
                # Calculate p-values (simplified)
                residuals = y_array - predictions
                mse = np.mean(residuals ** 2)
                se_intercept = np.sqrt(mse * (1/n + np.mean(x)**2 / np.sum((x - np.mean(x))**2)))
                se_slope = np.sqrt(mse / np.sum((x - np.mean(x))**2))
                
                t_intercept = model.intercept_ / se_intercept
                t_slope = model.coef_[0] / se_slope
                
                p_values = {
                    'intercept': 2 * (1 - stats.t.cdf(abs(t_intercept), n - 2)),
                    'slope': 2 * (1 - stats.t.cdf(abs(t_slope), n - 2))
                }
                
                # Confidence intervals
                t_critical = stats.t.ppf(1 - self.significance_level/2, n - 2)
                ci_intercept = (model.intercept_ - t_critical * se_intercept,
                              model.intercept_ + t_critical * se_intercept)
                ci_slope = (model.coef_[0] - t_critical * se_slope,
                          model.coef_[0] + t_critical * se_slope)
                
                confidence_intervals = {
                    'intercept': ci_intercept,
                    'slope': ci_slope
                }
                
                return RegressionResult(
                    model_type='linear',
                    r_squared=r_squared,
                    adjusted_r_squared=adjusted_r_squared,
                    coefficients=coefficients,
                    p_values=p_values,
                    confidence_intervals=confidence_intervals,
                    residuals=residuals.tolist(),
                    predictions=predictions.tolist()
                )
            
            else:
                return self._get_default_regression_result()
                
        except Exception as e:
            logger.error(f"Error performing regression analysis: {str(e)}")
            return self._get_default_regression_result()
    
    def test_normality(self, data: List[float], test_type: str = 'shapiro') -> Dict[str, Any]:
        """Test if data follows normal distribution"""
        try:
            if not data:
                return {'error': 'No data provided'}
            
            data_array = np.array(data)
            
            if test_type.lower() == 'shapiro':
                statistic, p_value = stats.shapiro(data_array)
                test_name = 'Shapiro-Wilk'
            elif test_type.lower() == 'kolmogorov':
                statistic, p_value = stats.kstest(data_array, 'norm', args=(np.mean(data_array), np.std(data_array)))
                test_name = 'Kolmogorov-Smirnov'
            elif test_type.lower() == 'anderson':
                result = stats.anderson(data_array, dist='norm')
                statistic = result.statistic
                p_value = None  # Anderson-Darling doesn't provide p-value directly
                test_name = 'Anderson-Darling'
            else:
                statistic, p_value = stats.shapiro(data_array)
                test_name = 'Shapiro-Wilk'
            
            is_normal = p_value > self.significance_level if p_value is not None else False
            
            return {
                'test_type': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'is_normal': is_normal,
                'significant': not is_normal
            }
            
        except Exception as e:
            logger.error(f"Error testing normality: {str(e)}")
            return {'error': str(e)}
    
    def calculate_power_analysis(self, effect_size: float, sample_size: int = None,
                               power: float = None, alpha: float = None) -> Dict[str, float]:
        """Calculate power analysis for sample size or power"""
        try:
            if alpha is None:
                alpha = self.significance_level
            
            if sample_size is not None and power is None:
                # Calculate power given sample size
                from statsmodels.stats.power import ttest_power
                power = ttest_power(effect_size, nobs=sample_size, alpha=alpha)
                
                return {
                    'effect_size': effect_size,
                    'sample_size': sample_size,
                    'power': power,
                    'alpha': alpha
                }
            
            elif power is not None and sample_size is None:
                # Calculate sample size given power
                from statsmodels.stats.power import ttest_power
                from scipy.optimize import minimize_scalar
                
                def power_diff(n):
                    return abs(ttest_power(effect_size, nobs=int(n), alpha=alpha) - power)
                
                result = minimize_scalar(power_diff, bounds=(10, 10000), method='bounded')
                sample_size = int(result.x)
                
                return {
                    'effect_size': effect_size,
                    'sample_size': sample_size,
                    'power': power,
                    'alpha': alpha
                }
            
            else:
                return {'error': 'Must specify either sample_size or power'}
                
        except Exception as e:
            logger.error(f"Error calculating power analysis: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        try:
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            
            if n1 + n2 - 2 == 0:
                return 0.0
            
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            
            if pooled_std == 0:
                return 0.0
            
            cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
            return cohens_d
            
        except Exception as e:
            logger.error(f"Error calculating Cohen's d: {str(e)}")
            return 0.0
    
    def _calculate_rank_biserial_correlation(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate rank-biserial correlation"""
        try:
            # Combine groups and rank
            combined = np.concatenate([group1, group2])
            ranks = stats.rankdata(combined)
            
            # Split ranks back
            ranks1 = ranks[:len(group1)]
            ranks2 = ranks[len(group1):]
            
            # Calculate rank-biserial correlation
            n1, n2 = len(group1), len(group2)
            mean_rank1 = np.mean(ranks1)
            mean_rank2 = np.mean(ranks2)
            
            r_b = (mean_rank1 - mean_rank2) / ((n1 + n2) / 2)
            
            return r_b
            
        except Exception as e:
            logger.error(f"Error calculating rank-biserial correlation: {str(e)}")
            return 0.0
    
    def _calculate_eta_squared(self, groups: List[np.ndarray]) -> float:
        """Calculate eta-squared effect size for ANOVA"""
        try:
            # Calculate means
            group_means = [np.mean(group) for group in groups]
            grand_mean = np.mean([np.mean(group) for group in groups])
            
            # Calculate sum of squares
            ss_between = sum(len(group) * (group_mean - grand_mean)**2 for group, group_mean in zip(groups, group_means))
            ss_total = sum(np.sum((group - grand_mean)**2) for group in groups)
            
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            return eta_squared
            
        except Exception as e:
            logger.error(f"Error calculating eta-squared: {str(e)}")
            return 0.0
    
    def _get_default_descriptive_stats(self) -> DescriptiveStatistics:
        """Return default descriptive statistics"""
        return DescriptiveStatistics(
            count=0, mean=0.0, median=0.0, mode=0.0, std=0.0, variance=0.0,
            min=0.0, max=0.0, range=0.0, quartiles={'q1': 0.0, 'q2': 0.0, 'q3': 0.0},
            skewness=0.0, kurtosis=0.0, outliers=[]
        )
    
    def _get_default_correlation_result(self) -> CorrelationResult:
        """Return default correlation result"""
        return CorrelationResult(
            correlation_type='pearson',
            correlation_coefficient=0.0,
            p_value=1.0,
            confidence_interval=(0.0, 0.0),
            strength='negligible',
            significance=False
        )
    
    def _get_default_regression_result(self) -> RegressionResult:
        """Return default regression result"""
        return RegressionResult(
            model_type='linear',
            r_squared=0.0,
            adjusted_r_squared=0.0,
            coefficients={},
            p_values={},
            confidence_intervals={},
            residuals=[],
            predictions=[]
        )
    
    def get_analysis_statistics(self) -> Dict[str, int]:
        """Get statistical analysis statistics"""
        try:
            return {
                'total_analyses': len(self.analysis_history),
                'descriptive_analyses': sum(1 for analysis in self.analysis_history if analysis.get('type') == 'descriptive'),
                'hypothesis_tests': sum(1 for analysis in self.analysis_history if analysis.get('type') == 'hypothesis'),
                'correlation_analyses': sum(1 for analysis in self.analysis_history if analysis.get('type') == 'correlation'),
                'regression_analyses': sum(1 for analysis in self.analysis_history if analysis.get('type') == 'regression')
            }
            
        except Exception as e:
            logger.error(f"Error getting analysis statistics: {str(e)}")
            return {
                'total_analyses': 0,
                'descriptive_analyses': 0,
                'hypothesis_tests': 0,
                'correlation_analyses': 0,
                'regression_analyses': 0
            }
