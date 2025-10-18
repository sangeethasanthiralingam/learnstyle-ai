"""
A/B Testing Framework Module

This module provides comprehensive A/B testing capabilities for educational interventions including:
- Experiment design and setup
- Random assignment and control
- Statistical analysis and significance testing
- Effect size calculation and power analysis
- Results interpretation and reporting

Author: LearnStyle AI Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
from scipy import stats
import random
import uuid

logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    """Experiment status classifications"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class AssignmentMethod(Enum):
    """User assignment methods"""
    RANDOM = "random"
    STRATIFIED = "stratified"
    MATCHED_PAIRS = "matched_pairs"
    BLOCK_RANDOMIZATION = "block_randomization"

class StatisticalTest(Enum):
    """Types of statistical tests"""
    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    FISHER_EXACT = "fisher_exact"

@dataclass
class ExperimentConfig:
    """A/B testing experiment configuration"""
    experiment_id: str
    name: str
    description: str
    hypothesis: str
    primary_metric: str
    secondary_metrics: List[str]
    assignment_method: AssignmentMethod
    statistical_test: StatisticalTest
    significance_level: float
    power: float
    minimum_effect_size: float
    max_duration_days: int
    min_sample_size: int
    max_sample_size: int
    stratification_variables: List[str]
    created_at: datetime
    created_by: str

@dataclass
class ExperimentGroup:
    """Experiment group configuration"""
    group_id: str
    group_name: str
    group_type: str  # 'control' or 'treatment'
    description: str
    intervention_config: Dict[str, Any]
    target_allocation: float  # Proportion of users to assign to this group

@dataclass
class UserAssignment:
    """User assignment to experiment group"""
    user_id: str
    experiment_id: str
    group_id: str
    assigned_at: datetime
    stratification_values: Dict[str, Any]

@dataclass
class ExperimentResults:
    """A/B testing experiment results"""
    experiment_id: str
    status: ExperimentStatus
    total_participants: int
    control_group_size: int
    treatment_group_size: int
    primary_metric_results: Dict[str, float]
    secondary_metric_results: Dict[str, Dict[str, float]]
    statistical_test_results: Dict[str, Any]
    effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: float
    is_significant: bool
    practical_significance: bool
    recommendations: List[str]
    analysis_timestamp: datetime

class ABTestingFramework:
    """
    Advanced A/B testing framework for educational interventions
    """
    
    def __init__(self):
        """Initialize A/B testing framework"""
        self.experiments = {}
        self.user_assignments = {}
        self.experiment_data = {}
        self.results_history = []
        
        logger.info("A/B Testing Framework initialized")
    
    def create_experiment(self, name: str, description: str, hypothesis: str,
                         primary_metric: str, secondary_metrics: List[str] = None,
                         assignment_method: AssignmentMethod = AssignmentMethod.RANDOM,
                         statistical_test: StatisticalTest = StatisticalTest.T_TEST,
                         significance_level: float = 0.05, power: float = 0.8,
                         minimum_effect_size: float = 0.2, max_duration_days: int = 30,
                         min_sample_size: int = 100, max_sample_size: int = 10000,
                         stratification_variables: List[str] = None,
                         created_by: str = "system") -> str:
        """
        Create a new A/B testing experiment
        
        Args:
            name: Experiment name
            description: Experiment description
            hypothesis: Research hypothesis
            primary_metric: Primary outcome metric
            secondary_metrics: Secondary outcome metrics
            assignment_method: Method for assigning users to groups
            statistical_test: Statistical test to use
            significance_level: Statistical significance level
            power: Statistical power
            minimum_effect_size: Minimum detectable effect size
            max_duration_days: Maximum experiment duration
            min_sample_size: Minimum sample size per group
            max_sample_size: Maximum sample size per group
            stratification_variables: Variables for stratified assignment
            created_by: User who created the experiment
            
        Returns:
            Experiment ID
        """
        try:
            experiment_id = str(uuid.uuid4())
            
            config = ExperimentConfig(
                experiment_id=experiment_id,
                name=name,
                description=description,
                hypothesis=hypothesis,
                primary_metric=primary_metric,
                secondary_metrics=secondary_metrics or [],
                assignment_method=assignment_method,
                statistical_test=statistical_test,
                significance_level=significance_level,
                power=power,
                minimum_effect_size=minimum_effect_size,
                max_duration_days=max_duration_days,
                min_sample_size=min_sample_size,
                max_sample_size=max_sample_size,
                stratification_variables=stratification_variables or [],
                created_at=datetime.now(),
                created_by=created_by
            )
            
            self.experiments[experiment_id] = {
                'config': config,
                'groups': {},
                'status': ExperimentStatus.DRAFT,
                'started_at': None,
                'ended_at': None
            }
            
            logger.info(f"Created experiment {experiment_id}: {name}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating experiment: {str(e)}")
            return None
    
    def add_experiment_group(self, experiment_id: str, group_name: str, 
                           group_type: str, description: str,
                           intervention_config: Dict[str, Any],
                           target_allocation: float = 0.5) -> str:
        """
        Add a group to an experiment
        
        Args:
            experiment_id: Experiment ID
            group_name: Group name
            group_type: 'control' or 'treatment'
            description: Group description
            intervention_config: Intervention configuration
            target_allocation: Target proportion of users for this group
            
        Returns:
            Group ID
        """
        try:
            if experiment_id not in self.experiments:
                return None
            
            group_id = str(uuid.uuid4())
            
            group = ExperimentGroup(
                group_id=group_id,
                group_name=group_name,
                group_type=group_type,
                description=description,
                intervention_config=intervention_config,
                target_allocation=target_allocation
            )
            
            self.experiments[experiment_id]['groups'][group_id] = group
            
            logger.info(f"Added group {group_id} to experiment {experiment_id}")
            return group_id
            
        except Exception as e:
            logger.error(f"Error adding experiment group: {str(e)}")
            return None
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B testing experiment"""
        try:
            if experiment_id not in self.experiments:
                return False
            
            experiment = self.experiments[experiment_id]
            
            # Validate experiment before starting
            if len(experiment['groups']) < 2:
                logger.error("Experiment must have at least 2 groups")
                return False
            
            # Check if there's at least one control and one treatment group
            has_control = any(group.group_type == 'control' for group in experiment['groups'].values())
            has_treatment = any(group.group_type == 'treatment' for group in experiment['groups'].values())
            
            if not has_control or not has_treatment:
                logger.error("Experiment must have both control and treatment groups")
                return False
            
            # Start experiment
            experiment['status'] = ExperimentStatus.RUNNING
            experiment['started_at'] = datetime.now()
            
            logger.info(f"Started experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting experiment: {str(e)}")
            return False
    
    def assign_user_to_experiment(self, experiment_id: str, user_id: str,
                                user_attributes: Dict[str, Any] = None) -> Optional[str]:
        """
        Assign a user to an experiment group
        
        Args:
            experiment_id: Experiment ID
            user_id: User ID
            user_attributes: User attributes for stratification
            
        Returns:
            Group ID assigned to user
        """
        try:
            if experiment_id not in self.experiments:
                return None
            
            experiment = self.experiments[experiment_id]
            
            if experiment['status'] != ExperimentStatus.RUNNING:
                return None
            
            # Check if user is already assigned
            if user_id in self.user_assignments:
                existing_assignment = self.user_assignments[user_id]
                if existing_assignment.experiment_id == experiment_id:
                    return existing_assignment.group_id
            
            # Assign user to group based on assignment method
            config = experiment['config']
            groups = experiment['groups']
            
            if config.assignment_method == AssignmentMethod.RANDOM:
                group_id = self._random_assignment(groups)
            elif config.assignment_method == AssignmentMethod.STRATIFIED:
                group_id = self._stratified_assignment(groups, user_attributes, config.stratification_variables)
            elif config.assignment_method == AssignmentMethod.MATCHED_PAIRS:
                group_id = self._matched_pairs_assignment(groups, user_attributes)
            elif config.assignment_method == AssignmentMethod.BLOCK_RANDOMIZATION:
                group_id = self._block_randomization_assignment(groups)
            else:
                group_id = self._random_assignment(groups)
            
            # Create assignment
            assignment = UserAssignment(
                user_id=user_id,
                experiment_id=experiment_id,
                group_id=group_id,
                assigned_at=datetime.now(),
                stratification_values=user_attributes or {}
            )
            
            self.user_assignments[user_id] = assignment
            
            logger.info(f"Assigned user {user_id} to group {group_id} in experiment {experiment_id}")
            return group_id
            
        except Exception as e:
            logger.error(f"Error assigning user to experiment: {str(e)}")
            return None
    
    def _random_assignment(self, groups: Dict[str, ExperimentGroup]) -> str:
        """Random assignment to groups"""
        try:
            group_ids = list(groups.keys())
            group_allocations = [groups[gid].target_allocation for gid in group_ids]
            
            # Normalize allocations
            total_allocation = sum(group_allocations)
            if total_allocation == 0:
                # Equal allocation if no target allocations specified
                group_allocations = [1.0 / len(group_ids)] * len(group_ids)
            else:
                group_allocations = [alloc / total_allocation for alloc in group_allocations]
            
            # Random assignment based on allocations
            return np.random.choice(group_ids, p=group_allocations)
            
        except Exception as e:
            logger.error(f"Error in random assignment: {str(e)}")
            return list(groups.keys())[0]  # Fallback to first group
    
    def _stratified_assignment(self, groups: Dict[str, ExperimentGroup], 
                             user_attributes: Dict[str, Any],
                             stratification_variables: List[str]) -> str:
        """Stratified assignment based on user attributes"""
        try:
            # For simplicity, use random assignment within strata
            # In a real implementation, this would maintain balance across strata
            return self._random_assignment(groups)
            
        except Exception as e:
            logger.error(f"Error in stratified assignment: {str(e)}")
            return self._random_assignment(groups)
    
    def _matched_pairs_assignment(self, groups: Dict[str, ExperimentGroup],
                                user_attributes: Dict[str, Any]) -> str:
        """Matched pairs assignment"""
        try:
            # For simplicity, use random assignment
            # In a real implementation, this would match users based on attributes
            return self._random_assignment(groups)
            
        except Exception as e:
            logger.error(f"Error in matched pairs assignment: {str(e)}")
            return self._random_assignment(groups)
    
    def _block_randomization_assignment(self, groups: Dict[str, ExperimentGroup]) -> str:
        """Block randomization assignment"""
        try:
            # For simplicity, use random assignment
            # In a real implementation, this would use block randomization
            return self._random_assignment(groups)
            
        except Exception as e:
            logger.error(f"Error in block randomization assignment: {str(e)}")
            return self._random_assignment(groups)
    
    def record_experiment_data(self, experiment_id: str, user_id: str, 
                             metric_name: str, value: float, timestamp: datetime = None):
        """Record data for an experiment participant"""
        try:
            if experiment_id not in self.experiment_data:
                self.experiment_data[experiment_id] = {}
            
            if user_id not in self.experiment_data[experiment_id]:
                self.experiment_data[experiment_id][user_id] = {}
            
            if metric_name not in self.experiment_data[experiment_id][user_id]:
                self.experiment_data[experiment_id][user_id][metric_name] = []
            
            self.experiment_data[experiment_id][user_id][metric_name].append({
                'value': value,
                'timestamp': timestamp or datetime.now()
            })
            
        except Exception as e:
            logger.error(f"Error recording experiment data: {str(e)}")
    
    def analyze_experiment(self, experiment_id: str) -> Optional[ExperimentResults]:
        """Analyze A/B testing experiment results"""
        try:
            if experiment_id not in self.experiments:
                return None
            
            experiment = self.experiments[experiment_id]
            config = experiment['config']
            
            # Get user assignments for this experiment
            experiment_assignments = {
                user_id: assignment for user_id, assignment in self.user_assignments.items()
                if assignment.experiment_id == experiment_id
            }
            
            if not experiment_assignments:
                return None
            
            # Separate control and treatment groups
            control_users = []
            treatment_users = []
            
            for user_id, assignment in experiment_assignments.items():
                group = experiment['groups'][assignment.group_id]
                if group.group_type == 'control':
                    control_users.append(user_id)
                elif group.group_type == 'treatment':
                    treatment_users.append(user_id)
            
            # Get data for each group
            control_data = self._get_group_data(experiment_id, control_users, config.primary_metric)
            treatment_data = self._get_group_data(experiment_id, treatment_users, config.primary_metric)
            
            if not control_data or not treatment_data:
                return None
            
            # Perform statistical analysis
            statistical_results = self._perform_statistical_test(
                control_data, treatment_data, config.statistical_test
            )
            
            # Calculate effect size
            effect_size = self._calculate_effect_size(control_data, treatment_data)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(control_data, treatment_data)
            
            # Determine significance
            is_significant = statistical_results['p_value'] < config.significance_level
            practical_significance = effect_size >= config.minimum_effect_size
            
            # Generate recommendations
            recommendations = self._generate_experiment_recommendations(
                is_significant, practical_significance, effect_size, statistical_results
            )
            
            # Create results
            results = ExperimentResults(
                experiment_id=experiment_id,
                status=experiment['status'],
                total_participants=len(experiment_assignments),
                control_group_size=len(control_users),
                treatment_group_size=len(treatment_users),
                primary_metric_results={
                    'control_mean': np.mean(control_data),
                    'treatment_mean': np.mean(treatment_data),
                    'control_std': np.std(control_data, ddof=1),
                    'treatment_std': np.std(treatment_data, ddof=1)
                },
                secondary_metric_results={},  # Would be populated with secondary metrics
                statistical_test_results=statistical_results,
                effect_size=effect_size,
                confidence_interval=confidence_interval,
                p_value=statistical_results['p_value'],
                is_significant=is_significant,
                practical_significance=practical_significance,
                recommendations=recommendations,
                analysis_timestamp=datetime.now()
            )
            
            # Store results
            self.results_history.append(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing experiment: {str(e)}")
            return None
    
    def _get_group_data(self, experiment_id: str, user_ids: List[str], metric_name: str) -> List[float]:
        """Get data for a group of users"""
        try:
            data = []
            
            for user_id in user_ids:
                if (experiment_id in self.experiment_data and 
                    user_id in self.experiment_data[experiment_id] and
                    metric_name in self.experiment_data[experiment_id][user_id]):
                    
                    user_metric_data = self.experiment_data[experiment_id][user_id][metric_name]
                    # Use the latest value for each user
                    if user_metric_data:
                        latest_value = user_metric_data[-1]['value']
                        data.append(latest_value)
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting group data: {str(e)}")
            return []
    
    def _perform_statistical_test(self, control_data: List[float], treatment_data: List[float],
                                test_type: StatisticalTest) -> Dict[str, Any]:
        """Perform statistical test on group data"""
        try:
            if test_type == StatisticalTest.T_TEST:
                t_stat, p_value = stats.ttest_ind(treatment_data, control_data)
                return {
                    'test_type': 't_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'degrees_of_freedom': len(control_data) + len(treatment_data) - 2
                }
            elif test_type == StatisticalTest.MANN_WHITNEY:
                statistic, p_value = stats.mannwhitneyu(treatment_data, control_data, alternative='two-sided')
                return {
                    'test_type': 'mann_whitney',
                    'statistic': statistic,
                    'p_value': p_value
                }
            elif test_type == StatisticalTest.WILCOXON:
                statistic, p_value = stats.wilcoxon(treatment_data, control_data)
                return {
                    'test_type': 'wilcoxon',
                    'statistic': statistic,
                    'p_value': p_value
                }
            else:
                # Default to t-test
                t_stat, p_value = stats.ttest_ind(treatment_data, control_data)
                return {
                    'test_type': 't_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'degrees_of_freedom': len(control_data) + len(treatment_data) - 2
                }
                
        except Exception as e:
            logger.error(f"Error performing statistical test: {str(e)}")
            return {
                'test_type': 'unknown',
                'statistic': 0.0,
                'p_value': 1.0
            }
    
    def _calculate_effect_size(self, control_data: List[float], treatment_data: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        try:
            if not control_data or not treatment_data:
                return 0.0
            
            # Calculate means and pooled standard deviation
            mean_control = np.mean(control_data)
            mean_treatment = np.mean(treatment_data)
            
            n1, n2 = len(control_data), len(treatment_data)
            var1, var2 = np.var(control_data, ddof=1), np.var(treatment_data, ddof=1)
            
            if n1 + n2 - 2 == 0:
                return 0.0
            
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            
            if pooled_std == 0:
                return 0.0
            
            cohens_d = (mean_treatment - mean_control) / pooled_std
            return cohens_d
            
        except Exception as e:
            logger.error(f"Error calculating effect size: {str(e)}")
            return 0.0
    
    def _calculate_confidence_interval(self, control_data: List[float], 
                                     treatment_data: List[float],
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for the difference in means"""
        try:
            if not control_data or not treatment_data:
                return (0.0, 0.0)
            
            # Calculate difference in means
            mean_diff = np.mean(treatment_data) - np.mean(control_data)
            
            # Calculate standard error
            n1, n2 = len(control_data), len(treatment_data)
            var1, var2 = np.var(control_data, ddof=1), np.var(treatment_data, ddof=1)
            
            if n1 + n2 - 2 == 0:
                return (0.0, 0.0)
            
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            se_diff = np.sqrt(pooled_var * (1/n1 + 1/n2))
            
            # Calculate confidence interval
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, n1 + n2 - 2)
            margin_error = t_critical * se_diff
            
            lower_bound = mean_diff - margin_error
            upper_bound = mean_diff + margin_error
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {str(e)}")
            return (0.0, 0.0)
    
    def _generate_experiment_recommendations(self, is_significant: bool,
                                           practical_significance: bool,
                                           effect_size: float,
                                           statistical_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on experiment results"""
        try:
            recommendations = []
            
            if is_significant and practical_significance:
                recommendations.append("The treatment shows statistically and practically significant improvement")
                recommendations.append("Consider implementing the treatment intervention")
            elif is_significant and not practical_significance:
                recommendations.append("The treatment shows statistical significance but small practical effect")
                recommendations.append("Consider the cost-benefit of implementing the treatment")
            elif not is_significant and practical_significance:
                recommendations.append("The treatment shows practical significance but not statistical significance")
                recommendations.append("Consider increasing sample size or extending experiment duration")
            else:
                recommendations.append("The treatment does not show significant improvement")
                recommendations.append("Consider alternative interventions or further investigation")
            
            # Effect size recommendations
            if abs(effect_size) < 0.2:
                recommendations.append("Effect size is small - consider more impactful interventions")
            elif abs(effect_size) < 0.5:
                recommendations.append("Effect size is medium - intervention shows moderate impact")
            else:
                recommendations.append("Effect size is large - intervention shows strong impact")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating experiment recommendations: {str(e)}")
            return ["Review experiment results and consider next steps"]
    
    def get_experiment_summary(self, experiment_id: str) -> Optional[Dict]:
        """Get summary of an experiment"""
        try:
            if experiment_id not in self.experiments:
                return None
            
            experiment = self.experiments[experiment_id]
            config = experiment['config']
            
            # Get recent results
            recent_results = None
            for results in reversed(self.results_history):
                if results.experiment_id == experiment_id:
                    recent_results = results
                    break
            
            summary = {
                'experiment_id': experiment_id,
                'name': config.name,
                'description': config.description,
                'status': experiment['status'].value,
                'created_at': config.created_at.isoformat(),
                'started_at': experiment['started_at'].isoformat() if experiment['started_at'] else None,
                'ended_at': experiment['ended_at'].isoformat() if experiment['ended_at'] else None,
                'groups': {
                    group_id: {
                        'name': group.group_name,
                        'type': group.group_type,
                        'description': group.description
                    } for group_id, group in experiment['groups'].items()
                },
                'results': {
                    'total_participants': recent_results.total_participants,
                    'is_significant': recent_results.is_significant,
                    'effect_size': recent_results.effect_size,
                    'p_value': recent_results.p_value
                } if recent_results else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting experiment summary: {str(e)}")
            return None
    
    def get_framework_statistics(self) -> Dict[str, int]:
        """Get A/B testing framework statistics"""
        try:
            stats = {
                'total_experiments': len(self.experiments),
                'active_experiments': sum(1 for exp in self.experiments.values() 
                                        if exp['status'] == ExperimentStatus.RUNNING),
                'completed_experiments': sum(1 for exp in self.experiments.values() 
                                           if exp['status'] == ExperimentStatus.COMPLETED),
                'total_assignments': len(self.user_assignments),
                'total_results': len(self.results_history)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting framework statistics: {str(e)}")
            return {
                'total_experiments': 0,
                'active_experiments': 0,
                'completed_experiments': 0,
                'total_assignments': 0,
                'total_results': 0
            }
