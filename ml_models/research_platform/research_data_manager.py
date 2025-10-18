"""
Research Data Management Module

This module provides comprehensive research data management including:
- Research data collection and storage
- Data validation and quality control
- Data export and import capabilities
- Research participant management
- Data privacy and compliance

Author: LearnStyle AI Team
Version: 1.0.0
"""

import json
import csv
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta
import uuid
import hashlib

logger = logging.getLogger(__name__)

class DataType(Enum):
    """Types of research data"""
    LEARNING_OUTCOMES = "learning_outcomes"
    ENGAGEMENT_METRICS = "engagement_metrics"
    BEHAVIORAL_DATA = "behavioral_data"
    SURVEY_RESPONSES = "survey_responses"
    INTERACTION_DATA = "interaction_data"
    PERFORMANCE_DATA = "performance_data"

class DataQuality(Enum):
    """Data quality classifications"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"

@dataclass
class ResearchParticipant:
    """Research participant information"""
    participant_id: str
    user_id: str
    demographics: Dict[str, Any]
    consent_status: bool
    enrollment_date: datetime
    last_activity: datetime
    data_quality_score: float
    participation_level: str

@dataclass
class ResearchDataset:
    """Research dataset information"""
    dataset_id: str
    name: str
    description: str
    data_type: DataType
    participant_count: int
    data_points: int
    created_at: datetime
    last_updated: datetime
    quality_score: float
    metadata: Dict[str, Any]

@dataclass
class DataValidationResult:
    """Data validation results"""
    is_valid: bool
    quality_score: float
    quality_level: DataQuality
    validation_errors: List[str]
    data_completeness: float
    data_consistency: float
    recommendations: List[str]

class ResearchDataManager:
    """
    Advanced research data management system
    """
    
    def __init__(self):
        """Initialize research data manager"""
        self.participants = {}
        self.datasets = {}
        self.raw_data = {}
        self.validation_history = []
        
        logger.info("Research Data Manager initialized")
    
    def register_participant(self, user_id: str, demographics: Dict[str, Any],
                           consent_status: bool = True) -> str:
        """
        Register a new research participant
        
        Args:
            user_id: User identifier
            demographics: Participant demographic information
            consent_status: Whether participant has given consent
            participation_level: Level of participation
            
        Returns:
            Participant ID
        """
        try:
            participant_id = str(uuid.uuid4())
            
            participant = ResearchParticipant(
                participant_id=participant_id,
                user_id=user_id,
                demographics=demographics,
                consent_status=consent_status,
                enrollment_date=datetime.now(),
                last_activity=datetime.now(),
                data_quality_score=0.0,
                participation_level='active'
            )
            
            self.participants[participant_id] = participant
            
            logger.info(f"Registered participant {participant_id} for user {user_id}")
            return participant_id
            
        except Exception as e:
            logger.error(f"Error registering participant: {str(e)}")
            return None
    
    def create_dataset(self, name: str, description: str, data_type: DataType,
                      metadata: Dict[str, Any] = None) -> str:
        """
        Create a new research dataset
        
        Args:
            name: Dataset name
            description: Dataset description
            data_type: Type of data in the dataset
            metadata: Additional metadata
            
        Returns:
            Dataset ID
        """
        try:
            dataset_id = str(uuid.uuid4())
            
            dataset = ResearchDataset(
                dataset_id=dataset_id,
                name=name,
                description=description,
                data_type=data_type,
                participant_count=0,
                data_points=0,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                quality_score=0.0,
                metadata=metadata or {}
            )
            
            self.datasets[dataset_id] = dataset
            self.raw_data[dataset_id] = []
            
            logger.info(f"Created dataset {dataset_id}: {name}")
            return dataset_id
            
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            return None
    
    def add_data_to_dataset(self, dataset_id: str, participant_id: str,
                           data: Dict[str, Any], timestamp: datetime = None) -> bool:
        """
        Add data to a research dataset
        
        Args:
            dataset_id: Dataset identifier
            participant_id: Participant identifier
            data: Data to add
            timestamp: Data timestamp
            
        Returns:
            Success status
        """
        try:
            if dataset_id not in self.datasets:
                return False
            
            if participant_id not in self.participants:
                return False
            
            # Validate data before adding
            validation_result = self._validate_data(data, self.datasets[dataset_id].data_type)
            
            if not validation_result.is_valid:
                logger.warning(f"Data validation failed for participant {participant_id}: {validation_result.validation_errors}")
                return False
            
            # Add data
            data_entry = {
                'data_id': str(uuid.uuid4()),
                'dataset_id': dataset_id,
                'participant_id': participant_id,
                'data': data,
                'timestamp': timestamp or datetime.now(),
                'validation_result': validation_result
            }
            
            self.raw_data[dataset_id].append(data_entry)
            
            # Update dataset statistics
            self._update_dataset_statistics(dataset_id)
            
            # Update participant activity
            self.participants[participant_id].last_activity = datetime.now()
            
            logger.info(f"Added data to dataset {dataset_id} for participant {participant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding data to dataset: {str(e)}")
            return False
    
    def validate_dataset(self, dataset_id: str) -> DataValidationResult:
        """
        Validate entire dataset
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            DataValidationResult object
        """
        try:
            if dataset_id not in self.datasets:
                return self._get_default_validation_result()
            
            dataset = self.datasets[dataset_id]
            data_entries = self.raw_data.get(dataset_id, [])
            
            if not data_entries:
                return self._get_default_validation_result()
            
            # Calculate validation metrics
            total_entries = len(data_entries)
            valid_entries = sum(1 for entry in data_entries if entry['validation_result'].is_valid)
            
            data_completeness = self._calculate_data_completeness(data_entries)
            data_consistency = self._calculate_data_consistency(data_entries)
            
            # Calculate overall quality score
            quality_score = (
                (valid_entries / total_entries) * 0.4 +
                data_completeness * 0.3 +
                data_consistency * 0.3
            )
            
            # Determine quality level
            if quality_score >= 0.9:
                quality_level = DataQuality.EXCELLENT
            elif quality_score >= 0.7:
                quality_level = DataQuality.GOOD
            elif quality_score >= 0.5:
                quality_level = DataQuality.FAIR
            elif quality_score >= 0.3:
                quality_level = DataQuality.POOR
            else:
                quality_level = DataQuality.INVALID
            
            # Collect validation errors
            validation_errors = []
            for entry in data_entries:
                if not entry['validation_result'].is_valid:
                    validation_errors.extend(entry['validation_result'].validation_errors)
            
            # Generate recommendations
            recommendations = self._generate_data_quality_recommendations(
                quality_score, data_completeness, data_consistency, validation_errors
            )
            
            # Update dataset quality score
            dataset.quality_score = quality_score
            dataset.last_updated = datetime.now()
            
            # Store validation result
            validation_result = DataValidationResult(
                is_valid=quality_level in [DataQuality.EXCELLENT, DataQuality.GOOD],
                quality_score=quality_score,
                quality_level=quality_level,
                validation_errors=validation_errors,
                data_completeness=data_completeness,
                data_consistency=data_consistency,
                recommendations=recommendations
            )
            
            self.validation_history.append({
                'dataset_id': dataset_id,
                'timestamp': datetime.now(),
                'validation_result': validation_result
            })
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating dataset: {str(e)}")
            return self._get_default_validation_result()
    
    def export_dataset(self, dataset_id: str, format: str = 'csv',
                      include_metadata: bool = True) -> Optional[str]:
        """
        Export dataset to file
        
        Args:
            dataset_id: Dataset identifier
            format: Export format ('csv', 'json', 'excel')
            include_metadata: Whether to include metadata
            
        Returns:
            File path or None if error
        """
        try:
            if dataset_id not in self.datasets:
                return None
            
            dataset = self.datasets[dataset_id]
            data_entries = self.raw_data.get(dataset_id, [])
            
            if not data_entries:
                return None
            
            # Prepare data for export
            export_data = []
            for entry in data_entries:
                row = {
                    'data_id': entry['data_id'],
                    'participant_id': entry['participant_id'],
                    'timestamp': entry['timestamp'].isoformat()
                }
                
                # Add data fields
                row.update(entry['data'])
                
                # Add metadata if requested
                if include_metadata:
                    row['validation_quality'] = entry['validation_result'].quality_score
                    row['validation_errors'] = '; '.join(entry['validation_result'].validation_errors)
                
                export_data.append(row)
            
            # Create filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{dataset.name}_{timestamp}.{format}"
            
            # Export based on format
            if format.lower() == 'csv':
                df = pd.DataFrame(export_data)
                df.to_csv(filename, index=False)
            elif format.lower() == 'json':
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format.lower() == 'excel':
                df = pd.DataFrame(export_data)
                df.to_excel(filename, index=False)
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
            
            logger.info(f"Exported dataset {dataset_id} to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting dataset: {str(e)}")
            return None
    
    def get_dataset_statistics(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a dataset"""
        try:
            if dataset_id not in self.datasets:
                return None
            
            dataset = self.datasets[dataset_id]
            data_entries = self.raw_data.get(dataset_id, [])
            
            # Calculate statistics
            total_entries = len(data_entries)
            unique_participants = len(set(entry['participant_id'] for entry in data_entries))
            
            # Time range
            if data_entries:
                timestamps = [entry['timestamp'] for entry in data_entries]
                time_range = {
                    'start': min(timestamps).isoformat(),
                    'end': max(timestamps).isoformat(),
                    'duration_days': (max(timestamps) - min(timestamps)).days
                }
            else:
                time_range = {'start': None, 'end': None, 'duration_days': 0}
            
            # Data quality distribution
            quality_scores = [entry['validation_result'].quality_score for entry in data_entries]
            quality_stats = {
                'mean': np.mean(quality_scores) if quality_scores else 0,
                'min': np.min(quality_scores) if quality_scores else 0,
                'max': np.max(quality_scores) if quality_scores else 0,
                'std': np.std(quality_scores) if quality_scores else 0
            }
            
            return {
                'dataset_id': dataset_id,
                'name': dataset.name,
                'data_type': dataset.data_type.value,
                'total_entries': total_entries,
                'unique_participants': unique_participants,
                'time_range': time_range,
                'quality_score': dataset.quality_score,
                'quality_stats': quality_stats,
                'created_at': dataset.created_at.isoformat(),
                'last_updated': dataset.last_updated.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dataset statistics: {str(e)}")
            return None
    
    def _validate_data(self, data: Dict[str, Any], data_type: DataType) -> DataValidationResult:
        """Validate individual data entry"""
        try:
            validation_errors = []
            
            # Type-specific validation
            if data_type == DataType.LEARNING_OUTCOMES:
                required_fields = ['pre_test_score', 'post_test_score', 'learning_time']
                for field in required_fields:
                    if field not in data:
                        validation_errors.append(f"Missing required field: {field}")
                    elif not isinstance(data[field], (int, float)):
                        validation_errors.append(f"Invalid type for {field}: expected number")
            
            elif data_type == DataType.ENGAGEMENT_METRICS:
                required_fields = ['engagement_score', 'attention_score']
                for field in required_fields:
                    if field not in data:
                        validation_errors.append(f"Missing required field: {field}")
                    elif not isinstance(data[field], (int, float)):
                        validation_errors.append(f"Invalid type for {field}: expected number")
                    elif not (0 <= data[field] <= 1):
                        validation_errors.append(f"Value out of range for {field}: expected 0-1")
            
            # General validation
            for key, value in data.items():
                if value is None:
                    validation_errors.append(f"Null value for field: {key}")
            
            # Calculate quality metrics
            data_completeness = 1.0 - (len(validation_errors) / max(1, len(data)))
            data_consistency = 1.0  # Simplified - would check for consistency patterns
            
            quality_score = (data_completeness + data_consistency) / 2
            
            # Determine quality level
            if quality_score >= 0.9:
                quality_level = DataQuality.EXCELLENT
            elif quality_score >= 0.7:
                quality_level = DataQuality.GOOD
            elif quality_score >= 0.5:
                quality_level = DataQuality.FAIR
            elif quality_score >= 0.3:
                quality_level = DataQuality.POOR
            else:
                quality_level = DataQuality.INVALID
            
            is_valid = quality_level in [DataQuality.EXCELLENT, DataQuality.GOOD]
            
            return DataValidationResult(
                is_valid=is_valid,
                quality_score=quality_score,
                quality_level=quality_level,
                validation_errors=validation_errors,
                data_completeness=data_completeness,
                data_consistency=data_consistency,
                recommendations=[]
            )
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return self._get_default_validation_result()
    
    def _calculate_data_completeness(self, data_entries: List[Dict]) -> float:
        """Calculate data completeness score"""
        try:
            if not data_entries:
                return 0.0
            
            total_fields = 0
            missing_fields = 0
            
            for entry in data_entries:
                data = entry['data']
                total_fields += len(data)
                missing_fields += sum(1 for value in data.values() if value is None)
            
            if total_fields == 0:
                return 0.0
            
            completeness = 1.0 - (missing_fields / total_fields)
            return completeness
            
        except Exception as e:
            logger.error(f"Error calculating data completeness: {str(e)}")
            return 0.0
    
    def _calculate_data_consistency(self, data_entries: List[Dict]) -> float:
        """Calculate data consistency score"""
        try:
            if len(data_entries) < 2:
                return 1.0
            
            # Simplified consistency check - would implement more sophisticated checks
            consistency_score = 1.0
            
            # Check for outliers in numeric fields
            numeric_fields = {}
            for entry in data_entries:
                for key, value in entry['data'].items():
                    if isinstance(value, (int, float)):
                        if key not in numeric_fields:
                            numeric_fields[key] = []
                        numeric_fields[key].append(value)
            
            for field, values in numeric_fields.items():
                if len(values) > 2:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    outliers = sum(1 for v in values if abs(v - mean_val) > 3 * std_val)
                    field_consistency = 1.0 - (outliers / len(values))
                    consistency_score = min(consistency_score, field_consistency)
            
            return consistency_score
            
        except Exception as e:
            logger.error(f"Error calculating data consistency: {str(e)}")
            return 0.0
    
    def _generate_data_quality_recommendations(self, quality_score: float,
                                             data_completeness: float,
                                             data_consistency: float,
                                             validation_errors: List[str]) -> List[str]:
        """Generate recommendations for improving data quality"""
        try:
            recommendations = []
            
            if quality_score < 0.7:
                recommendations.append("Overall data quality needs improvement")
            
            if data_completeness < 0.8:
                recommendations.append("Improve data collection to reduce missing values")
                recommendations.append("Implement data validation at the point of entry")
            
            if data_consistency < 0.8:
                recommendations.append("Review data collection procedures for consistency")
                recommendations.append("Implement data quality checks and validation rules")
            
            if validation_errors:
                recommendations.append("Address specific validation errors in the data")
                recommendations.append("Update data collection forms to prevent common errors")
            
            if not recommendations:
                recommendations.append("Data quality is good - continue current practices")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating data quality recommendations: {str(e)}")
            return ["Review data quality and implement improvements"]
    
    def _update_dataset_statistics(self, dataset_id: str):
        """Update dataset statistics"""
        try:
            dataset = self.datasets[dataset_id]
            data_entries = self.raw_data.get(dataset_id, [])
            
            # Update counts
            dataset.data_points = len(data_entries)
            dataset.participant_count = len(set(entry['participant_id'] for entry in data_entries))
            dataset.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating dataset statistics: {str(e)}")
    
    def _get_default_validation_result(self) -> DataValidationResult:
        """Return default validation result"""
        return DataValidationResult(
            is_valid=False,
            quality_score=0.0,
            quality_level=DataQuality.INVALID,
            validation_errors=["No data provided"],
            data_completeness=0.0,
            data_consistency=0.0,
            recommendations=["Provide valid data for analysis"]
        )
    
    def get_research_statistics(self) -> Dict[str, int]:
        """Get research data management statistics"""
        try:
            return {
                'total_participants': len(self.participants),
                'total_datasets': len(self.datasets),
                'total_data_entries': sum(len(entries) for entries in self.raw_data.values()),
                'total_validations': len(self.validation_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting research statistics: {str(e)}")
            return {
                'total_participants': 0,
                'total_datasets': 0,
                'total_data_entries': 0,
                'total_validations': 0
            }
