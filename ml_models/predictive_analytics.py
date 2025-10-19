"""
Predictive Analytics and Intervention System
Early warning system for learning difficulties and proactive support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class RiskLevel(Enum):
    """Risk levels for learning difficulties"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class InterventionType(Enum):
    """Types of interventions available"""
    CONTENT_ADJUSTMENT = "content_adjustment"
    DIFFICULTY_REDUCTION = "difficulty_reduction"
    STYLE_ADAPTATION = "style_adaptation"
    PEER_SUPPORT = "peer_support"
    INSTRUCTOR_ALERT = "instructor_alert"
    BREAK_RECOMMENDATION = "break_recommendation"

@dataclass
class LearningMetrics:
    """Comprehensive learning metrics for analysis"""
    user_id: int
    timestamp: datetime
    content_completion_rate: float
    average_session_duration: float
    quiz_scores: List[float]
    engagement_score: float
    style_content_mismatch: float
    time_spent_per_content: Dict[str, float]
    error_rate: float
    help_seeking_frequency: float
    social_interaction_score: float

@dataclass
class RiskAssessment:
    """Risk assessment result"""
    user_id: int
    risk_level: RiskLevel
    risk_score: float
    contributing_factors: List[str]
    predicted_outcome: str
    confidence: float
    timestamp: datetime

@dataclass
class InterventionRecommendation:
    """Intervention recommendation"""
    intervention_type: InterventionType
    priority: int  # 1-5, 5 being highest
    description: str
    expected_impact: float
    implementation_effort: int  # 1-5
    timeline: str
    success_metrics: List[str]

class PredictiveAnalyticsEngine:
    """
    Advanced predictive analytics system for learning intervention
    """
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.9
        }
        self.intervention_history = []
        self.user_risk_profiles = {}
        self._is_fitted = False
    
    def analyze_learning_patterns(self, user_metrics: LearningMetrics) -> Dict:
        """Analyze user's learning patterns and identify anomalies"""
        
        # Convert metrics to feature vector
        features = self._extract_features(user_metrics)
        
        # Detect anomalies
        anomaly_score = self._detect_anomalies(features)
        
        # Calculate risk factors
        risk_factors = self._identify_risk_factors(user_metrics)
        
        # Predict learning trajectory
        trajectory = self._predict_learning_trajectory(user_metrics)
        
        # Generate insights
        insights = self._generate_insights(user_metrics, risk_factors, trajectory)
        
        return {
            'anomaly_score': anomaly_score,
            'risk_factors': risk_factors,
            'trajectory': trajectory,
            'insights': insights,
            'timestamp': datetime.now()
        }
    
    def assess_learning_risk(self, user_id: int, metrics: LearningMetrics) -> RiskAssessment:
        """Assess risk level for learning difficulties"""
        
        # Calculate composite risk score
        risk_score = self._calculate_risk_score(metrics)
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(metrics)
        
        # Predict outcome
        predicted_outcome = self._predict_outcome(risk_score, metrics)
        
        # Calculate confidence
        confidence = self._calculate_confidence(metrics)
        
        risk_assessment = RiskAssessment(
            user_id=user_id,
            risk_level=risk_level,
            risk_score=risk_score,
            contributing_factors=contributing_factors,
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        # Store risk profile
        self.user_risk_profiles[user_id] = risk_assessment
        
        return risk_assessment
    
    def generate_interventions(self, risk_assessment: RiskAssessment) -> List[InterventionRecommendation]:
        """Generate intervention recommendations based on risk assessment"""
        
        interventions = []
        
        if risk_assessment.risk_level == RiskLevel.CRITICAL:
            interventions.extend(self._generate_critical_interventions(risk_assessment))
        elif risk_assessment.risk_level == RiskLevel.HIGH:
            interventions.extend(self._generate_high_risk_interventions(risk_assessment))
        elif risk_assessment.risk_level == RiskLevel.MEDIUM:
            interventions.extend(self._generate_medium_risk_interventions(risk_assessment))
        else:
            interventions.extend(self._generate_low_risk_interventions(risk_assessment))
        
        # Sort by priority
        interventions.sort(key=lambda x: x.priority, reverse=True)
        
        return interventions
    
    def _extract_features(self, metrics: LearningMetrics) -> np.array:
        """Extract numerical features from learning metrics"""
        
        features = [
            metrics.content_completion_rate,
            metrics.average_session_duration / 3600,  # Convert to hours
            np.mean(metrics.quiz_scores) if metrics.quiz_scores else 0,
            metrics.engagement_score,
            metrics.style_content_mismatch,
            metrics.error_rate,
            metrics.help_seeking_frequency,
            metrics.social_interaction_score
        ]
        
        return np.array(features).reshape(1, -1)
    
    def _detect_anomalies(self, features: np.array) -> float:
        """Detect anomalies in learning patterns"""
        
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Fit anomaly detector if not already fitted
        if not self._is_fitted:
            # Use dummy data for initial fitting with correct number of features
            dummy_data = np.random.normal(0, 1, (100, features.shape[1]))
            self.anomaly_detector.fit(dummy_data)
            self._is_fitted = True
        
        # Calculate anomaly score
        try:
            anomaly_score = self.anomaly_detector.decision_function(features)[0]
            # Normalize to 0-1 scale
            return max(0, min(1, (anomaly_score + 0.5)))
        except Exception as e:
            # If there's an error, return a default score
            print(f"Warning: Anomaly detection failed: {e}")
            return 0.5
    
    def _identify_risk_factors(self, metrics: LearningMetrics) -> List[str]:
        """Identify specific risk factors in learning patterns"""
        
        risk_factors = []
        
        # Low completion rate
        if metrics.content_completion_rate < 0.5:
            risk_factors.append("Low content completion rate")
        
        # Short session duration
        if metrics.average_session_duration < 300:  # Less than 5 minutes
            risk_factors.append("Very short learning sessions")
        
        # Poor quiz performance
        if metrics.quiz_scores and np.mean(metrics.quiz_scores) < 0.6:
            risk_factors.append("Below average quiz performance")
        
        # Low engagement
        if metrics.engagement_score < 0.4:
            risk_factors.append("Low engagement with content")
        
        # High style mismatch
        if metrics.style_content_mismatch > 0.7:
            risk_factors.append("Content doesn't match learning style")
        
        # High error rate
        if metrics.error_rate > 0.3:
            risk_factors.append("High error rate in activities")
        
        # Frequent help seeking
        if metrics.help_seeking_frequency > 0.8:
            risk_factors.append("Frequent help seeking behavior")
        
        # Low social interaction
        if metrics.social_interaction_score < 0.2:
            risk_factors.append("Limited social learning engagement")
        
        return risk_factors
    
    def _calculate_risk_score(self, metrics: LearningMetrics) -> float:
        """Calculate composite risk score"""
        
        # Weighted combination of risk indicators
        weights = {
            'completion_rate': 0.25,
            'session_duration': 0.15,
            'quiz_performance': 0.20,
            'engagement': 0.15,
            'style_mismatch': 0.10,
            'error_rate': 0.10,
            'help_seeking': 0.05
        }
        
        # Calculate individual risk components
        completion_risk = 1 - metrics.content_completion_rate
        duration_risk = max(0, 1 - (metrics.average_session_duration / 1800))  # 30 min ideal
        quiz_risk = 1 - (np.mean(metrics.quiz_scores) if metrics.quiz_scores else 0.5)
        engagement_risk = 1 - metrics.engagement_score
        style_risk = metrics.style_content_mismatch
        error_risk = metrics.error_rate
        help_risk = min(1, metrics.help_seeking_frequency)
        
        # Calculate weighted risk score
        risk_score = (
            weights['completion_rate'] * completion_risk +
            weights['session_duration'] * duration_risk +
            weights['quiz_performance'] * quiz_risk +
            weights['engagement'] * engagement_risk +
            weights['style_mismatch'] * style_risk +
            weights['error_rate'] * error_risk +
            weights['help_seeking'] * help_risk
        )
        
        return min(max(risk_score, 0), 1)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on score"""
        
        if risk_score >= self.risk_thresholds['critical']:
            return RiskLevel.CRITICAL
        elif risk_score >= self.risk_thresholds['high']:
            return RiskLevel.HIGH
        elif risk_score >= self.risk_thresholds['medium']:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _identify_contributing_factors(self, metrics: LearningMetrics) -> List[str]:
        """Identify factors contributing to risk"""
        return self._identify_risk_factors(metrics)
    
    def _predict_outcome(self, risk_score: float, metrics: LearningMetrics) -> str:
        """Predict likely learning outcome"""
        
        if risk_score >= 0.8:
            return "High risk of course failure or dropout"
        elif risk_score >= 0.6:
            return "Moderate risk of falling behind"
        elif risk_score >= 0.4:
            return "May need additional support"
        else:
            return "Likely to succeed with current approach"
    
    def _calculate_confidence(self, metrics: LearningMetrics) -> float:
        """Calculate confidence in risk assessment"""
        
        # Confidence based on data quality and consistency
        data_quality = 0.5  # Base quality
        
        # More data points = higher confidence
        if len(metrics.quiz_scores) > 5:
            data_quality += 0.2
        
        # Consistent patterns = higher confidence
        if metrics.quiz_scores:
            quiz_consistency = 1 - np.std(metrics.quiz_scores)
            data_quality += quiz_consistency * 0.3
        
        return min(data_quality, 1.0)
    
    def _generate_critical_interventions(self, risk_assessment: RiskAssessment) -> List[InterventionRecommendation]:
        """Generate interventions for critical risk level"""
        
        interventions = [
            InterventionRecommendation(
                intervention_type=InterventionType.INSTRUCTOR_ALERT,
                priority=5,
                description="Immediate instructor notification for one-on-one support",
                expected_impact=0.9,
                implementation_effort=2,
                timeline="Immediate",
                success_metrics=["Instructor contact within 24 hours", "Student response to outreach"]
            ),
            InterventionRecommendation(
                intervention_type=InterventionType.DIFFICULTY_REDUCTION,
                priority=4,
                description="Reduce content difficulty and provide additional scaffolding",
                expected_impact=0.8,
                implementation_effort=3,
                timeline="Within 48 hours",
                success_metrics=["Content completion rate", "Student engagement"]
            ),
            InterventionRecommendation(
                intervention_type=InterventionType.PEER_SUPPORT,
                priority=4,
                description="Assign study partner or peer mentor",
                expected_impact=0.7,
                implementation_effort=2,
                timeline="Within 72 hours",
                success_metrics=["Peer interaction frequency", "Collaborative learning activities"]
            )
        ]
        
        return interventions
    
    def _generate_high_risk_interventions(self, risk_assessment: RiskAssessment) -> List[InterventionRecommendation]:
        """Generate interventions for high risk level"""
        
        interventions = [
            InterventionRecommendation(
                intervention_type=InterventionType.CONTENT_ADJUSTMENT,
                priority=4,
                description="Adjust content delivery to better match learning style",
                expected_impact=0.7,
                implementation_effort=3,
                timeline="Within 1 week",
                success_metrics=["Style-content alignment", "Engagement improvement"]
            ),
            InterventionRecommendation(
                intervention_type=InterventionType.PEER_SUPPORT,
                priority=3,
                description="Connect with study group or learning community",
                expected_impact=0.6,
                implementation_effort=2,
                timeline="Within 3 days",
                success_metrics=["Group participation", "Social learning engagement"]
            ),
            InterventionRecommendation(
                intervention_type=InterventionType.BREAK_RECOMMENDATION,
                priority=2,
                description="Suggest structured breaks and study schedule",
                expected_impact=0.5,
                implementation_effort=1,
                timeline="Immediate",
                success_metrics=["Session duration", "Burnout prevention"]
            )
        ]
        
        return interventions
    
    def _generate_medium_risk_interventions(self, risk_assessment: RiskAssessment) -> List[InterventionRecommendation]:
        """Generate interventions for medium risk level"""
        
        interventions = [
            InterventionRecommendation(
                intervention_type=InterventionType.STYLE_ADAPTATION,
                priority=3,
                description="Provide alternative content formats for better learning",
                expected_impact=0.6,
                implementation_effort=2,
                timeline="Within 1 week",
                success_metrics=["Content preference alignment", "Learning effectiveness"]
            ),
            InterventionRecommendation(
                intervention_type=InterventionType.BREAK_RECOMMENDATION,
                priority=2,
                description="Optimize study schedule and break patterns",
                expected_impact=0.4,
                implementation_effort=1,
                timeline="Immediate",
                success_metrics=["Study consistency", "Energy levels"]
            )
        ]
        
        return interventions
    
    def _generate_low_risk_interventions(self, risk_assessment: RiskAssessment) -> List[InterventionRecommendation]:
        """Generate interventions for low risk level"""
        
        interventions = [
            InterventionRecommendation(
                intervention_type=InterventionType.CONTENT_ADJUSTMENT,
                priority=2,
                description="Fine-tune content recommendations for optimal learning",
                expected_impact=0.4,
                implementation_effort=1,
                timeline="Within 2 weeks",
                success_metrics=["Learning efficiency", "Content satisfaction"]
            )
        ]
        
        return interventions
    
    def _predict_learning_trajectory(self, metrics: LearningMetrics) -> Dict:
        """Predict future learning trajectory"""
        
        # Simple trajectory prediction based on current trends
        if metrics.quiz_scores:
            recent_trend = np.polyfit(range(len(metrics.quiz_scores)), metrics.quiz_scores, 1)[0]
        else:
            recent_trend = 0
        
        # Predict based on current performance and trends
        if recent_trend > 0.1:
            trajectory = "Improving performance"
        elif recent_trend < -0.1:
            trajectory = "Declining performance"
        else:
            trajectory = "Stable performance"
        
        return {
            'trend': recent_trend,
            'prediction': trajectory,
            'confidence': 0.7  # Placeholder confidence
        }
    
    def _generate_insights(self, metrics: LearningMetrics, risk_factors: List[str], trajectory: Dict) -> List[str]:
        """Generate actionable insights from analysis"""
        
        insights = []
        
        # Performance insights
        if metrics.quiz_scores and np.mean(metrics.quiz_scores) > 0.8:
            insights.append("Strong performance on assessments - consider advanced content")
        
        if metrics.engagement_score > 0.8:
            insights.append("High engagement levels - good learning momentum")
        
        # Risk factor insights
        if "Low content completion rate" in risk_factors:
            insights.append("Consider breaking content into smaller, manageable chunks")
        
        if "Content doesn't match learning style" in risk_factors:
            insights.append("Content format may not align with your learning preferences")
        
        # Trajectory insights
        if trajectory['prediction'] == "Improving performance":
            insights.append("Learning progress is accelerating - maintain current approach")
        elif trajectory['prediction'] == "Declining performance":
            insights.append("Performance trend is declining - consider intervention strategies")
        
        return insights
    
    def track_intervention_effectiveness(self, user_id: int, intervention: InterventionRecommendation) -> Dict:
        """Track effectiveness of implemented interventions"""
        
        # This would integrate with actual user data in production
        effectiveness_metrics = {
            'intervention_id': f"{user_id}_{intervention.intervention_type.value}",
            'implementation_date': datetime.now(),
            'success_metrics': intervention.success_metrics,
            'effectiveness_score': 0.0,  # Would be calculated from actual data
            'status': 'monitoring'
        }
        
        self.intervention_history.append(effectiveness_metrics)
        
        return effectiveness_metrics

# Example usage
if __name__ == "__main__":
    # Initialize predictive analytics engine
    analytics = PredictiveAnalyticsEngine()
    
    # Example learning metrics
    metrics = LearningMetrics(
        user_id=1,
        timestamp=datetime.now(),
        content_completion_rate=0.3,  # Low completion
        average_session_duration=180,  # 3 minutes - very short
        quiz_scores=[0.4, 0.5, 0.3, 0.6],  # Below average
        engagement_score=0.2,  # Low engagement
        style_content_mismatch=0.8,  # High mismatch
        time_spent_per_content={'video': 120, 'text': 60, 'interactive': 0},
        error_rate=0.4,  # High error rate
        help_seeking_frequency=0.9,  # Frequent help seeking
        social_interaction_score=0.1  # Low social interaction
    )
    
    # Analyze patterns
    analysis = analytics.analyze_learning_patterns(metrics)
    print("Learning Pattern Analysis:")
    print(f"Anomaly Score: {analysis['anomaly_score']:.3f}")
    print(f"Risk Factors: {analysis['risk_factors']}")
    print(f"Insights: {analysis['insights']}")
    
    # Assess risk
    risk_assessment = analytics.assess_learning_risk(1, metrics)
    print(f"\nRisk Assessment:")
    print(f"Risk Level: {risk_assessment.risk_level.value}")
    print(f"Risk Score: {risk_assessment.risk_score:.3f}")
    print(f"Predicted Outcome: {risk_assessment.predicted_outcome}")
    
    # Generate interventions
    interventions = analytics.generate_interventions(risk_assessment)
    print(f"\nIntervention Recommendations:")
    for i, intervention in enumerate(interventions, 1):
        print(f"{i}. {intervention.description}")
        print(f"   Priority: {intervention.priority}, Impact: {intervention.expected_impact:.1f}")
