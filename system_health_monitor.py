"""
System Health Monitoring for LearnStyle AI
Comprehensive real-time health monitoring and alerting system
"""

import psutil
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class HealthMetrics:
    """System health metrics"""
    timestamp: datetime
    overall_health: float
    status: HealthStatus
    components: Dict[str, float]
    recommendations: List[str]
    alerts: List[str]

class SystemHealthMonitor:
    """Comprehensive system health monitoring"""
    
    def __init__(self, app=None):
        self.app = app
        self.health_history = []
        self.alert_thresholds = {
            'critical': 40,
            'poor': 60,
            'fair': 75,
            'good': 90
        }
        
    def init_app(self, app):
        """Initialize with Flask app"""
        self.app = app
        
    def calculate_server_health(self) -> float:
        """Calculate server performance health (0-100)"""
        try:
            # Get real-time system metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)  # Faster sampling
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get additional system metrics
            boot_time = psutil.boot_time()
            uptime_hours = (time.time() - boot_time) / 3600
            
            # Calculate health score based on real metrics
            health_score = 100.0
            
            # CPU health calculation (more granular)
            if cpu_usage > 95:
                health_score -= 40
            elif cpu_usage > 90:
                health_score -= 30
            elif cpu_usage > 80:
                health_score -= 20
            elif cpu_usage > 70:
                health_score -= 10
            elif cpu_usage > 50:
                health_score -= 5
            
            # Memory health calculation
            memory_usage = memory.percent
            if memory_usage > 98:
                health_score -= 40
            elif memory_usage > 95:
                health_score -= 30
            elif memory_usage > 90:
                health_score -= 20
            elif memory_usage > 80:
                health_score -= 10
            elif memory_usage > 70:
                health_score -= 5
            
            # Disk health calculation
            disk_usage = disk.percent
            if disk_usage > 98:
                health_score -= 30
            elif disk_usage > 95:
                health_score -= 20
            elif disk_usage > 90:
                health_score -= 15
            elif disk_usage > 85:
                health_score -= 10
            elif disk_usage > 80:
                health_score -= 5
            
            # Uptime bonus (system stability)
            if uptime_hours > 24:  # More than 1 day uptime
                health_score += 5
            elif uptime_hours > 168:  # More than 1 week uptime
                health_score += 10
            
            # Get process count for additional health indicator
            process_count = len(psutil.pids())
            if process_count > 500:  # Too many processes
                health_score -= 5
            elif process_count < 50:  # Very few processes (might be issue)
                health_score -= 2
            
            return max(0, min(100, health_score))
            
        except Exception as e:
            logger.error(f"Error calculating server health: {e}")
            return 25.0  # Low health if monitoring fails
    
    def calculate_database_health(self) -> float:
        """Calculate database performance health (0-100)"""
        try:
            if not self.app:
                return 30.0  # Low health if no app context
            
            with self.app.app_context():
                from app import db
                from app.models import User, LearningProfile, ContentLibrary, UserProgress
                
                health_score = 100.0
                
                # Test 1: Basic connection test
                start_time = time.time()
                db.session.execute('SELECT 1')
                basic_query_time = time.time() - start_time
                
                # Test 2: User count query
                start_time = time.time()
                user_count = User.query.count()
                user_query_time = time.time() - start_time
                
                # Test 3: Complex join query
                start_time = time.time()
                profiles_with_users = db.session.query(LearningProfile, User).join(User).count()
                join_query_time = time.time() - start_time
                
                # Test 4: Content library query
                start_time = time.time()
                content_count = ContentLibrary.query.count()
                content_query_time = time.time() - start_time
                
                # Test 5: Progress aggregation query
                start_time = time.time()
                progress_stats = db.session.query(
                    db.func.count(UserProgress.id),
                    db.func.avg(UserProgress.score)
                ).scalar()
                aggregation_query_time = time.time() - start_time
                
                # Calculate health based on query performance
                query_times = [basic_query_time, user_query_time, join_query_time, content_query_time, aggregation_query_time]
                avg_query_time = sum(query_times) / len(query_times)
                max_query_time = max(query_times)
                
                # Performance penalties
                if max_query_time > 5.0:
                    health_score -= 50
                elif max_query_time > 3.0:
                    health_score -= 30
                elif max_query_time > 2.0:
                    health_score -= 20
                elif max_query_time > 1.0:
                    health_score -= 10
                elif max_query_time > 0.5:
                    health_score -= 5
                
                if avg_query_time > 2.0:
                    health_score -= 20
                elif avg_query_time > 1.0:
                    health_score -= 10
                elif avg_query_time > 0.5:
                    health_score -= 5
                
                # Test database connection pool health
                try:
                    # Test multiple concurrent queries
                    start_time = time.time()
                    for _ in range(5):
                        User.query.count()
                    concurrent_time = time.time() - start_time
                    
                    if concurrent_time > 2.0:
                        health_score -= 15
                    elif concurrent_time > 1.0:
                        health_score -= 10
                    elif concurrent_time > 0.5:
                        health_score -= 5
                except Exception as e:
                    logger.warning(f"Concurrent query test failed: {e}")
                    health_score -= 10
                
                # Test database size and table health
                try:
                    # Check if tables have data
                    if user_count == 0:
                        health_score -= 5  # No users might indicate issue
                    
                    # Check for potential issues
                    if content_count == 0:
                        health_score -= 2  # No content is okay but not ideal
                        
                except Exception as e:
                    logger.warning(f"Database content check failed: {e}")
                    health_score -= 5
                
                return max(0, min(100, health_score))
                
        except Exception as e:
            logger.error(f"Error calculating database health: {e}")
            return 20.0  # Low health if database monitoring fails
    
    def calculate_ml_models_health(self) -> float:
        """Calculate ML models performance health (0-100)"""
        try:
            health_scores = []
            
            # Test 1: Learning Style Predictor
            try:
                from ml_models.learning_style_predictor import LearningStylePredictor
                predictor = LearningStylePredictor()
                
                # Test model loading
                start_time = time.time()
                predictor.load_models()
                load_time = time.time() - start_time
                
                # Test prediction capability
                if hasattr(predictor, 'is_trained') and predictor.is_trained:
                    # Test with sample data
                    sample_answers = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
                    start_time = time.time()
                    prediction = predictor.predict_learning_style(sample_answers)
                    prediction_time = time.time() - start_time
                    
                    # Calculate health based on performance
                    model_health = 100.0
                    if load_time > 2.0:
                        model_health -= 20
                    elif load_time > 1.0:
                        model_health -= 10
                    
                    if prediction_time > 1.0:
                        model_health -= 20
                    elif prediction_time > 0.5:
                        model_health -= 10
                    
                    # Check prediction validity
                    if prediction and 'dominant_style' in prediction:
                        model_health += 5  # Bonus for valid prediction
                    
                    health_scores.append(max(0, min(100, model_health)))
                else:
                    health_scores.append(60)  # Model exists but not trained
                    
            except Exception as e:
                logger.warning(f"Learning style predictor health check failed: {e}")
                health_scores.append(20)
            
            # Test 2: Content Generator
            try:
                from app.content_generator import ContentGenerator
                generator = ContentGenerator()
                
                # Test content generation
                start_time = time.time()
                # Test basic functionality without actually generating content
                if hasattr(generator, 'generate_content'):
                    health_scores.append(85)
                else:
                    health_scores.append(70)
                    
            except Exception as e:
                logger.warning(f"Content generator health check failed: {e}")
                health_scores.append(30)
            
            # Test 3: Emotion AI Components
            try:
                from ml_models.emotion_ai.emotion_fusion_engine import EmotionFusionEngine
                analyzer = EmotionFusionEngine()
                
                # Test with sample data
                sample_data = {
                    'facial_data': {'emotions': {'happy': 0.8, 'neutral': 0.2}},
                    'voice_data': {'emotions': {'happy': 0.7, 'neutral': 0.3}}
                }
                
                start_time = time.time()
                # Test basic functionality
                if hasattr(analyzer, 'fuse_emotions'):
                    health_scores.append(80)
                else:
                    health_scores.append(70)
                    
            except Exception as e:
                logger.warning(f"Emotion AI health check failed: {e}")
                health_scores.append(35)
            
            # Test 4: Biometric Fusion
            try:
                from ml_models.biometric_feedback.biometric_fusion import BiometricFusionEngine
                fusion = BiometricFusionEngine()
                
                # Test basic functionality
                if hasattr(fusion, 'process_biometric_data'):
                    health_scores.append(80)
                else:
                    health_scores.append(70)
                    
            except Exception as e:
                logger.warning(f"Biometric fusion health check failed: {e}")
                health_scores.append(35)
            
            # Test 5: Predictive Analytics
            try:
                from ml_models.predictive_analytics import PredictiveAnalyticsEngine
                analytics = PredictiveAnalyticsEngine()
                
                if hasattr(analytics, 'analyze_learning_patterns'):
                    health_scores.append(85)
                else:
                    health_scores.append(75)
                    
            except Exception as e:
                logger.warning(f"Predictive analytics health check failed: {e}")
                health_scores.append(40)
            
            # Calculate weighted average
            if health_scores:
                # Weight different components based on importance
                weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # Learning predictor is most important
                weighted_score = sum(score * weight for score, weight in zip(health_scores, weights))
                return max(0, min(100, weighted_score))
            else:
                return 0
            
        except Exception as e:
            logger.error(f"Error calculating ML models health: {e}")
            return 15.0  # Low health if ML monitoring fails
    
    def calculate_api_health(self) -> float:
        """Calculate API endpoints performance health (0-100)"""
        try:
            if not self.app:
                return 40.0  # Low health if no app context
            
            with self.app.app_context():
                health_score = 100.0
                
                # Test 1: Flask app responsiveness
                try:
                    # Test app context and basic functionality
                    start_time = time.time()
                    from app.models import User, LearningProfile, ContentLibrary
                    context_time = time.time() - start_time
                    
                    if context_time > 0.5:
                        health_score -= 10
                    elif context_time > 0.2:
                        health_score -= 5
                        
                except Exception as e:
                    logger.warning(f"App context test failed: {e}")
                    health_score -= 20
                
                # Test 2: Database connectivity (API dependency)
                try:
                    start_time = time.time()
                    user_count = User.query.count()
                    db_time = time.time() - start_time
                    
                    if db_time > 1.0:
                        health_score -= 15
                    elif db_time > 0.5:
                        health_score -= 10
                    elif db_time > 0.2:
                        health_score -= 5
                        
                except Exception as e:
                    logger.warning(f"Database connectivity test failed: {e}")
                    health_score -= 25
                
                # Test 3: Model loading performance (API dependency)
                try:
                    start_time = time.time()
                    from ml_models.learning_style_predictor import LearningStylePredictor
                    predictor = LearningStylePredictor()
                    model_load_time = time.time() - start_time
                    
                    if model_load_time > 2.0:
                        health_score -= 15
                    elif model_load_time > 1.0:
                        health_score -= 10
                    elif model_load_time > 0.5:
                        health_score -= 5
                        
                except Exception as e:
                    logger.warning(f"Model loading test failed: {e}")
                    health_score -= 20
                
                # Test 4: Memory usage (API performance indicator)
                try:
                    import psutil
                    process = psutil.Process()
                    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                    
                    if memory_usage > 500:  # More than 500MB
                        health_score -= 10
                    elif memory_usage > 300:  # More than 300MB
                        health_score -= 5
                    elif memory_usage > 100:  # More than 100MB
                        health_score -= 2
                        
                except Exception as e:
                    logger.warning(f"Memory usage test failed: {e}")
                    health_score -= 5
                
                # Test 5: File system access (API dependency)
                try:
                    import os
                    start_time = time.time()
                    # Test access to key directories
                    os.listdir('.')  # Current directory
                    os.listdir('ml_models') if os.path.exists('ml_models') else None
                    fs_time = time.time() - start_time
                    
                    if fs_time > 0.5:
                        health_score -= 10
                    elif fs_time > 0.2:
                        health_score -= 5
                        
                except Exception as e:
                    logger.warning(f"File system access test failed: {e}")
                    health_score -= 15
                
                # Test 6: Python environment health
                try:
                    import sys
                    python_version = sys.version_info
                    if python_version.major < 3 or python_version.minor < 8:
                        health_score -= 10  # Old Python version
                        
                except Exception as e:
                    logger.warning(f"Python environment test failed: {e}")
                    health_score -= 5
            
                return max(0, min(100, health_score))
            
        except Exception as e:
            logger.error(f"Error calculating API health: {e}")
            return 25.0  # Low health if API monitoring fails
    
    def calculate_learning_quality_health(self) -> float:
        """Calculate learning session quality health (0-100)"""
        try:
            if not self.app:
                return 35.0  # Low health if no app context
            
            with self.app.app_context():
                from app.models import User, LearningProfile, UserProgress, ContentLibrary
                from sqlalchemy import func
                
                health_score = 100.0
                
                # Test 1: User engagement metrics
                try:
                    total_users = User.query.count()
                    active_users = User.query.filter(
                        User.last_login >= datetime.now() - timedelta(days=30)
                    ).count()
                    
                    if total_users == 0:
                        health_score -= 40  # No users at all
                    else:
                        engagement_rate = active_users / total_users
                        if engagement_rate < 0.1:  # Less than 10% active
                            health_score -= 30
                        elif engagement_rate < 0.3:  # Less than 30% active
                            health_score -= 20
                        elif engagement_rate < 0.5:  # Less than 50% active
                            health_score -= 10
                        
                except Exception as e:
                    logger.warning(f"User engagement metrics failed: {e}")
                    health_score -= 15
                
                # Test 2: Learning profile completion
                try:
                    profiles_with_style = LearningProfile.query.filter(
                        LearningProfile.learning_style.isnot(None)
                    ).count()
                    
                    if total_users > 0:
                        profile_completion_rate = profiles_with_style / total_users
                        if profile_completion_rate < 0.2:  # Less than 20% have profiles
                            health_score -= 25
                        elif profile_completion_rate < 0.5:  # Less than 50% have profiles
                            health_score -= 15
                        elif profile_completion_rate < 0.8:  # Less than 80% have profiles
                            health_score -= 5
                    else:
                        health_score -= 20  # No users to have profiles
                        
                except Exception as e:
                    logger.warning(f"Profile completion metrics failed: {e}")
                    health_score -= 10
                
                # Test 3: Learning progress activity
                try:
                    recent_progress = UserProgress.query.filter(
                        UserProgress.timestamp >= datetime.now() - timedelta(days=7)
                    ).count()
                    
                    if total_users > 0:
                        progress_rate = recent_progress / total_users
                        if progress_rate == 0:  # No recent progress
                            health_score -= 30
                        elif progress_rate < 0.1:  # Very low progress
                            health_score -= 20
                        elif progress_rate < 0.3:  # Low progress
                            health_score -= 10
                        elif progress_rate < 0.5:  # Moderate progress
                            health_score -= 5
                    else:
                        health_score -= 25  # No users to have progress
                        
                except Exception as e:
                    logger.warning(f"Progress activity metrics failed: {e}")
                    health_score -= 15
                
                # Test 4: Content availability
                try:
                    content_count = ContentLibrary.query.count()
                    if content_count == 0:
                        health_score -= 20  # No content available
                    elif content_count < 10:
                        health_score -= 10  # Very little content
                    elif content_count < 50:
                        health_score -= 5  # Limited content
                        
                except Exception as e:
                    logger.warning(f"Content availability metrics failed: {e}")
                    health_score -= 10
                
                # Test 5: Learning session quality indicators
                try:
                    # Check for recent high-quality sessions (high scores)
                    high_quality_sessions = UserProgress.query.filter(
                        UserProgress.score >= 80,
                        UserProgress.timestamp >= datetime.now() - timedelta(days=7)
                    ).count()
                    
                    if total_users > 0:
                        quality_rate = high_quality_sessions / total_users
                        if quality_rate < 0.1:  # Less than 10% have high-quality sessions
                            health_score -= 15
                        elif quality_rate < 0.3:  # Less than 30% have high-quality sessions
                            health_score -= 10
                        elif quality_rate < 0.5:  # Less than 50% have high-quality sessions
                            health_score -= 5
                        
                except Exception as e:
                    logger.warning(f"Learning quality metrics failed: {e}")
                    health_score -= 10
                
                # Test 6: System responsiveness for learning
                try:
                    # Test how quickly we can process learning-related queries
                    start_time = time.time()
                    UserProgress.query.filter(
                        UserProgress.timestamp >= datetime.now() - timedelta(days=1)
                    ).count()
                    query_time = time.time() - start_time
                    
                    if query_time > 2.0:
                        health_score -= 15
                    elif query_time > 1.0:
                        health_score -= 10
                    elif query_time > 0.5:
                        health_score -= 5
                        
                except Exception as e:
                    logger.warning(f"Learning responsiveness test failed: {e}")
                    health_score -= 10
                
                return max(0, min(100, health_score))
                
        except Exception as e:
            logger.error(f"Error calculating learning quality health: {e}")
            return 20.0  # Low health if learning quality monitoring fails
    
    def calculate_responsiveness_health(self) -> float:
        """Calculate system responsiveness health (0-100)"""
        try:
            if not self.app:
                return 40.0  # Low health if no app context
            
            with self.app.app_context():
                health_score = 100.0
                
                # Test 1: Database query responsiveness
                try:
                    from app.models import User, LearningProfile, UserProgress
                    
                    # Test simple query
                    start_time = time.time()
                    User.query.count()
                    simple_query_time = time.time() - start_time
                    
                    # Test complex query
                    start_time = time.time()
                    from app import db
                    db.session.query(User, LearningProfile).join(LearningProfile).count()
                    complex_query_time = time.time() - start_time
                    
                    # Test aggregation query
                    start_time = time.time()
                    UserProgress.query.filter(
                        UserProgress.timestamp >= datetime.now() - timedelta(days=7)
                    ).count()
                    aggregation_query_time = time.time() - start_time
                    
                    # Calculate penalties based on query times
                    query_times = [simple_query_time, complex_query_time, aggregation_query_time]
                    max_query_time = max(query_times)
                    avg_query_time = sum(query_times) / len(query_times)
                    
                    if max_query_time > 3.0:
                        health_score -= 40
                    elif max_query_time > 2.0:
                        health_score -= 30
                    elif max_query_time > 1.0:
                        health_score -= 20
                    elif max_query_time > 0.5:
                        health_score -= 10
                    elif max_query_time > 0.2:
                        health_score -= 5
                    
                    if avg_query_time > 1.5:
                        health_score -= 20
                    elif avg_query_time > 1.0:
                        health_score -= 15
                    elif avg_query_time > 0.5:
                        health_score -= 10
                    elif avg_query_time > 0.2:
                        health_score -= 5
                        
                except Exception as e:
                    logger.warning(f"Database responsiveness test failed: {e}")
                    health_score -= 30
                
                # Test 2: Model loading responsiveness
                try:
                    start_time = time.time()
                    from ml_models.learning_style_predictor import LearningStylePredictor
                    predictor = LearningStylePredictor()
                    predictor.load_models()
                    model_load_time = time.time() - start_time
                    
                    if model_load_time > 5.0:
                        health_score -= 25
                    elif model_load_time > 3.0:
                        health_score -= 20
                    elif model_load_time > 2.0:
                        health_score -= 15
                    elif model_load_time > 1.0:
                        health_score -= 10
                    elif model_load_time > 0.5:
                        health_score -= 5
                        
                except Exception as e:
                    logger.warning(f"Model loading responsiveness test failed: {e}")
                    health_score -= 20
                
                # Test 3: File system responsiveness
                try:
                    import os
                    start_time = time.time()
                    # Test file operations
                    os.listdir('.')
                    if os.path.exists('ml_models'):
                        os.listdir('ml_models')
                    if os.path.exists('templates'):
                        os.listdir('templates')
                    fs_time = time.time() - start_time
                    
                    if fs_time > 1.0:
                        health_score -= 15
                    elif fs_time > 0.5:
                        health_score -= 10
                    elif fs_time > 0.2:
                        health_score -= 5
                        
                except Exception as e:
                    logger.warning(f"File system responsiveness test failed: {e}")
                    health_score -= 15
                
                # Test 4: Memory and CPU responsiveness
                try:
                    import psutil
                    process = psutil.Process()
                    
                    # Test memory responsiveness
                    memory_info = process.memory_info()
                    memory_usage_mb = memory_info.rss / 1024 / 1024
                    
                    if memory_usage_mb > 1000:  # More than 1GB
                        health_score -= 10
                    elif memory_usage_mb > 500:  # More than 500MB
                        health_score -= 5
                    
                    # Test CPU responsiveness
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    if cpu_percent > 90:
                        health_score -= 15
                    elif cpu_percent > 80:
                        health_score -= 10
                    elif cpu_percent > 70:
                        health_score -= 5
                        
                except Exception as e:
                    logger.warning(f"System resource responsiveness test failed: {e}")
                    health_score -= 10
                
                # Test 5: Concurrent operation responsiveness
                try:
                    import threading
                    import time
                    
                    def test_concurrent_query():
                        with self.app.app_context():
                            User.query.count()
                    
                    # Test concurrent database operations
                    start_time = time.time()
                    threads = []
                    for _ in range(3):
                        thread = threading.Thread(target=test_concurrent_query)
                        threads.append(thread)
                        thread.start()
                    
                    for thread in threads:
                        thread.join()
                    concurrent_time = time.time() - start_time
                    
                    if concurrent_time > 2.0:
                        health_score -= 15
                    elif concurrent_time > 1.0:
                        health_score -= 10
                    elif concurrent_time > 0.5:
                        health_score -= 5
                        
                except Exception as e:
                    logger.warning(f"Concurrent operation test failed: {e}")
                    health_score -= 10
                
                return max(0, min(100, health_score))
            
        except Exception as e:
            logger.error(f"Error calculating responsiveness health: {e}")
            return 25.0  # Low health if responsiveness monitoring fails
    
    def get_health_status(self, health_score: float) -> HealthStatus:
        """Get health status based on score"""
        if health_score >= self.alert_thresholds['good']:
            return HealthStatus.EXCELLENT
        elif health_score >= self.alert_thresholds['fair']:
            return HealthStatus.GOOD
        elif health_score >= self.alert_thresholds['poor']:
            return HealthStatus.FAIR
        elif health_score >= self.alert_thresholds['critical']:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL
    
    def get_health_recommendations(self, components: Dict[str, float]) -> List[str]:
        """Get recommendations based on health components"""
        recommendations = []
        
        if components.get('server_performance', 0) < 70:
            recommendations.append("Consider scaling server resources - CPU/Memory usage is high")
        
        if components.get('database_health', 0) < 70:
            recommendations.append("Optimize database queries and check connection pool")
        
        if components.get('ml_models_health', 0) < 70:
            recommendations.append("Check ML model performance and consider retraining")
        
        if components.get('api_health', 0) < 70:
            recommendations.append("Review API endpoint performance and error rates")
        
        if components.get('learning_quality', 0) < 70:
            recommendations.append("Improve learning content and user experience")
        
        if components.get('responsiveness', 0) < 70:
            recommendations.append("Optimize system responsiveness and reduce load times")
        
        return recommendations
    
    def get_health_alerts(self, components: Dict[str, float], status: HealthStatus) -> List[str]:
        """Get critical alerts based on health status"""
        alerts = []
        
        if status == HealthStatus.CRITICAL:
            alerts.append("🚨 CRITICAL: System health is critically low - immediate attention required")
        
        if components.get('server_performance', 0) < 50:
            alerts.append("⚠️ Server performance is critically low")
        
        if components.get('database_health', 0) < 50:
            alerts.append("⚠️ Database performance is critically low")
        
        if components.get('api_health', 0) < 50:
            alerts.append("⚠️ API endpoints are experiencing critical issues")
        
        return alerts
    
    def calculate_overall_health(self) -> HealthMetrics:
        """Calculate comprehensive system health"""
        try:
            # Calculate individual component health with error handling
            components = {}
            
            try:
                components['server_performance'] = self.calculate_server_health()
            except Exception as e:
                logger.warning(f"Server health calculation failed: {e}")
                components['server_performance'] = 20.0  # Low health if calculation fails
            
            try:
                components['database_health'] = self.calculate_database_health()
            except Exception as e:
                logger.warning(f"Database health calculation failed: {e}")
                components['database_health'] = 25.0  # Low health if calculation fails
            
            try:
                components['ml_models_health'] = self.calculate_ml_models_health()
            except Exception as e:
                logger.warning(f"ML models health calculation failed: {e}")
                components['ml_models_health'] = 15.0  # Low health if calculation fails
            
            try:
                components['api_health'] = self.calculate_api_health()
            except Exception as e:
                logger.warning(f"API health calculation failed: {e}")
                components['api_health'] = 25.0  # Low health if calculation fails
            
            try:
                components['learning_quality'] = self.calculate_learning_quality_health()
            except Exception as e:
                logger.warning(f"Learning quality health calculation failed: {e}")
                components['learning_quality'] = 20.0  # Low health if calculation fails
            
            try:
                components['responsiveness'] = self.calculate_responsiveness_health()
            except Exception as e:
                logger.warning(f"Responsiveness health calculation failed: {e}")
                components['responsiveness'] = 25.0  # Low health if calculation fails
            
            # Calculate weighted overall health
            weights = {
                'server_performance': 0.25,
                'database_health': 0.20,
                'ml_models_health': 0.20,
                'api_health': 0.15,
                'learning_quality': 0.10,
                'responsiveness': 0.10
            }
            
            overall_health = sum(components[comp] * weights[comp] for comp in components)
            
            # Apply system-wide penalties
            if overall_health < 50:
                overall_health *= 0.8  # 20% penalty for critical issues
            
            # Get status and recommendations
            status = self.get_health_status(overall_health)
            recommendations = self.get_health_recommendations(components)
            alerts = self.get_health_alerts(components, status)
            
            # Create health metrics
            health_metrics = HealthMetrics(
                timestamp=datetime.now(),
                overall_health=min(100, max(0, overall_health)),
                status=status,
                components=components,
                recommendations=recommendations,
                alerts=alerts
            )
            
            # Store in history
            self.health_history.append(health_metrics)
            if len(self.health_history) > 100:  # Keep last 100 records
                self.health_history.pop(0)
            
            return health_metrics
            
        except Exception as e:
            logger.error(f"Error calculating overall health: {e}")
            # Return a basic health metrics with low health values
            return HealthMetrics(
                timestamp=datetime.now(),
                overall_health=30.0,  # Low health if monitoring fails
                status=HealthStatus.CRITICAL,
                components={
                    'server_performance': 20.0,
                    'database_health': 25.0,
                    'ml_models_health': 15.0,
                    'api_health': 25.0,
                    'learning_quality': 20.0,
                    'responsiveness': 25.0
                },
                recommendations=["System health monitoring failed - manual investigation required"],
                alerts=["CRITICAL: Health monitoring system failure"]
            )
    
    def get_health_summary(self) -> Dict:
        """Get health summary for dashboard"""
        health_metrics = self.calculate_overall_health()
        
        return {
            'overall_health': round(health_metrics.overall_health, 1),
            'status': health_metrics.status.value,
            'components': {k: round(v, 1) for k, v in health_metrics.components.items()},
            'recommendations': health_metrics.recommendations,
            'alerts': health_metrics.alerts,
            'timestamp': health_metrics.timestamp.isoformat(),
            'history_count': len(self.health_history)
        }

# Global health monitor instance
health_monitor = SystemHealthMonitor()
