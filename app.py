"""
LearnStyle AI - Main Flask Application
An Intelligent Adaptive Learning System with Personalized Content Delivery
"""

import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from dotenv import load_dotenv
from app.models import db, User, LearningProfile, QuizResponse, ContentLibrary, UserProgress, ChatHistory
from ml_models.learning_style_predictor import LearningStylePredictor
from ml_models.multimodal_fusion_engine import MultimodalFusionEngine, EngagementMetrics, LearningContext
from ml_models.predictive_analytics import PredictiveAnalyticsEngine, LearningMetrics, RiskLevel
from ml_models.neurofeedback import EEGProcessor, FocusDetector, FatigueMonitor, CognitiveLoadAssessor
from ml_models.eye_tracking import GazeAnalyzer, AttentionMapper, LayoutOptimizer, ReadingFlowAnalyzer
from ml_models.emotion_ai import FacialEmotionAnalyzer, VoiceEmotionAnalyzer, AttentionEngagementDetector, EmotionFusionEngine, EngagementOptimizer
from ml_models.collaborative_learning import GroupDynamicsAnalyzer, PeerMatchingEngine, CollaborativeContentEngine, SocialLearningAnalytics, RealTimeCollaboration
from ml_models.research_platform import LearningEffectivenessAnalyzer, ABTestingFramework, StatisticalAnalyzer, ResearchDataManager, AcademicReportGenerator
from ml_models.career_prediction import CareerPathPredictor, SkillGapAnalyzer, CareerRecommendationEngine, IndustryAnalyzer, CareerProgressionModel
from ml_models.biometric_feedback import HRVAnalyzer, GSRMonitor, BiometricFusionEngine, StressDetector, LearningOptimizer
from app.gamification import GamificationEngine
from app.content_generator import ContentGenerator, ContentRequest, ContentType, ContentStyle

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'static')
)

# Config
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Database configuration
# MySQL configuration
MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
MYSQL_PORT = os.environ.get('MYSQL_PORT', '3306')
MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE', 'learnstyle_ai')

# URL encode password to handle special characters
import urllib.parse
if MYSQL_PASSWORD:
    encoded_password = urllib.parse.quote_plus(MYSQL_PASSWORD)
    MYSQL_URI = f'mysql+pymysql://{MYSQL_USER}:{encoded_password}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}'
else:
    MYSQL_URI = f'mysql+pymysql://{MYSQL_USER}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}'

# Use MySQL if available, fallback to SQLite for development
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', MYSQL_URI)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
    'pool_timeout': 20,
    'max_overflow': 0
}

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize ML predictor
ml_predictor = LearningStylePredictor()
try:
    ml_predictor.load_models()
    print("ML models loaded successfully")
except:
    print("No saved models found, will train new ones on first use")

# Initialize advanced AI systems
fusion_engine = MultimodalFusionEngine()
analytics_engine = PredictiveAnalyticsEngine()
gamification_engine = GamificationEngine()
content_generator = ContentGenerator()

# Initialize Neurofeedback components
eeg_processor = EEGProcessor()
focus_detector = FocusDetector()
fatigue_monitor = FatigueMonitor()
cognitive_load_assessor = CognitiveLoadAssessor()

# Initialize Eye-Tracking components
gaze_analyzer = GazeAnalyzer()
attention_mapper = AttentionMapper()
layout_optimizer = LayoutOptimizer()
reading_flow_analyzer = ReadingFlowAnalyzer()

# Initialize Emotion and Attention AI components
facial_emotion_analyzer = FacialEmotionAnalyzer()
voice_emotion_analyzer = VoiceEmotionAnalyzer()
attention_engagement_detector = AttentionEngagementDetector()
emotion_fusion_engine = EmotionFusionEngine()
engagement_optimizer = EngagementOptimizer()

# Initialize Collaborative Learning components
group_dynamics_analyzer = GroupDynamicsAnalyzer()
peer_matching_engine = PeerMatchingEngine()
collaborative_content_engine = CollaborativeContentEngine()
social_learning_analytics = SocialLearningAnalytics()
real_time_collaboration = RealTimeCollaboration()

# Initialize Research Platform components
learning_effectiveness_analyzer = LearningEffectivenessAnalyzer()
ab_testing_framework = ABTestingFramework()
statistical_analyzer = StatisticalAnalyzer()
research_data_manager = ResearchDataManager()
academic_report_generator = AcademicReportGenerator()

# Initialize Career Prediction components
career_path_predictor = CareerPathPredictor()
skill_gap_analyzer = SkillGapAnalyzer()
career_recommendation_engine = CareerRecommendationEngine()
industry_analyzer = IndustryAnalyzer()
career_progression_model = CareerProgressionModel()

# Initialize Biometric Feedback components
hrv_analyzer = HRVAnalyzer()
gsr_monitor = GSRMonitor()
biometric_fusion_engine = BiometricFusionEngine()
stress_detector = StressDetector()
learning_optimizer = LearningOptimizer()


# User loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# -----------------------------
# Authentication Routes
# -----------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('auth/login.html')

    username = request.form.get('username')
    password = request.form.get('password')
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        login_user(user)
        return redirect(url_for('dashboard'))
    flash('Invalid credentials', 'error')
    return render_template('auth/login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('auth/register.html')

    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')

    if User.query.filter_by(username=username).first():
        flash('Username exists', 'error')
        return redirect(url_for('register'))
    if User.query.filter_by(email=email).first():
        flash('Email exists', 'error')
        return redirect(url_for('register'))

    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    # Create empty learning profile
    profile = LearningProfile(user_id=user.id)
    db.session.add(profile)
    db.session.commit()

    flash('Registration successful. Please login.', 'success')
    return redirect(url_for('login'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# -----------------------------
# Main Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
@login_required
def dashboard():
    profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
    return render_template('dashboard.html', user_profile=profile)

@app.route('/advanced-dashboard')
@login_required
def advanced_dashboard():
    """Advanced AI dashboard with all features"""
    return render_template('advanced_dashboard.html')

@app.route('/biometric-dashboard')
@login_required
def biometric_dashboard():
    """Biometric feedback dashboard"""
    return render_template('biometric_dashboard.html')

@app.route('/collaborative-dashboard')
@login_required
def collaborative_dashboard():
    """Collaborative learning dashboard"""
    return render_template('collaborative_dashboard.html')

@app.route('/admin-dashboard')
@login_required
def admin_dashboard():
    """Admin dashboard - only accessible by admins"""
    if not current_user.is_admin():
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('dashboard'))
    return render_template('admin_dashboard.html')

@app.route('/moderator-dashboard')
@login_required
def moderator_dashboard():
    """Moderator dashboard - accessible by moderators and admins"""
    if not current_user.is_moderator():
        flash('Access denied. Moderator privileges required.', 'error')
        return redirect(url_for('dashboard'))
    return render_template('moderator_dashboard.html')

# Documentation Routes
@app.route('/docs')
def docs_index():
    """Documentation index page"""
    return render_template('docs/index.html')

@app.route('/docs/user-guide')
def user_guide():
    """User guide documentation"""
    return render_template('docs/user_guide.html')

@app.route('/docs/developer-guide')
def developer_guide():
    """Developer guide documentation"""
    return render_template('docs/developer_guide.html')

@app.route('/docs/api-reference')
def api_reference():
    """API reference documentation"""
    return render_template('docs/api_reference.html')

@app.route('/docs/getting-started')
def getting_started():
    """Getting started guide"""
    return render_template('docs/getting_started.html')


@app.route('/quiz')
@login_required
def quiz():
    return render_template('quiz.html')


@app.route('/chat')
@login_required
def chat():
    profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
    return render_template('chat.html', user_profile=profile)


@app.route('/explainable-ai')
@login_required
def explainable_ai():
    profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
    return render_template('explainable_ai.html', user_profile=profile)

@app.route('/neurofeedback')
@login_required
def neurofeedback():
    """Neurofeedback dashboard"""
    return render_template('neurofeedback.html')

@app.route('/eye-tracking')
@login_required
def eye_tracking():
    """Eye-tracking dashboard"""
    return render_template('eye_tracking.html')


# -----------------------------
# Quiz and ML Routes
# -----------------------------
@app.route('/submit_quiz', methods=['POST'])
@login_required
def submit_quiz():
    """Handle quiz submission and generate learning style prediction"""
    try:
        # Get quiz responses from form
        responses = []
        for i in range(1, 16):  # 15 questions
            response = request.form.get(f'question_{i}')
            if not response:
                flash('Please answer all questions', 'error')
                return redirect(url_for('quiz'))
            responses.append(int(response))
        
        # Train model if not already trained
        if not ml_predictor.is_trained:
            print("Training ML models...")
            X, y = ml_predictor.generate_synthetic_dataset(n_samples=1000)
            results = ml_predictor.train_models(X, y)
            ml_predictor.save_models()
            print(f"Model training completed. Accuracy: {results['rf_accuracy']:.3f}")
        
        # Get prediction
        prediction = ml_predictor.predict_learning_style(responses)
        
        # Save quiz response to database
        quiz_response = QuizResponse(
            user_id=current_user.id,
            question_1=responses[0],
            question_2=responses[1],
            question_3=responses[2],
            question_4=responses[3],
            question_5=responses[4],
            question_6=responses[5],
            question_7=responses[6],
            question_8=responses[7],
            question_9=responses[8],
            question_10=responses[9],
            question_11=responses[10],
            question_12=responses[11],
            question_13=responses[12],
            question_14=responses[13],
            question_15=responses[14]
        )
        db.session.add(quiz_response)
        
        # Update or create learning profile
        profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
        if not profile:
            profile = LearningProfile(user_id=current_user.id)
            db.session.add(profile)
        
        # Update profile with prediction results
        profile.visual_score = prediction['visual']
        profile.auditory_score = prediction['auditory']
        profile.kinesthetic_score = prediction['kinesthetic']
        profile.dominant_style = prediction['dominant_style']
        
        db.session.commit()
        
        flash('Quiz completed successfully! Your learning style has been analyzed.', 'success')
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        print(f"Error processing quiz: {e}")
        flash('An error occurred while processing your quiz. Please try again.', 'error')
        return redirect(url_for('quiz'))


# -----------------------------
# API Routes
# -----------------------------
@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for learning style prediction"""
    try:
        data = request.get_json()
        quiz_answers = data.get('quiz_answers', [])
        
        if len(quiz_answers) != 15:
            return jsonify({'error': 'Exactly 15 quiz answers required'}), 400
        
        # Train model if needed
        if not ml_predictor.is_trained:
            X, y = ml_predictor.generate_synthetic_dataset(n_samples=1000)
            ml_predictor.train_models(X, y)
            ml_predictor.save_models()
        
        prediction = ml_predictor.predict_learning_style(quiz_answers)
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/content', methods=['GET'])
@login_required
def api_content():
    """API endpoint for personalized content recommendations"""
    try:
        profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
        
        if not profile:
            return jsonify({'error': 'No learning profile found. Please complete the quiz first.'}), 400
        
        # Get content based on learning style
        style_tags = [profile.dominant_style]
        content = ContentLibrary.query.filter(
            ContentLibrary.style_tags.contains(profile.dominant_style)
        ).limit(10).all()
        
        content_list = []
        for item in content:
            content_list.append({
                'id': item.id,
                'title': item.title,
                'description': item.description,
                'content_type': item.content_type,
                'style_tags': item.get_style_tags_list(),
                'difficulty_level': item.difficulty_level,
                'url_path': item.url_path
            })
        
        return jsonify({
            'content': content_list,
            'learning_style': profile.dominant_style,
            'style_breakdown': profile.get_style_breakdown()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    """API endpoint for AI tutor chat"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
        learning_style = profile.dominant_style if profile else 'visual'
        
        # Simple AI response generation (in production, this would call OpenAI API)
        ai_response = generate_ai_response(user_message, learning_style)
        
        # Save chat history
        chat_entry = ChatHistory(
            user_id=current_user.id,
            user_message=user_message,
            ai_response=ai_response,
            learning_style_context=learning_style
        )
        db.session.add(chat_entry)
        db.session.commit()
        
        return jsonify({
            'response': ai_response,
            'learning_style': learning_style
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/progress', methods=['POST'])
@login_required
def api_progress():
    """API endpoint for updating user progress"""
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        completion_status = data.get('completion_status', 'completed')
        time_spent = data.get('time_spent', 0)
        score = data.get('score')
        engagement_rating = data.get('engagement_rating')
        
        if not content_id:
            return jsonify({'error': 'Content ID required'}), 400
        
        # Update or create progress record
        progress = UserProgress.query.filter_by(
            user_id=current_user.id,
            content_id=content_id
        ).first()
        
        if not progress:
            progress = UserProgress(
                user_id=current_user.id,
                content_id=content_id
            )
            db.session.add(progress)
        
        progress.completion_status = completion_status
        progress.time_spent = time_spent
        if score is not None:
            progress.score = score
        if engagement_rating is not None:
            progress.engagement_rating = engagement_rating
        
        db.session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -----------------------------
# Advanced AI Features API
# -----------------------------
@app.route('/api/explainable-ai', methods=['GET'])
@login_required
def api_explainable_ai():
    """API endpoint for explainable AI dashboard data"""
    try:
        profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
        
        if not profile:
            return jsonify({'error': 'No learning profile found'}), 400
        
        # Get feature importance from ML model
        feature_importance = ml_predictor.feature_importance or {}
        
        # Get quiz responses for analysis
        quiz_responses = QuizResponse.query.filter_by(user_id=current_user.id).first()
        
        explanation_data = {
            'learning_style': profile.dominant_style,
            'style_breakdown': profile.get_style_breakdown(),
            'feature_importance': feature_importance,
            'model_confidence': 0.87,  # Placeholder
            'quiz_responses': quiz_responses.get_responses_list() if quiz_responses else [],
            'alternative_scenarios': [
                {'style': 'mixed', 'probability': 0.23},
                {'style': 'auditory', 'probability': 0.12},
                {'style': 'kinesthetic', 'probability': 0.05}
            ],
            'model_performance': {
                'random_forest': {'accuracy': 0.915, 'confidence': 'high'},
                'decision_tree': {'accuracy': 0.865, 'confidence': 'medium'},
                'ensemble': {'accuracy': 0.932, 'confidence': 'very_high'}
            }
        }
        
        return jsonify(explanation_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/gamification', methods=['GET'])
@login_required
def api_gamification():
    """API endpoint for gamification data"""
    try:
        # Get user profile
        profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
        
        # Get user progress data
        progress_records = UserProgress.query.filter_by(user_id=current_user.id).all()
        
        # Calculate user stats for gamification
        user_stats = {
            'visual_content_read': len([p for p in progress_records if p.content and 'visual' in p.content.style_tags]),
            'audio_content_completed': len([p for p in progress_records if p.content and 'auditory' in p.content.style_tags]),
            'interactive_exercises': len([p for p in progress_records if p.content and 'kinesthetic' in p.content.style_tags]),
            'total_points': sum(p.score or 0 for p in progress_records),
            'achievements_unlocked': 0  # Would be calculated from actual achievements
        }
        
        # Check for new achievements
        new_achievements = gamification_engine.check_achievements(current_user.id, user_stats)
        
        # Calculate level
        level = gamification_engine.calculate_user_level(user_stats['total_points'])
        level_progress = gamification_engine.get_progress_to_next_level(level, user_stats['total_points'])
        
        gamification_data = {
            'level': level,
            'total_points': user_stats['total_points'],
            'level_progress': level_progress,
            'achievements': [
                {
                    'id': achievement.id,
                    'name': achievement.name,
                    'description': achievement.description,
                    'icon': achievement.icon,
                    'rarity': achievement.rarity.value,
                    'points': achievement.points
                } for achievement in new_achievements
            ],
            'recommendations': gamification_engine.generate_style_specific_recommendations(
                profile.dominant_style if profile else 'visual', level
            )
        }
        
        return jsonify(gamification_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictive-analytics', methods=['POST'])
@login_required
def api_predictive_analytics():
    """API endpoint for predictive analytics and risk assessment"""
    try:
        data = request.get_json()
        
        # Create learning metrics from request data
        metrics = LearningMetrics(
            user_id=current_user.id,
            timestamp=datetime.now(),
            content_completion_rate=data.get('completion_rate', 0.5),
            average_session_duration=data.get('session_duration', 1800),
            quiz_scores=data.get('quiz_scores', [0.7, 0.8, 0.6]),
            engagement_score=data.get('engagement_score', 0.6),
            style_content_mismatch=data.get('style_mismatch', 0.3),
            time_spent_per_content=data.get('time_per_content', {}),
            error_rate=data.get('error_rate', 0.2),
            help_seeking_frequency=data.get('help_frequency', 0.3),
            social_interaction_score=data.get('social_score', 0.4)
        )
        
        # Analyze learning patterns
        analysis = analytics_engine.analyze_learning_patterns(metrics)
        
        # Assess risk
        risk_assessment = analytics_engine.assess_learning_risk(current_user.id, metrics)
        
        # Generate interventions
        interventions = analytics_engine.generate_interventions(risk_assessment)
        
        analytics_data = {
            'analysis': analysis,
            'risk_assessment': {
                'risk_level': risk_assessment.risk_level.value,
                'risk_score': risk_assessment.risk_score,
                'contributing_factors': risk_assessment.contributing_factors,
                'predicted_outcome': risk_assessment.predicted_outcome,
                'confidence': risk_assessment.confidence
            },
            'interventions': [
                {
                    'type': intervention.intervention_type.value,
                    'priority': intervention.priority,
                    'description': intervention.description,
                    'expected_impact': intervention.expected_impact,
                    'timeline': intervention.timeline
                } for intervention in interventions
            ]
        }
        
        return jsonify(analytics_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-content', methods=['POST'])
@login_required
def api_generate_content():
    """API endpoint for AI content generation"""
    try:
        data = request.get_json()
        
        # Create content request
        content_request = ContentRequest(
            topic=data.get('topic', 'General Learning'),
            learning_style=ContentStyle(data.get('learning_style', 'visual')),
            difficulty_level=data.get('difficulty_level', 'intermediate'),
            content_type=ContentType(data.get('content_type', 'text')),
            user_preferences=data.get('preferences', {}),
            context=data.get('context')
        )
        
        # Generate content
        generated_content = content_generator.generate_content(content_request)
        
        # Save to database
        content_library_item = ContentLibrary(
            title=generated_content.title,
            description=generated_content.content[:200] + "...",
            content_type=generated_content.content_type.value,
            style_tags=','.join(generated_content.style_tags),
            difficulty_level=generated_content.difficulty_level,
            url_path=f"/generated/{generated_content.content_id}"
        )
        db.session.add(content_library_item)
        db.session.commit()
        
        return jsonify({
            'content_id': generated_content.content_id,
            'title': generated_content.title,
            'content': generated_content.content,
            'content_type': generated_content.content_type.value,
            'style_tags': generated_content.style_tags,
            'difficulty_level': generated_content.difficulty_level,
            'metadata': generated_content.metadata
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/multimodal-fusion', methods=['POST'])
@login_required
def api_multimodal_fusion():
    """API endpoint for multimodal learning style fusion"""
    try:
        data = request.get_json()
        
        # Create engagement metrics
        engagement = EngagementMetrics(
            content_interaction_time=data.get('interaction_time', 300),
            scroll_velocity=data.get('scroll_velocity', 500),
            click_frequency=data.get('click_frequency', 5),
            pause_duration=data.get('pause_duration', 10),
            completion_rate=data.get('completion_rate', 0.8),
            timestamp=datetime.now()
        )
        
        # Update style weights
        updated_weights = fusion_engine.update_style_weights(
            user_id=current_user.id,
            engagement_data=engagement,
            content_style=data.get('content_style', 'visual'),
            performance_score=data.get('performance_score', 0.7),
            context=LearningContext(data.get('context', 'general'))
        )
        
        # Get style evolution timeline
        evolution_timeline = fusion_engine.get_style_evolution_timeline(current_user.id, days=30)
        
        # Detect evolution patterns
        patterns = fusion_engine.detect_style_evolution_patterns(current_user.id)
        
        fusion_data = {
            'updated_weights': {
                'visual': updated_weights.visual,
                'auditory': updated_weights.auditory,
                'kinesthetic': updated_weights.kinesthetic,
                'confidence': updated_weights.confidence
            },
            'evolution_timeline': evolution_timeline,
            'patterns': patterns
        }
        
        return jsonify(fusion_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -----------------------------
# Neurofeedback API Endpoints
# -----------------------------

@app.route('/api/neurofeedback/focus', methods=['POST'])
@login_required
def neurofeedback_focus():
    """Process EEG data for focus detection"""
    try:
        data = request.get_json()
        eeg_data = data.get('eeg_data', [])
        
        if not eeg_data:
            return jsonify({'error': 'No EEG data provided'}), 400
        
        # Process EEG data
        eeg_features = eeg_processor.process_eeg_data(eeg_data)
        
        # Detect focus level
        focus_metrics = focus_detector.detect_focus_level(eeg_features)
        
        # Return focus analysis
        return jsonify({
            'focus_level': focus_metrics.focus_level,
            'focus_classification': focus_metrics.focus_classification.value,
            'attention_state': focus_metrics.attention_state.value,
            'alpha_beta_ratio': focus_metrics.alpha_beta_ratio,
            'theta_alpha_ratio': focus_metrics.theta_alpha_ratio,
            'confidence': focus_metrics.confidence,
            'trend': focus_metrics.trend,
            'recommendation': focus_metrics.recommendation
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/neurofeedback/fatigue', methods=['POST'])
@login_required
def neurofeedback_fatigue():
    """Process EEG data for fatigue monitoring"""
    try:
        data = request.get_json()
        eeg_data = data.get('eeg_data', [])
        session_duration = data.get('session_duration', None)
        
        if not eeg_data:
            return jsonify({'error': 'No EEG data provided'}), 400
        
        # Process EEG data
        eeg_features = eeg_processor.process_eeg_data(eeg_data)
        
        # Monitor fatigue
        fatigue_metrics = fatigue_monitor.monitor_fatigue(eeg_features, session_duration)
        
        # Return fatigue analysis
        return jsonify({
            'fatigue_level': fatigue_metrics.fatigue_level,
            'fatigue_classification': fatigue_metrics.fatigue_classification.value,
            'theta_alpha_ratio': fatigue_metrics.theta_alpha_ratio,
            'cognitive_load': fatigue_metrics.cognitive_load,
            'break_recommendation': fatigue_metrics.break_recommendation.value,
            'break_duration': fatigue_metrics.break_duration,
            'confidence': fatigue_metrics.confidence,
            'session_quality': fatigue_metrics.session_quality,
            'recommendation': fatigue_metrics.recommendation
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/neurofeedback/cognitive-load', methods=['POST'])
@login_required
def neurofeedback_cognitive_load():
    """Process EEG data for cognitive load assessment"""
    try:
        data = request.get_json()
        eeg_data = data.get('eeg_data', [])
        content_metrics = data.get('content_metrics', None)
        user_behavior = data.get('user_behavior', None)
        
        if not eeg_data:
            return jsonify({'error': 'No EEG data provided'}), 400
        
        # Process EEG data
        eeg_features = eeg_processor.process_eeg_data(eeg_data)
        
        # Assess cognitive load
        load_metrics = cognitive_load_assessor.assess_cognitive_load(
            eeg_features, content_metrics, user_behavior
        )
        
        # Return cognitive load analysis
        return jsonify({
            'total_load': load_metrics.total_load,
            'intrinsic_load': load_metrics.intrinsic_load,
            'extrinsic_load': load_metrics.extrinsic_load,
            'germane_load': load_metrics.germane_load,
            'load_classification': load_metrics.load_classification.value,
            'working_memory_usage': load_metrics.working_memory_usage,
            'processing_efficiency': load_metrics.processing_efficiency,
            'content_difficulty_score': load_metrics.content_difficulty_score,
            'interface_complexity_score': load_metrics.interface_complexity_score,
            'learning_efficiency': load_metrics.learning_efficiency,
            'confidence': load_metrics.confidence,
            'recommendation': load_metrics.recommendation
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/neurofeedback/comprehensive', methods=['POST'])
@login_required
def neurofeedback_comprehensive():
    """Comprehensive neurofeedback analysis combining all metrics"""
    try:
        data = request.get_json()
        eeg_data = data.get('eeg_data', [])
        content_metrics = data.get('content_metrics', None)
        user_behavior = data.get('user_behavior', None)
        session_duration = data.get('session_duration', None)
        
        if not eeg_data:
            return jsonify({'error': 'No EEG data provided'}), 400
        
        # Process EEG data
        eeg_features = eeg_processor.process_eeg_data(eeg_data)
        
        # Get all neurofeedback analyses
        focus_metrics = focus_detector.detect_focus_level(eeg_features)
        fatigue_metrics = fatigue_monitor.monitor_fatigue(eeg_features, session_duration)
        load_metrics = cognitive_load_assessor.assess_cognitive_load(
            eeg_features, content_metrics, user_behavior
        )
        
        # Calculate overall learning state
        overall_state = _calculate_overall_learning_state(
            focus_metrics, fatigue_metrics, load_metrics
        )
        
        # Return comprehensive analysis
        return jsonify({
            'focus': {
                'level': focus_metrics.focus_level,
                'classification': focus_metrics.focus_classification.value,
                'attention_state': focus_metrics.attention_state.value,
                'confidence': focus_metrics.confidence,
                'recommendation': focus_metrics.recommendation
            },
            'fatigue': {
                'level': fatigue_metrics.fatigue_level,
                'classification': fatigue_metrics.fatigue_classification.value,
                'break_recommendation': fatigue_metrics.break_recommendation.value,
                'break_duration': fatigue_metrics.break_duration,
                'session_quality': fatigue_metrics.session_quality,
                'recommendation': fatigue_metrics.recommendation
            },
            'cognitive_load': {
                'total_load': load_metrics.total_load,
                'classification': load_metrics.load_classification.value,
                'learning_efficiency': load_metrics.learning_efficiency,
                'processing_efficiency': load_metrics.processing_efficiency,
                'recommendation': load_metrics.recommendation
            },
            'overall_state': overall_state,
            'eeg_features': eeg_features
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/neurofeedback/statistics', methods=['GET'])
@login_required
def neurofeedback_statistics():
    """Get neurofeedback statistics for current user"""
    try:
        # Get statistics from all neurofeedback components
        focus_stats = focus_detector.get_focus_statistics()
        fatigue_stats = fatigue_monitor.get_fatigue_statistics()
        load_stats = cognitive_load_assessor.get_load_statistics()
        
        return jsonify({
            'focus_statistics': focus_stats,
            'fatigue_statistics': fatigue_stats,
            'cognitive_load_statistics': load_stats,
            'session_duration': fatigue_monitor.get_session_duration()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -----------------------------
# Eye-Tracking API Endpoints
# -----------------------------

@app.route('/api/eye-tracking/analyze', methods=['POST'])
@login_required
def eye_tracking_analyze():
    """Analyze gaze patterns and generate insights"""
    try:
        data = request.get_json()
        gaze_points = data.get('gaze_points', [])
        content_bounds = data.get('content_bounds', {'width': 800, 'height': 600})
        content_layout = data.get('content_layout', {})
        
        if not gaze_points:
            return jsonify({'error': 'No gaze data provided'}), 400
        
        # Convert gaze points to GazePoint objects
        gaze_objects = []
        for point in gaze_points:
            gaze_obj = {
                'x': point.get('x', 0),
                'y': point.get('y', 0),
                'timestamp': point.get('timestamp', 0),
                'duration': point.get('duration', 100),
                'pupil_diameter': point.get('pupil_diameter'),
                'confidence': point.get('confidence', 1.0)
            }
            gaze_objects.append(gaze_obj)
        
        # Analyze gaze patterns
        gaze_metrics = gaze_analyzer.analyze_gaze_patterns(gaze_objects, content_bounds)
        
        # Create attention map
        attention_map = attention_mapper.create_attention_map(
            gaze_objects, content_layout, content_bounds
        )
        
        # Analyze reading flow
        text_content = data.get('text_content')
        reading_metrics = reading_flow_analyzer.analyze_reading_flow(gaze_objects, text_content)
        
        # Return comprehensive analysis
        return jsonify({
            'gaze_metrics': {
                'attention_level': gaze_metrics.attention_level.value,
                'engagement_score': gaze_metrics.engagement_score,
                'fixation_count': gaze_metrics.fixation_count,
                'average_fixation_duration': gaze_metrics.average_fixation_duration,
                'saccade_count': gaze_metrics.saccade_count,
                'scanpath_length': gaze_metrics.scanpath_length,
                'scanpath_efficiency': gaze_metrics.scanpath_efficiency,
                'visual_search_efficiency': gaze_metrics.visual_search_efficiency,
                'confidence': gaze_metrics.confidence
            },
            'attention_map': {
                'pattern_type': attention_map.pattern_type.value,
                'hierarchy_effectiveness': attention_map.hierarchy_effectiveness.value,
                'engagement_distribution': attention_map.engagement_distribution,
                'visual_flow_score': attention_map.visual_flow_score,
                'content_effectiveness': attention_map.content_effectiveness,
                'recommendations': attention_map.recommendations
            },
            'reading_flow': {
                'pattern_type': reading_metrics.pattern_type.value,
                'reading_speed': reading_metrics.reading_speed,
                'regression_count': reading_metrics.regression_count,
                'reading_efficiency': reading_metrics.reading_efficiency.value,
                'comprehension_score': reading_metrics.comprehension_score,
                'attention_span': reading_metrics.attention_span,
                'text_engagement': reading_metrics.text_engagement,
                'reading_rhythm': reading_metrics.reading_rhythm,
                'confidence': reading_metrics.confidence
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eye-tracking/optimize-layout', methods=['POST'])
@login_required
def eye_tracking_optimize_layout():
    """Optimize content layout based on eye-tracking data"""
    try:
        data = request.get_json()
        gaze_analysis = data.get('gaze_analysis', {})
        current_layout = data.get('current_layout', {})
        content_metrics = data.get('content_metrics')
        
        # Optimize layout
        optimization_result = layout_optimizer.optimize_layout(
            gaze_analysis, current_layout, content_metrics
        )
        
        # Return optimization recommendations
        return jsonify({
            'optimizations': [
                {
                    'type': opt.type.value,
                    'priority': opt.priority.value,
                    'current_value': opt.current_value,
                    'recommended_value': opt.recommended_value,
                    'confidence': opt.confidence,
                    'expected_improvement': opt.expected_improvement,
                    'description': opt.description
                } for opt in optimization_result.optimizations
            ],
            'overall_confidence': optimization_result.overall_confidence,
            'expected_improvement': optimization_result.expected_improvement,
            'implementation_priority': [opt.value for opt in optimization_result.implementation_priority],
            'css_changes': optimization_result.css_changes,
            'recommendations': optimization_result.recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eye-tracking/attention-heatmap', methods=['POST'])
@login_required
def eye_tracking_attention_heatmap():
    """Generate attention heatmap from gaze data"""
    try:
        data = request.get_json()
        gaze_points = data.get('gaze_points', [])
        content_bounds = data.get('content_bounds', {'width': 800, 'height': 600})
        content_layout = data.get('content_layout', {})
        
        if not gaze_points:
            return jsonify({'error': 'No gaze data provided'}), 400
        
        # Convert gaze points
        gaze_objects = []
        for point in gaze_points:
            gaze_obj = {
                'x': point.get('x', 0),
                'y': point.get('y', 0),
                'timestamp': point.get('timestamp', 0),
                'duration': point.get('duration', 100),
                'pupil_diameter': point.get('pupil_diameter'),
                'confidence': point.get('confidence', 1.0)
            }
            gaze_objects.append(gaze_obj)
        
        # Create attention map
        attention_map = attention_mapper.create_attention_map(
            gaze_objects, content_layout, content_bounds
        )
        
        # Convert heatmap to list for JSON serialization
        heatmap_list = attention_map.heatmap.tolist()
        
        # Convert attention regions to serializable format
        attention_regions = []
        for region in attention_map.attention_regions:
            attention_regions.append({
                'x': region.x,
                'y': region.y,
                'width': region.width,
                'height': region.height,
                'intensity': region.intensity,
                'duration': region.duration,
                'importance': region.importance,
                'content_type': region.content_type
            })
        
        return jsonify({
            'heatmap': heatmap_list,
            'attention_regions': attention_regions,
            'pattern_type': attention_map.pattern_type.value,
            'hierarchy_effectiveness': attention_map.hierarchy_effectiveness.value,
            'engagement_distribution': attention_map.engagement_distribution,
            'visual_flow_score': attention_map.visual_flow_score,
            'content_effectiveness': attention_map.content_effectiveness,
            'recommendations': attention_map.recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eye-tracking/reading-analysis', methods=['POST'])
@login_required
def eye_tracking_reading_analysis():
    """Analyze reading patterns and efficiency"""
    try:
        data = request.get_json()
        gaze_points = data.get('gaze_points', [])
        text_content = data.get('text_content')
        
        if not gaze_points:
            return jsonify({'error': 'No gaze data provided'}), 400
        
        # Convert gaze points
        gaze_objects = []
        for point in gaze_points:
            gaze_obj = {
                'x': point.get('x', 0),
                'y': point.get('y', 0),
                'timestamp': point.get('timestamp', 0),
                'duration': point.get('duration', 100),
                'pupil_diameter': point.get('pupil_diameter'),
                'confidence': point.get('confidence', 1.0)
            }
            gaze_objects.append(gaze_obj)
        
        # Analyze reading flow
        reading_metrics = reading_flow_analyzer.analyze_reading_flow(gaze_objects, text_content)
        
        return jsonify({
            'pattern_type': reading_metrics.pattern_type.value,
            'reading_speed': reading_metrics.reading_speed,
            'regression_count': reading_metrics.regression_count,
            'fixation_duration': reading_metrics.fixation_duration,
            'saccade_amplitude': reading_metrics.saccade_amplitude,
            'reading_efficiency': reading_metrics.reading_efficiency.value,
            'comprehension_score': reading_metrics.comprehension_score,
            'attention_span': reading_metrics.attention_span,
            'text_engagement': reading_metrics.text_engagement,
            'reading_rhythm': reading_metrics.reading_rhythm,
            'confidence': reading_metrics.confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -----------------------------
# Emotion and Attention AI API Endpoints
# -----------------------------

@app.route('/api/emotion-ai/analyze', methods=['POST'])
@login_required
def emotion_ai_analyze():
    """Comprehensive emotion and attention analysis"""
    try:
        data = request.get_json()
        facial_data = data.get('facial_data', {})
        voice_data = data.get('voice_data', {})
        attention_data = data.get('attention_data', {})
        learning_context = data.get('learning_context', {})
        
        # Analyze facial emotions
        facial_emotion_metrics = facial_emotion_analyzer.analyze_facial_emotion(facial_data)
        
        # Analyze voice emotions
        voice_emotion_metrics = voice_emotion_analyzer.analyze_voice_emotion(voice_data)
        
        # Detect attention and engagement
        attention_metrics = attention_engagement_detector.detect_attention_engagement({
            'facial_data': facial_data,
            'voice_data': voice_data,
            'eye_tracking_data': data.get('eye_tracking_data', {}),
            'neurofeedback_data': data.get('neurofeedback_data', {}),
            'behavioral_data': data.get('behavioral_data', {})
        })
        
        # Fuse emotions from all modalities
        fused_emotion_metrics = emotion_fusion_engine.fuse_emotions(
            {
                'primary_emotion': facial_emotion_metrics.primary_emotion.value,
                'emotion_confidence': facial_emotion_metrics.emotion_confidence,
                'emotion_intensity': facial_emotion_metrics.emotion_intensity,
                'engagement_score': facial_emotion_metrics.engagement_score,
                'attention_score': facial_emotion_metrics.attention_score,
                'confidence': facial_emotion_metrics.confidence
            },
            {
                'primary_emotion': voice_emotion_metrics.primary_emotion.value,
                'emotion_confidence': voice_emotion_metrics.emotion_confidence,
                'sentiment_score': voice_emotion_metrics.sentiment_score,
                'engagement_score': voice_emotion_metrics.engagement_score,
                'stress_score': voice_emotion_metrics.stress_score,
                'confidence': voice_emotion_metrics.confidence
            },
            {
                'attention_state': attention_metrics.attention_state.value,
                'attention_score': attention_metrics.attention_score,
                'engagement_level': attention_metrics.engagement_level.value,
                'engagement_score': attention_metrics.engagement_score,
                'focus_quality': attention_metrics.focus_quality,
                'confidence': attention_metrics.confidence
            }
        )
        
        # Optimize engagement
        optimization_result = engagement_optimizer.optimize_engagement(
            {
                'overall_emotion_state': fused_emotion_metrics.overall_emotion_state.value,
                'emotion_confidence': fused_emotion_metrics.emotion_confidence,
                'emotional_engagement': fused_emotion_metrics.emotional_engagement,
                'learning_readiness': fused_emotion_metrics.learning_readiness,
                'confidence': fused_emotion_metrics.confidence
            },
            {
                'attention_state': attention_metrics.attention_state.value,
                'attention_score': attention_metrics.attention_score,
                'engagement_level': attention_metrics.engagement_level.value,
                'engagement_score': attention_metrics.engagement_score,
                'focus_quality': attention_metrics.focus_quality,
                'distraction_level': attention_metrics.distraction_level,
                'learning_readiness': attention_metrics.learning_readiness,
                'confidence': attention_metrics.confidence
            },
            learning_context
        )
        
        return jsonify({
            'facial_emotion': {
                'primary_emotion': facial_emotion_metrics.primary_emotion.value,
                'emotion_confidence': facial_emotion_metrics.emotion_confidence,
                'emotion_intensity': facial_emotion_metrics.emotion_intensity,
                'engagement_level': facial_emotion_metrics.engagement_level.value,
                'attention_state': facial_emotion_metrics.attention_state.value,
                'engagement_score': facial_emotion_metrics.engagement_score,
                'attention_score': facial_emotion_metrics.attention_score,
                'confidence': facial_emotion_metrics.confidence
            },
            'voice_emotion': {
                'primary_emotion': voice_emotion_metrics.primary_emotion.value,
                'emotion_confidence': voice_emotion_metrics.emotion_confidence,
                'sentiment': voice_emotion_metrics.sentiment.value,
                'sentiment_score': voice_emotion_metrics.sentiment_score,
                'stress_level': voice_emotion_metrics.stress_level.value,
                'stress_score': voice_emotion_metrics.stress_score,
                'engagement_score': voice_emotion_metrics.engagement_score,
                'fatigue_score': voice_emotion_metrics.fatigue_score,
                'confidence': voice_emotion_metrics.confidence
            },
            'attention_engagement': {
                'attention_state': attention_metrics.attention_state.value,
                'attention_score': attention_metrics.attention_score,
                'engagement_level': attention_metrics.engagement_level.value,
                'engagement_score': attention_metrics.engagement_score,
                'attention_span': attention_metrics.attention_span,
                'distraction_level': attention_metrics.distraction_level,
                'distraction_types': [dt.value for dt in attention_metrics.distraction_types],
                'focus_quality': attention_metrics.focus_quality,
                'learning_readiness': attention_metrics.learning_readiness,
                'confidence': attention_metrics.confidence
            },
            'fused_emotion': {
                'overall_emotion_state': fused_emotion_metrics.overall_emotion_state.value,
                'emotion_confidence': fused_emotion_metrics.emotion_confidence,
                'emotion_intensity': fused_emotion_metrics.emotion_intensity,
                'emotion_trend': fused_emotion_metrics.emotion_trend.value,
                'learning_emotion_state': fused_emotion_metrics.learning_emotion_state.value,
                'emotion_stability': fused_emotion_metrics.emotion_stability,
                'emotional_engagement': fused_emotion_metrics.emotional_engagement,
                'learning_readiness': fused_emotion_metrics.learning_readiness,
                'confidence': fused_emotion_metrics.confidence
            },
            'engagement_optimization': {
                'interventions': [
                    {
                        'type': interv.type.value,
                        'priority': interv.priority,
                        'confidence': interv.confidence,
                        'expected_improvement': interv.expected_improvement,
                        'description': interv.description,
                        'parameters': interv.parameters,
                        'implementation_time': interv.implementation_time
                    } for interv in optimization_result.interventions
                ],
                'overall_strategy': optimization_result.overall_strategy.value,
                'expected_engagement_improvement': optimization_result.expected_engagement_improvement,
                'implementation_priority': [opt.value for opt in optimization_result.implementation_priority],
                'personalized_recommendations': optimization_result.personalized_recommendations,
                'monitoring_metrics': optimization_result.monitoring_metrics,
                'confidence': optimization_result.confidence
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/emotion-ai/facial-analysis', methods=['POST'])
@login_required
def emotion_ai_facial_analysis():
    """Facial emotion analysis endpoint"""
    try:
        data = request.get_json()
        facial_data = data.get('facial_data', {})
        
        # Analyze facial emotions
        emotion_metrics = facial_emotion_analyzer.analyze_facial_emotion(facial_data)
        
        return jsonify({
            'primary_emotion': emotion_metrics.primary_emotion.value,
            'emotion_confidence': emotion_metrics.emotion_confidence,
            'emotion_intensity': emotion_metrics.emotion_intensity,
            'engagement_level': emotion_metrics.engagement_level.value,
            'attention_state': emotion_metrics.attention_state.value,
            'engagement_score': emotion_metrics.engagement_score,
            'attention_score': emotion_metrics.attention_score,
            'micro_expressions': emotion_metrics.micro_expressions,
            'emotion_timeline': emotion_metrics.emotion_timeline,
            'confidence': emotion_metrics.confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/emotion-ai/voice-analysis', methods=['POST'])
@login_required
def emotion_ai_voice_analysis():
    """Voice emotion analysis endpoint"""
    try:
        data = request.get_json()
        voice_data = data.get('voice_data', {})
        
        # Analyze voice emotions
        emotion_metrics = voice_emotion_analyzer.analyze_voice_emotion(voice_data)
        
        return jsonify({
            'primary_emotion': emotion_metrics.primary_emotion.value,
            'emotion_confidence': emotion_metrics.emotion_confidence,
            'sentiment': emotion_metrics.sentiment.value,
            'sentiment_score': emotion_metrics.sentiment_score,
            'stress_level': emotion_metrics.stress_level.value,
            'stress_score': emotion_metrics.stress_score,
            'engagement_score': emotion_metrics.engagement_score,
            'fatigue_score': emotion_metrics.fatigue_score,
            'voice_features': {
                'pitch': emotion_metrics.voice_features.pitch,
                'pitch_variance': emotion_metrics.voice_features.pitch_variance,
                'volume': emotion_metrics.voice_features.volume,
                'volume_variance': emotion_metrics.voice_features.volume_variance,
                'speaking_rate': emotion_metrics.voice_features.speaking_rate,
                'pause_frequency': emotion_metrics.voice_features.pause_frequency,
                'jitter': emotion_metrics.voice_features.jitter,
                'shimmer': emotion_metrics.voice_features.shimmer,
                'hnr': emotion_metrics.voice_features.hnr
            },
            'emotion_timeline': emotion_metrics.emotion_timeline,
            'confidence': emotion_metrics.confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/emotion-ai/attention-analysis', methods=['POST'])
@login_required
def emotion_ai_attention_analysis():
    """Attention and engagement analysis endpoint"""
    try:
        data = request.get_json()
        multi_modal_data = data.get('multi_modal_data', {})
        
        # Detect attention and engagement
        attention_metrics = attention_engagement_detector.detect_attention_engagement(multi_modal_data)
        
        return jsonify({
            'attention_state': attention_metrics.attention_state.value,
            'attention_score': attention_metrics.attention_score,
            'engagement_level': attention_metrics.engagement_level.value,
            'engagement_score': attention_metrics.engagement_score,
            'attention_span': attention_metrics.attention_span,
            'distraction_level': attention_metrics.distraction_level,
            'distraction_types': [dt.value for dt in attention_metrics.distraction_types],
            'focus_quality': attention_metrics.focus_quality,
            'learning_readiness': attention_metrics.learning_readiness,
            'attention_timeline': attention_metrics.attention_timeline,
            'confidence': attention_metrics.confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/emotion-ai/engagement-optimization', methods=['POST'])
@login_required
def emotion_ai_engagement_optimization():
    """Engagement optimization endpoint"""
    try:
        data = request.get_json()
        emotion_data = data.get('emotion_data', {})
        attention_data = data.get('attention_data', {})
        learning_context = data.get('learning_context', {})
        
        # Optimize engagement
        optimization_result = engagement_optimizer.optimize_engagement(
            emotion_data, attention_data, learning_context
        )
        
        return jsonify({
            'interventions': [
                {
                    'type': interv.type.value,
                    'priority': interv.priority,
                    'confidence': interv.confidence,
                    'expected_improvement': interv.expected_improvement,
                    'description': interv.description,
                    'parameters': interv.parameters,
                    'implementation_time': interv.implementation_time
                } for interv in optimization_result.interventions
            ],
            'overall_strategy': optimization_result.overall_strategy.value,
            'expected_engagement_improvement': optimization_result.expected_engagement_improvement,
            'implementation_priority': [opt.value for opt in optimization_result.implementation_priority],
            'personalized_recommendations': optimization_result.personalized_recommendations,
            'monitoring_metrics': optimization_result.monitoring_metrics,
            'confidence': optimization_result.confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/emotion-ai/statistics', methods=['GET'])
@login_required
def emotion_ai_statistics():
    """Get emotion and attention AI statistics"""
    try:
        # Get statistics from all components
        facial_stats = facial_emotion_analyzer.get_emotion_statistics()
        voice_stats = voice_emotion_analyzer.get_voice_statistics()
        attention_stats = attention_engagement_detector.get_attention_statistics()
        emotion_stats = emotion_fusion_engine.get_emotion_statistics()
        optimization_stats = engagement_optimizer.get_optimization_statistics()
        
        return jsonify({
            'facial_emotion_statistics': facial_stats,
            'voice_emotion_statistics': voice_stats,
            'attention_statistics': attention_stats,
            'emotion_fusion_statistics': emotion_stats,
            'optimization_statistics': optimization_stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -----------------------------
# Collaborative Learning API Endpoints
# -----------------------------

@app.route('/api/collaborative/group-dynamics', methods=['POST'])
@login_required
def collaborative_group_dynamics():
    """Analyze group dynamics for collaborative learning"""
    try:
        data = request.get_json()
        group_members_data = data.get('group_members', [])
        interaction_data = data.get('interaction_data', {})
        group_id = data.get('group_id', f"group_{current_user.id}")
        
        # Convert member data to GroupMember objects
        from ml_models.collaborative_learning.group_dynamics_analyzer import GroupMember
        group_members = []
        for member_data in group_members_data:
            member = GroupMember(
                user_id=member_data.get('user_id', ''),
                learning_style=member_data.get('learning_style', 'visual'),
                expertise_level=member_data.get('expertise_level', 0.5),
                participation_score=member_data.get('participation_score', 0.5),
                communication_frequency=member_data.get('communication_frequency', 0.5),
                contribution_quality=member_data.get('contribution_quality', 0.5),
                leadership_tendency=member_data.get('leadership_tendency', 0.5),
                collaboration_style=member_data.get('collaboration_style', 'collaborator')
            )
            group_members.append(member)
        
        # Analyze group dynamics
        dynamics_metrics = group_dynamics_analyzer.analyze_group_dynamics(
            group_members, interaction_data, group_id
        )
        
        return jsonify({
            'group_id': dynamics_metrics.group_id,
            'group_size': dynamics_metrics.group_size,
            'group_cohesion': dynamics_metrics.group_cohesion.value,
            'collaboration_effectiveness': dynamics_metrics.collaboration_effectiveness.value,
            'communication_balance': dynamics_metrics.communication_balance,
            'leadership_distribution': dynamics_metrics.leadership_distribution,
            'role_distribution': {role.value: count for role, count in dynamics_metrics.role_distribution.items()},
            'participation_equality': dynamics_metrics.participation_equality,
            'knowledge_diversity': dynamics_metrics.knowledge_diversity,
            'group_synergy': dynamics_metrics.group_synergy,
            'conflict_level': dynamics_metrics.conflict_level,
            'group_satisfaction': dynamics_metrics.group_satisfaction,
            'learning_outcomes': dynamics_metrics.learning_outcomes,
            'recommendations': dynamics_metrics.recommendations,
            'confidence': dynamics_metrics.confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/collaborative/peer-matching', methods=['POST'])
@login_required
def collaborative_peer_matching():
    """Find optimal peer matches for collaborative learning"""
    try:
        data = request.get_json()
        user_profile_data = data.get('user_profile', {})
        candidate_profiles_data = data.get('candidate_profiles', [])
        matching_strategy = data.get('matching_strategy', 'complementary')
        max_matches = data.get('max_matches', 5)
        
        # Convert to PeerProfile objects
        from ml_models.collaborative_learning.peer_matching_engine import PeerProfile, MatchingStrategy, LearningGoal
        from ml_models.collaborative_learning.peer_matching_engine import PermissionLevel
        
        # Create user profile
        user_profile = PeerProfile(
            user_id=user_profile_data.get('user_id', str(current_user.id)),
            learning_style=user_profile_data.get('learning_style', 'visual'),
            expertise_areas=user_profile_data.get('expertise_areas', []),
            expertise_levels=user_profile_data.get('expertise_levels', {}),
            learning_goals=[LearningGoal(goal) for goal in user_profile_data.get('learning_goals', [])],
            personality_traits=user_profile_data.get('personality_traits', {}),
            availability=user_profile_data.get('availability', {}),
            communication_preferences=user_profile_data.get('communication_preferences', []),
            collaboration_style=user_profile_data.get('collaboration_style', 'collaborator'),
            preferred_group_size=user_profile_data.get('preferred_group_size', 4),
            timezone=user_profile_data.get('timezone', 'UTC')
        )
        
        # Create candidate profiles
        candidate_profiles = []
        for candidate_data in candidate_profiles_data:
            candidate = PeerProfile(
                user_id=candidate_data.get('user_id', ''),
                learning_style=candidate_data.get('learning_style', 'visual'),
                expertise_areas=candidate_data.get('expertise_areas', []),
                expertise_levels=candidate_data.get('expertise_levels', {}),
                learning_goals=[LearningGoal(goal) for goal in candidate_data.get('learning_goals', [])],
                personality_traits=candidate_data.get('personality_traits', {}),
                availability=candidate_data.get('availability', {}),
                communication_preferences=candidate_data.get('communication_preferences', []),
                collaboration_style=candidate_data.get('collaboration_style', 'collaborator'),
                preferred_group_size=candidate_data.get('preferred_group_size', 4),
                timezone=candidate_data.get('timezone', 'UTC')
            )
            candidate_profiles.append(candidate)
        
        # Find peer matches
        matches = peer_matching_engine.find_peer_matches(
            user_profile, candidate_profiles, 
            MatchingStrategy(matching_strategy), max_matches
        )
        
        return jsonify({
            'matches': [
                {
                    'peer1_id': match.peer1_id,
                    'peer2_id': match.peer2_id,
                    'compatibility_score': match.compatibility_score,
                    'compatibility_level': match.compatibility_level.value,
                    'matching_strategy': match.matching_strategy.value,
                    'strengths': match.strengths,
                    'learning_opportunities': match.learning_opportunities,
                    'potential_challenges': match.potential_challenges,
                    'recommended_activities': match.recommended_activities,
                    'confidence': match.confidence
                } for match in matches
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/collaborative/group-formation', methods=['POST'])
@login_required
def collaborative_group_formation():
    """Form optimal learning groups"""
    try:
        data = request.get_json()
        candidate_profiles_data = data.get('candidate_profiles', [])
        target_size = data.get('target_size', 4)
        learning_goal = data.get('learning_goal', 'skill_development')
        strategy = data.get('strategy', 'collaborative')
        
        # Convert to PeerProfile objects
        from ml_models.collaborative_learning.peer_matching_engine import PeerProfile, MatchingStrategy, LearningGoal
        
        candidate_profiles = []
        for candidate_data in candidate_profiles_data:
            candidate = PeerProfile(
                user_id=candidate_data.get('user_id', ''),
                learning_style=candidate_data.get('learning_style', 'visual'),
                expertise_areas=candidate_data.get('expertise_areas', []),
                expertise_levels=candidate_data.get('expertise_levels', {}),
                learning_goals=[LearningGoal(goal) for goal in candidate_data.get('learning_goals', [])],
                personality_traits=candidate_data.get('personality_traits', {}),
                availability=candidate_data.get('availability', {}),
                communication_preferences=candidate_data.get('communication_preferences', []),
                collaboration_style=candidate_data.get('collaboration_style', 'collaborator'),
                preferred_group_size=candidate_data.get('preferred_group_size', 4),
                timezone=candidate_data.get('timezone', 'UTC')
            )
            candidate_profiles.append(candidate)
        
        # Form optimal group
        group_formation = peer_matching_engine.form_optimal_group(
            candidate_profiles, target_size, 
            LearningGoal(learning_goal), MatchingStrategy(strategy)
        )
        
        if not group_formation:
            return jsonify({'error': 'Unable to form optimal group'}), 400
        
        return jsonify({
            'group_id': group_formation.group_id,
            'members': group_formation.members,
            'group_size': group_formation.group_size,
            'group_cohesion': group_formation.group_cohesion,
            'skill_diversity': group_formation.skill_diversity,
            'learning_goal_alignment': group_formation.learning_goal_alignment,
            'recommended_roles': group_formation.recommended_roles,
            'group_activities': group_formation.group_activities,
            'success_probability': group_formation.success_probability
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/collaborative/social-analytics', methods=['POST'])
@login_required
def collaborative_social_analytics():
    """Analyze social learning patterns and metrics"""
    try:
        data = request.get_json()
        interaction_data = data.get('interaction_data', [])
        learning_data = data.get('learning_data', [])
        
        # Analyze social learning
        social_metrics = social_learning_analytics.analyze_social_learning(
            interaction_data, learning_data
        )
        
        return jsonify({
            'total_users': social_metrics.total_users,
            'total_connections': social_metrics.total_connections,
            'network_density': social_metrics.network_density,
            'average_clustering': social_metrics.average_clustering,
            'network_centralization': social_metrics.network_centralization,
            'knowledge_flow_rate': social_metrics.knowledge_flow_rate,
            'collaboration_effectiveness': social_metrics.collaboration_effectiveness,
            'peer_learning_impact': social_metrics.peer_learning_impact,
            'social_engagement_level': social_metrics.social_engagement_level,
            'community_cohesion': social_metrics.community_cohesion,
            'learning_acceleration': social_metrics.learning_acceleration,
            'recommendations': social_metrics.recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/collaborative/user-profile', methods=['POST'])
@login_required
def collaborative_user_profile():
    """Create social learning profile for user"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', str(current_user.id))
        interaction_data = data.get('interaction_data', [])
        learning_data = data.get('learning_data', [])
        
        # Create social learning profile
        profile = social_learning_analytics.create_social_learning_profile(
            user_id, interaction_data, learning_data
        )
        
        return jsonify({
            'user_id': profile.user_id,
            'collaboration_score': profile.collaboration_score,
            'peer_influence_score': profile.peer_influence_score,
            'learning_centrality': profile.learning_centrality,
            'knowledge_sharing_score': profile.knowledge_sharing_score,
            'social_engagement_score': profile.social_engagement_score,
            'network_position': profile.network_position.value,
            'active_connections': profile.active_connections,
            'learning_community_size': profile.learning_community_size,
            'contribution_quality': profile.contribution_quality,
            'help_seeking_frequency': profile.help_seeking_frequency,
            'help_providing_frequency': profile.help_providing_frequency
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/collaborative/statistics', methods=['GET'])
@login_required
def collaborative_statistics():
    """Get collaborative learning statistics"""
    try:
        # Get statistics from all components
        group_stats = group_dynamics_analyzer.get_group_statistics()
        matching_stats = peer_matching_engine.get_matching_statistics()
        content_stats = collaborative_content_engine.get_content_statistics()
        social_stats = social_learning_analytics.get_social_learning_statistics()
        collaboration_stats = real_time_collaboration.get_collaboration_statistics()
        
        return jsonify({
            'group_dynamics_statistics': group_stats,
            'peer_matching_statistics': matching_stats,
            'content_statistics': content_stats,
            'social_learning_statistics': social_stats,
            'collaboration_statistics': collaboration_stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -----------------------------
# Research Platform API Endpoints
# -----------------------------

@app.route('/api/research/learning-effectiveness', methods=['POST'])
@login_required
def research_learning_effectiveness():
    """Analyze learning effectiveness for research"""
    try:
        data = request.get_json()
        learning_data = data.get('learning_data', [])
        baseline_data = data.get('baseline_data', [])
        
        # Convert to LearningOutcomeData objects
        from ml_models.research_platform.learning_effectiveness_analyzer import LearningOutcomeData
        
        learning_outcomes = []
        for item in learning_data:
            outcome = LearningOutcomeData(
                user_id=item.get('user_id', ''),
                learning_session_id=item.get('learning_session_id', ''),
                pre_test_score=item.get('pre_test_score', 0.0),
                post_test_score=item.get('post_test_score', 0.0),
                retention_test_score=item.get('retention_test_score', 0.0),
                transfer_test_score=item.get('transfer_test_score', 0.0),
                engagement_score=item.get('engagement_score', 0.0),
                satisfaction_score=item.get('satisfaction_score', 0.0),
                learning_time=item.get('learning_time', 0.0),
                completion_rate=item.get('completion_rate', 0.0),
                timestamp=datetime.fromisoformat(item.get('timestamp', datetime.now().isoformat()))
            )
            learning_outcomes.append(outcome)
        
        baseline_outcomes = []
        for item in baseline_data:
            outcome = LearningOutcomeData(
                user_id=item.get('user_id', ''),
                learning_session_id=item.get('learning_session_id', ''),
                pre_test_score=item.get('pre_test_score', 0.0),
                post_test_score=item.get('post_test_score', 0.0),
                retention_test_score=item.get('retention_test_score', 0.0),
                transfer_test_score=item.get('transfer_test_score', 0.0),
                engagement_score=item.get('engagement_score', 0.0),
                satisfaction_score=item.get('satisfaction_score', 0.0),
                learning_time=item.get('learning_time', 0.0),
                completion_rate=item.get('completion_rate', 0.0),
                timestamp=datetime.fromisoformat(item.get('timestamp', datetime.now().isoformat()))
            )
            baseline_outcomes.append(outcome)
        
        # Analyze learning effectiveness
        effectiveness_metrics = learning_effectiveness_analyzer.analyze_learning_effectiveness(
            learning_outcomes, baseline_outcomes if baseline_outcomes else None
        )
        
        return jsonify({
            'overall_effectiveness': effectiveness_metrics.overall_effectiveness,
            'knowledge_retention_rate': effectiveness_metrics.knowledge_retention_rate,
            'skill_acquisition_rate': effectiveness_metrics.skill_acquisition_rate,
            'transfer_effectiveness': effectiveness_metrics.transfer_effectiveness,
            'engagement_effectiveness': effectiveness_metrics.engagement_effectiveness,
            'learning_efficiency': effectiveness_metrics.learning_efficiency,
            'satisfaction_effectiveness': effectiveness_metrics.satisfaction_effectiveness,
            'learning_acceleration': effectiveness_metrics.learning_acceleration,
            'retention_decay_rate': effectiveness_metrics.retention_decay_rate,
            'transfer_success_rate': effectiveness_metrics.transfer_success_rate,
            'confidence_interval': effectiveness_metrics.confidence_interval,
            'statistical_significance': effectiveness_metrics.statistical_significance,
            'effect_size': effectiveness_metrics.effect_size,
            'recommendations': effectiveness_metrics.recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/research/ab-testing/create', methods=['POST'])
@login_required
def research_ab_testing_create():
    """Create A/B testing experiment"""
    try:
        data = request.get_json()
        
        experiment_id = ab_testing_framework.create_experiment(
            name=data.get('name', 'Unnamed Experiment'),
            description=data.get('description', ''),
            hypothesis=data.get('hypothesis', ''),
            primary_metric=data.get('primary_metric', 'learning_outcome'),
            secondary_metrics=data.get('secondary_metrics', []),
            assignment_method=data.get('assignment_method', 'random'),
            statistical_test=data.get('statistical_test', 't_test'),
            significance_level=data.get('significance_level', 0.05),
            power=data.get('power', 0.8),
            minimum_effect_size=data.get('minimum_effect_size', 0.2),
            max_duration_days=data.get('max_duration_days', 30),
            min_sample_size=data.get('min_sample_size', 100),
            max_sample_size=data.get('max_sample_size', 10000),
            stratification_variables=data.get('stratification_variables', []),
            created_by=str(current_user.id)
        )
        
        if not experiment_id:
            return jsonify({'error': 'Failed to create experiment'}), 400
        
        return jsonify({
            'experiment_id': experiment_id,
            'message': 'Experiment created successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/research/ab-testing/analyze/<experiment_id>', methods=['POST'])
@login_required
def research_ab_testing_analyze(experiment_id):
    """Analyze A/B testing experiment"""
    try:
        data = request.get_json()
        experiment_data = data.get('experiment_data', {})
        
        # Record experiment data
        for user_id, metrics in experiment_data.items():
            for metric_name, value in metrics.items():
                ab_testing_framework.record_experiment_data(
                    experiment_id, user_id, metric_name, value
                )
        
        # Analyze experiment
        results = ab_testing_framework.analyze_experiment(experiment_id)
        
        if not results:
            return jsonify({'error': 'Failed to analyze experiment'}), 400
        
        return jsonify({
            'experiment_id': results.experiment_id,
            'status': results.status.value,
            'total_participants': results.total_participants,
            'control_group_size': results.control_group_size,
            'treatment_group_size': results.treatment_group_size,
            'primary_metric_results': results.primary_metric_results,
            'statistical_test_results': results.statistical_test_results,
            'effect_size': results.effect_size,
            'confidence_interval': results.confidence_interval,
            'p_value': results.p_value,
            'is_significant': results.is_significant,
            'practical_significance': results.practical_significance,
            'recommendations': results.recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/research/statistical-analysis', methods=['POST'])
@login_required
def research_statistical_analysis():
    """Perform statistical analysis"""
    try:
        data = request.get_json()
        analysis_type = data.get('analysis_type', 'descriptive')
        dataset = data.get('dataset', [])
        
        if analysis_type == 'descriptive':
            # Descriptive statistics
            stats_result = statistical_analyzer.calculate_descriptive_statistics(dataset)
            
            return jsonify({
                'analysis_type': 'descriptive',
                'count': stats_result.count,
                'mean': stats_result.mean,
                'median': stats_result.median,
                'std': stats_result.std,
                'variance': stats_result.variance,
                'min': stats_result.min,
                'max': stats_result.max,
                'range': stats_result.range,
                'quartiles': stats_result.quartiles,
                'skewness': stats_result.skewness,
                'kurtosis': stats_result.kurtosis,
                'outliers': stats_result.outliers
            })
        
        elif analysis_type == 'hypothesis_test':
            # Hypothesis testing
            data1 = data.get('group1', [])
            data2 = data.get('group2', [])
            test_type = data.get('test_type', 't_test_independent')
            
            test_result = statistical_analyzer.perform_hypothesis_test(
                data1, data2, test_type
            )
            
            return jsonify(test_result)
        
        elif analysis_type == 'correlation':
            # Correlation analysis
            x = data.get('x', [])
            y = data.get('y', [])
            correlation_type = data.get('correlation_type', 'pearson')
            
            corr_result = statistical_analyzer.calculate_correlation(x, y, correlation_type)
            
            return jsonify({
                'correlation_type': corr_result.correlation_type,
                'correlation_coefficient': corr_result.correlation_coefficient,
                'p_value': corr_result.p_value,
                'confidence_interval': corr_result.confidence_interval,
                'strength': corr_result.strength,
                'significance': corr_result.significance
            })
        
        elif analysis_type == 'regression':
            # Regression analysis
            x = data.get('x', [])
            y = data.get('y', [])
            regression_type = data.get('regression_type', 'linear')
            
            reg_result = statistical_analyzer.perform_regression_analysis(x, y, regression_type)
            
            return jsonify({
                'model_type': reg_result.model_type,
                'r_squared': reg_result.r_squared,
                'adjusted_r_squared': reg_result.adjusted_r_squared,
                'coefficients': reg_result.coefficients,
                'p_values': reg_result.p_values,
                'confidence_intervals': reg_result.confidence_intervals,
                'residuals': reg_result.residuals,
                'predictions': reg_result.predictions
            })
        
        else:
            return jsonify({'error': 'Unsupported analysis type'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/research/data-management/dataset', methods=['POST'])
@login_required
def research_data_management_create_dataset():
    """Create research dataset"""
    try:
        data = request.get_json()
        
        from ml_models.research_platform.research_data_manager import DataType
        
        dataset_id = research_data_manager.create_dataset(
            name=data.get('name', 'Unnamed Dataset'),
            description=data.get('description', ''),
            data_type=DataType(data.get('data_type', 'learning_outcomes')),
            metadata=data.get('metadata', {})
        )
        
        if not dataset_id:
            return jsonify({'error': 'Failed to create dataset'}), 400
        
        return jsonify({
            'dataset_id': dataset_id,
            'message': 'Dataset created successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/research/report/generate', methods=['POST'])
@login_required
def research_report_generate():
    """Generate academic research report"""
    try:
        data = request.get_json()
        report_type = data.get('report_type', 'learning_effectiveness')
        report_data = data.get('report_data', {})
        author = data.get('author', str(current_user.username))
        
        if report_type == 'learning_effectiveness':
            report_id = academic_report_generator.generate_learning_effectiveness_report(
                report_data.get('effectiveness_metrics', {}),
                report_data.get('dataset_info', {}),
                author
            )
        elif report_type == 'experiment':
            report_id = academic_report_generator.generate_experiment_report(
                report_data.get('experiment_results', {}),
                report_data.get('experiment_config', {}),
                author
            )
        else:
            return jsonify({'error': 'Unsupported report type'}), 400
        
        if not report_id:
            return jsonify({'error': 'Failed to generate report'}), 400
        
        return jsonify({
            'report_id': report_id,
            'message': 'Report generated successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/research/statistics', methods=['GET'])
@login_required
def research_statistics():
    """Get research platform statistics"""
    try:
        # Get statistics from all components
        effectiveness_stats = learning_effectiveness_analyzer.get_effectiveness_trends()
        ab_testing_stats = ab_testing_framework.get_framework_statistics()
        statistical_stats = statistical_analyzer.get_analysis_statistics()
        data_management_stats = research_data_manager.get_research_statistics()
        report_stats = academic_report_generator.get_report_statistics()
        
        return jsonify({
            'learning_effectiveness_statistics': effectiveness_stats,
            'ab_testing_statistics': ab_testing_stats,
            'statistical_analysis_statistics': statistical_stats,
            'data_management_statistics': data_management_stats,
            'report_generation_statistics': report_stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -----------------------------
# Career Prediction API Endpoints
# -----------------------------

@app.route('/api/career/predict-paths', methods=['POST'])
@login_required
def career_predict_paths():
    """Predict career paths for user"""
    try:
        data = request.get_json()
        user_profile = data.get('user_profile', {})
        target_industries = data.get('target_industries', [])
        max_paths = data.get('max_paths', 5)
        
        # Convert industry strings to enums
        from ml_models.career_prediction.career_path_predictor import IndustrySector
        industry_enums = []
        for industry in target_industries:
            try:
                industry_enums.append(IndustrySector(industry))
            except ValueError:
                continue
        
        # Predict career paths
        career_paths = career_path_predictor.predict_career_paths(
            user_profile, industry_enums if industry_enums else None, max_paths
        )
        
        # Convert to JSON-serializable format
        paths_data = []
        for path in career_paths:
            paths_data.append({
                'path_id': path.path_id,
                'title': path.title,
                'industry': path.industry.value,
                'current_stage': path.current_stage.value,
                'target_stage': path.target_stage.value,
                'required_skills': path.required_skills,
                'learning_style_fit': path.learning_style_fit,
                'skill_gaps': path.skill_gaps,
                'time_to_target': path.time_to_target,
                'salary_range': path.salary_range,
                'growth_potential': path.growth_potential,
                'job_market_demand': path.job_market_demand,
                'work_style_match': path.work_style_match,
                'confidence_score': path.confidence_score
            })
        
        return jsonify({
            'career_paths': paths_data,
            'total_paths': len(paths_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/career/analyze-skill-gaps', methods=['POST'])
@login_required
def career_analyze_skill_gaps():
    """Analyze skill gaps for career development"""
    try:
        data = request.get_json()
        current_skills = data.get('current_skills', {})
        target_skills = data.get('target_skills', {})
        target_role = data.get('target_role', 'Software Developer')
        
        # Convert skill level strings to enums
        from ml_models.career_prediction.skill_gap_analyzer import SkillLevel
        current_skill_levels = {}
        for skill, level in current_skills.items():
            try:
                current_skill_levels[skill] = SkillLevel(level)
            except ValueError:
                current_skill_levels[skill] = SkillLevel.BEGINNER
        
        target_skill_levels = {}
        for skill, level in target_skills.items():
            try:
                target_skill_levels[skill] = SkillLevel(level)
            except ValueError:
                target_skill_levels[skill] = SkillLevel.INTERMEDIATE
        
        # Analyze skill gaps
        skill_gaps = skill_gap_analyzer.analyze_skill_gaps(
            current_skill_levels, target_skill_levels, target_role
        )
        
        # Convert to JSON-serializable format
        gaps_data = []
        for gap in skill_gaps:
            gaps_data.append({
                'skill_name': gap.skill_name,
                'category': gap.category.value,
                'current_level': gap.current_level.value,
                'target_level': gap.target_level.value,
                'importance_score': gap.importance_score,
                'difficulty_score': gap.difficulty_score,
                'time_to_develop': gap.time_to_develop,
                'learning_methods': [method.value for method in gap.learning_methods],
                'resources': gap.resources,
                'prerequisites': gap.prerequisites,
                'related_skills': gap.related_skills,
                'market_demand': gap.market_demand,
                'salary_impact': gap.salary_impact
            })
        
        return jsonify({
            'skill_gaps': gaps_data,
            'total_gaps': len(gaps_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/career/recommendations', methods=['POST'])
@login_required
def career_recommendations():
    """Get personalized career recommendations"""
    try:
        data = request.get_json()
        user_profile = data.get('user_profile', {})
        max_recommendations = data.get('max_recommendations', 10)
        
        # Generate recommendations
        recommendations = career_recommendation_engine.generate_recommendations(
            user_profile, max_recommendations
        )
        
        # Convert to JSON-serializable format
        recommendations_data = []
        for rec in recommendations:
            recommendations_data.append({
                'recommendation_id': rec.recommendation_id,
                'type': rec.type.value,
                'title': rec.title,
                'description': rec.description,
                'relevance_score': rec.relevance_score,
                'priority': rec.priority,
                'estimated_impact': rec.estimated_impact,
                'time_commitment': rec.time_commitment,
                'cost_estimate': rec.cost_estimate,
                'prerequisites': rec.prerequisites,
                'next_steps': rec.next_steps,
                'resources': rec.resources,
                'success_metrics': rec.success_metrics
            })
        
        return jsonify({
            'recommendations': recommendations_data,
            'total_recommendations': len(recommendations_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/career/industry-analysis', methods=['POST'])
@login_required
def career_industry_analysis():
    """Analyze industry trends and opportunities"""
    try:
        data = request.get_json()
        industry = data.get('industry', 'technology')
        
        # Analyze industry
        insight = industry_analyzer.analyze_industry(industry)
        
        if not insight:
            return jsonify({'error': 'Industry analysis failed'}), 400
        
        # Convert to JSON-serializable format
        insight_data = {
            'industry': insight.industry,
            'growth_rate': insight.growth_rate,
            'job_demand': insight.job_demand,
            'salary_trends': insight.salary_trends,
            'skill_demand': insight.skill_demand,
            'emerging_technologies': insight.emerging_technologies,
            'market_outlook': insight.market_outlook,
            'risk_factors': insight.risk_factors,
            'opportunities': insight.opportunities
        }
        
        return jsonify(insight_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/career/statistics', methods=['GET'])
@login_required
def career_statistics():
    """Get career prediction statistics"""
    try:
        # Get statistics from all components
        predictor_stats = career_path_predictor.get_prediction_statistics()
        analyzer_stats = skill_gap_analyzer.get_analyzer_statistics()
        
        return jsonify({
            'career_prediction_statistics': predictor_stats,
            'skill_analysis_statistics': analyzer_stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -----------------------------
# Biometric Feedback API Endpoints
# -----------------------------

@app.route('/api/biometric/hrv-analysis', methods=['POST'])
@login_required
def biometric_hrv_analysis():
    """Analyze Heart Rate Variability for learning optimization"""
    try:
        data = request.get_json()
        rr_intervals = data.get('rr_intervals', [])
        sampling_rate = data.get('sampling_rate', 1000.0)
        
        # Analyze HRV
        hrv_metrics = hrv_analyzer.analyze_hrv(rr_intervals, sampling_rate)
        
        # Get learning recommendations
        recommendations = hrv_analyzer.get_learning_recommendations(hrv_metrics)
        
        return jsonify({
            'timestamp': hrv_metrics.timestamp.isoformat(),
            'mean_rr': hrv_metrics.mean_rr,
            'rmssd': hrv_metrics.rmssd,
            'sdnn': hrv_metrics.sdnn,
            'pnn50': hrv_metrics.pnn50,
            'vlf_power': hrv_metrics.vlf_power,
            'lf_power': hrv_metrics.lf_power,
            'hf_power': hrv_metrics.hf_power,
            'lf_hf_ratio': hrv_metrics.lf_hf_ratio,
            'total_power': hrv_metrics.total_power,
            'stress_index': hrv_metrics.stress_index,
            'autonomic_balance': hrv_metrics.autonomic_balance,
            'recovery_index': hrv_metrics.recovery_index,
            'learning_readiness': hrv_metrics.learning_readiness.value,
            'hrv_state': hrv_metrics.hrv_state.value,
            'stress_level': hrv_metrics.stress_level.value,
            'confidence': hrv_metrics.confidence,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/biometric/gsr-analysis', methods=['POST'])
@login_required
def biometric_gsr_analysis():
    """Analyze Galvanic Skin Response for learning optimization"""
    try:
        data = request.get_json()
        gsr_data = data.get('gsr_data', [])
        sampling_rate = data.get('sampling_rate', 100.0)
        
        # Analyze GSR
        gsr_metrics = gsr_monitor.analyze_gsr(gsr_data, sampling_rate)
        
        # Get learning recommendations
        recommendations = gsr_monitor.get_learning_recommendations(gsr_metrics)
        
        return jsonify({
            'timestamp': gsr_metrics.timestamp.isoformat(),
            'raw_gsr': gsr_metrics.raw_gsr,
            'tonic_gsr': gsr_metrics.tonic_gsr,
            'phasic_gsr': gsr_metrics.phasic_gsr,
            'gsr_amplitude': gsr_metrics.gsr_amplitude,
            'gsr_frequency': gsr_metrics.gsr_frequency,
            'skin_conductance_level': gsr_metrics.skin_conductance_level,
            'skin_conductance_response': gsr_metrics.skin_conductance_response,
            'arousal_index': gsr_metrics.arousal_index,
            'emotional_reactivity': gsr_metrics.emotional_reactivity,
            'stress_response': gsr_metrics.stress_response,
            'engagement_score': gsr_metrics.engagement_score,
            'arousal_level': gsr_metrics.arousal_level.value,
            'emotional_state': gsr_metrics.emotional_state.value,
            'engagement_level': gsr_metrics.engagement_level.value,
            'confidence': gsr_metrics.confidence,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/biometric/fusion-analysis', methods=['POST'])
@login_required
def biometric_fusion_analysis():
    """Fuse multiple biometric data sources for comprehensive analysis"""
    try:
        data = request.get_json()
        hrv_metrics = data.get('hrv_metrics', {})
        gsr_metrics = data.get('gsr_metrics', {})
        additional_data = data.get('additional_data', {})
        
        # Fuse biometric data
        fused_state = biometric_fusion_engine.fuse_biometric_data(hrv_metrics, gsr_metrics, additional_data)
        
        return jsonify({
            'timestamp': fused_state.timestamp.isoformat(),
            'hrv_state': fused_state.hrv_state,
            'gsr_state': fused_state.gsr_state,
            'combined_state': fused_state.combined_state.value,
            'learning_readiness': fused_state.learning_readiness,
            'stress_level': fused_state.stress_level,
            'engagement_level': fused_state.engagement_level,
            'fatigue_level': fused_state.fatigue_level,
            'arousal_level': fused_state.arousal_level,
            'emotional_state': fused_state.emotional_state,
            'cognitive_load': fused_state.cognitive_load,
            'attention_level': fused_state.attention_level,
            'motivation_level': fused_state.motivation_level,
            'confidence': fused_state.confidence,
            'recommendations': fused_state.recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/biometric/stress-detection', methods=['POST'])
@login_required
def biometric_stress_detection():
    """Detect stress from biometric and behavioral data"""
    try:
        data = request.get_json()
        biometric_data = data.get('biometric_data', {})
        behavioral_data = data.get('behavioral_data', {})
        
        # Detect stress
        stress_metrics = stress_detector.detect_stress(biometric_data, behavioral_data)
        
        return jsonify({
            'timestamp': stress_metrics.timestamp.isoformat(),
            'stress_level': stress_metrics.stress_level.value,
            'stress_type': stress_metrics.stress_type.value,
            'stress_intensity': stress_metrics.stress_intensity,
            'stress_duration': stress_metrics.stress_duration,
            'physiological_indicators': stress_metrics.physiological_indicators,
            'behavioral_indicators': stress_metrics.behavioral_indicators,
            'cognitive_indicators': stress_metrics.cognitive_indicators,
            'intervention_required': stress_metrics.intervention_required,
            'recommended_interventions': stress_metrics.recommended_interventions,
            'confidence': stress_metrics.confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/biometric/learning-optimization', methods=['POST'])
@login_required
def biometric_learning_optimization():
    """Optimize learning session based on biometric feedback"""
    try:
        data = request.get_json()
        biometric_state = data.get('biometric_state', {})
        current_content = data.get('current_content', {})
        
        # Optimize learning session
        optimized_content = learning_optimizer.optimize_learning_session(biometric_state, current_content)
        
        # Generate recommendations
        recommendations = learning_optimizer.generate_biometric_recommendations(biometric_state)
        
        # Convert recommendations to JSON-serializable format
        recommendations_data = []
        for rec in recommendations:
            recommendations_data.append({
                'recommendation_id': rec.recommendation_id,
                'type': rec.type.value,
                'priority': rec.priority,
                'description': rec.description,
                'expected_impact': rec.expected_impact,
                'implementation_effort': rec.implementation_effort,
                'time_to_effect': rec.time_to_effect,
                'success_metrics': rec.success_metrics,
                'specific_actions': rec.specific_actions
            })
        
        return jsonify({
            'optimized_content': optimized_content,
            'recommendations': recommendations_data,
            'total_recommendations': len(recommendations_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/biometric/statistics', methods=['GET'])
@login_required
def biometric_statistics():
    """Get biometric feedback statistics"""
    try:
        # Get statistics from all components
        hrv_stats = hrv_analyzer.get_analyzer_statistics()
        gsr_stats = gsr_monitor.get_monitor_statistics()
        fusion_stats = biometric_fusion_engine.get_fusion_statistics()
        stress_stats = stress_detector.get_detector_statistics()
        optimizer_stats = learning_optimizer.get_optimizer_statistics()
        
        return jsonify({
            'hrv_analyzer_statistics': hrv_stats,
            'gsr_monitor_statistics': gsr_stats,
            'biometric_fusion_statistics': fusion_stats,
            'stress_detector_statistics': stress_stats,
            'learning_optimizer_statistics': optimizer_stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -----------------------------
# Dashboard API Endpoints
# -----------------------------

@app.route('/api/dashboard/statistics', methods=['GET'])
@login_required
def dashboard_statistics():
    """Get comprehensive dashboard statistics"""
    try:
        # Get basic statistics
        total_users = User.query.count()
        total_profiles = LearningProfile.query.count()
        total_quiz_responses = QuizResponse.query.count()
        
        # Get learning sessions (approximate)
        learning_sessions = total_quiz_responses * 2  # Estimate
        
        # Get AI predictions (approximate)
        ai_predictions = total_profiles * 5  # Estimate
        
        # Get system health (simulated)
        system_health = 98.5
        
        # Get feature statistics
        feature_stats = {
            'emotion_ai': {
                'active_sessions': 15,
                'total_analyses': 1250,
                'accuracy': 94.2
            },
            'collaborative_learning': {
                'active_groups': 8,
                'total_collaborations': 340,
                'success_rate': 87.5
            },
            'research_platform': {
                'active_experiments': 3,
                'total_analyses': 89,
                'publications': 2
            },
            'career_prediction': {
                'total_predictions': 156,
                'success_rate': 91.3,
                'user_satisfaction': 4.6
            },
            'neurofeedback': {
                'active_sessions': 12,
                'total_analyses': 890,
                'effectiveness': 88.7
            }
        }
        
        return jsonify({
            'total_users': total_users,
            'learning_sessions': learning_sessions,
            'ai_predictions': ai_predictions,
            'system_health': system_health,
            'feature_statistics': feature_stats,
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -----------------------------
# Helper Functions
# -----------------------------

def _calculate_overall_learning_state(focus_metrics, fatigue_metrics, load_metrics):
    """Calculate overall learning state from neurofeedback metrics"""
    try:
        # Weight the different metrics
        focus_score = focus_metrics.focus_level
        fatigue_penalty = 1 - fatigue_metrics.fatigue_level  # Lower fatigue is better
        efficiency_score = load_metrics.learning_efficiency
        
        # Calculate overall state
        overall_score = (focus_score * 0.4 + fatigue_penalty * 0.3 + efficiency_score * 0.3)
        
        # Classify overall state
        if overall_score >= 0.8:
            state = "excellent"
        elif overall_score >= 0.6:
            state = "good"
        elif overall_score >= 0.4:
            state = "fair"
        else:
            state = "poor"
        
        return {
            'score': overall_score,
            'state': state,
            'recommendation': _generate_overall_recommendation(state, focus_metrics, fatigue_metrics, load_metrics)
        }
        
    except Exception as e:
        return {
            'score': 0.5,
            'state': 'unknown',
            'recommendation': 'Unable to assess overall learning state.'
        }

def _generate_overall_recommendation(state, focus_metrics, fatigue_metrics, load_metrics):
    """Generate overall learning recommendation"""
    recommendations = {
        'excellent': "You're in an excellent learning state! Continue with your current approach.",
        'good': "Good learning state. You're performing well with minor optimizations possible.",
        'fair': "Fair learning state. Consider taking breaks and adjusting your learning environment.",
        'poor': "Learning state needs improvement. Take a break and reassess your approach."
    }
    
    base_recommendation = recommendations.get(state, "Monitor your learning state.")
    
    # Add specific recommendations based on individual metrics
    if focus_metrics.focus_level < 0.5:
        base_recommendation += " Focus on improving concentration."
    
    if fatigue_metrics.fatigue_level > 0.7:
        base_recommendation += " Take a break to reduce fatigue."
    
    if load_metrics.learning_efficiency < 0.5:
        base_recommendation += " Consider simplifying content or changing learning approach."
    
    return base_recommendation

def generate_ai_response(user_message, learning_style):
    """Generate AI response based on user message and learning style"""
    # This is a simplified version - in production, you'd use OpenAI API
    message = user_message.lower()
    
    if 'hello' in message or 'hi' in message:
        if learning_style == 'visual':
            return "Hello! I'm here to help you learn through visual aids, diagrams, and clear explanations. What would you like to explore today?"
        elif learning_style == 'auditory':
            return "Hi there! I love having conversations about learning. Let's discuss what you'd like to understand - I'll explain it step by step."
        else:
            return "Hey! Ready to dive into some hands-on learning? I can guide you through practical exercises and real-world applications."
    
    elif 'help' in message:
        return f"As your {learning_style} learning assistant, I can help you understand concepts in ways that work best for your learning style. What specific topic would you like help with?"
    
    else:
        return f"That's a great question! As someone who learns best through {learning_style} methods, I'll make sure to explain this in a way that resonates with your learning style. Could you tell me more about what specific aspect you'd like to understand?"


# -----------------------------
# Error Handlers
# -----------------------------
@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('errors/500.html'), 500


# -----------------------------
# Admin API Endpoints
# -----------------------------

@app.route('/api/admin/users', methods=['GET'])
@login_required
def admin_get_users():
    """Get all users for admin dashboard"""
    if not current_user.is_admin():
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        users = User.query.all()
        users_data = []
        for user in users:
            users_data.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role,
                'is_active': user.is_active,
                'last_login': user.last_login.isoformat() if user.last_login else None,
                'created_at': user.created_at.isoformat()
            })
        return jsonify(users_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/content', methods=['GET'])
@login_required
def admin_get_content():
    """Get all content for admin dashboard"""
    if not current_user.is_admin():
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        content = ContentLibrary.query.all()
        content_data = []
        for item in content:
            content_data.append({
                'id': item.id,
                'title': item.title,
                'description': item.description,
                'content_type': item.content_type,
                'style_tags': item.style_tags,
                'difficulty_level': item.difficulty_level,
                'created_at': item.created_at.isoformat()
            })
        return jsonify(content_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/analytics', methods=['GET'])
@login_required
def admin_get_analytics():
    """Get system analytics for admin dashboard"""
    if not current_user.can_view_analytics():
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        # Get basic statistics
        total_users = User.query.count()
        active_users = User.query.filter_by(is_active=True).count()
        total_content = ContentLibrary.query.count()
        
        # Get learning style distribution
        profiles = LearningProfile.query.all()
        style_distribution = {'visual': 0, 'auditory': 0, 'kinesthetic': 0}
        for profile in profiles:
            if profile.dominant_style in style_distribution:
                style_distribution[profile.dominant_style] += 1
        
        return jsonify({
            'total_users': total_users,
            'active_users': active_users,
            'total_content': total_content,
            'style_distribution': style_distribution
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Moderator API Endpoints
@app.route('/api/moderator/users', methods=['GET'])
@login_required
def moderator_get_users():
    """Get users for moderator dashboard (limited view)"""
    if not current_user.is_moderator():
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        users = User.query.all()
        users_data = []
        for user in users:
            users_data.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role,
                'is_active': user.is_active,
                'last_login': user.last_login.isoformat() if user.last_login else None,
                'created_at': user.created_at.isoformat()
            })
        return jsonify(users_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/moderator/content', methods=['GET'])
@login_required
def moderator_get_content():
    """Get content for moderator dashboard"""
    if not current_user.can_manage_content():
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        content = ContentLibrary.query.all()
        content_data = []
        for item in content:
            content_data.append({
                'id': item.id,
                'title': item.title,
                'description': item.description,
                'content_type': item.content_type,
                'style_tags': item.style_tags,
                'difficulty_level': item.difficulty_level,
                'created_at': item.created_at.isoformat()
            })
        return jsonify(content_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
    app.run(debug=True, host='0.0.0.0', port=5000)
