"""
LearnStyle AI - API Routes
RESTful API endpoints for all application functionality
"""

from flask import Blueprint, request, jsonify, session, render_template, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import time
from sqlalchemy import func, or_
import sys
import os

# Add the parent directory to sys.path to import ml_models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app import db
from app.models import User, LearningProfile, QuizResponse, ContentLibrary, UserProgress, ChatHistory
from ml_models.learning_style_predictor import LearningStylePredictor

# Initialize ML predictor
ml_predictor = LearningStylePredictor()

# Create blueprints
api_bp = Blueprint('api', __name__, url_prefix='/api')
auth_bp = Blueprint('auth', __name__)

# Helper functions for browser extension authentication
def validate_extension_token(token):
    """Validate extension token and return user_id"""
    try:
        # Simple token validation - in production, use JWT or similar
        if token and token.startswith('ext_'):
            # Extract user_id from token (simple implementation)
            user_id = int(token.split('_')[1]) if len(token.split('_')) > 1 else None
            if user_id and User.query.get(user_id):
                return user_id
    except:
        pass
    return None

def get_or_create_guest_user():
    """Get or create a guest user for anonymous tracking"""
    try:
        # Look for existing guest user
        guest_user = User.query.filter_by(username='guest_user').first()
        if guest_user:
            return guest_user.id
        
        # Create new guest user
        guest_user = User(
            username='guest_user',
            email='guest@learnstyle.ai'
        )
        guest_user.set_password('guest123')
        db.session.add(guest_user)
        db.session.commit()
        
        # Create learning profile for guest
        profile = LearningProfile(user_id=guest_user.id)
        db.session.add(profile)
        db.session.commit()
        
        print(f"Created guest user with ID: {guest_user.id}")
        return guest_user.id
    except Exception as e:
        print(f"Error creating guest user: {e}")
        # Rollback the session and try to get existing user
        db.session.rollback()
        try:
            guest_user = User.query.filter_by(username='guest_user').first()
            if guest_user:
                return guest_user.id
        except:
            pass
        # Return a fallback user ID
        return 1

# Authentication Routes
@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'GET':
        return render_template('auth/register.html')
    
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        # Validation
        if not username or not email or not password:
            if request.is_json:
                return jsonify({'error': 'All fields are required'}), 400
            flash('All fields are required', 'error')
            return render_template('auth/register.html')
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            if request.is_json:
                return jsonify({'error': 'Username already exists'}), 400
            flash('Username already exists', 'error')
            return render_template('auth/register.html')
        
        if User.query.filter_by(email=email).first():
            if request.is_json:
                return jsonify({'error': 'Email already exists'}), 400
            flash('Email already exists', 'error')
            return render_template('auth/register.html')
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        # Create empty learning profile
        profile = LearningProfile(user_id=user.id)
        db.session.add(profile)
        db.session.commit()
        
        if request.is_json:
            return jsonify({'message': 'User registered successfully', 'user_id': user.id}), 201
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth.login'))

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'GET':
        return render_template('auth/login.html')
    
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            if request.is_json:
                return jsonify({'error': 'Username and password required'}), 400
            flash('Username and password required', 'error')
            return render_template('auth/login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            if request.is_json:
                return jsonify({'message': 'Login successful', 'user_id': user.id}), 200
            return redirect(url_for('dashboard'))
        
        if request.is_json:
            return jsonify({'error': 'Invalid credentials'}), 401
        flash('Invalid credentials', 'error')
        return render_template('auth/login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    if request.is_json:
        return jsonify({'message': 'Logged out successfully'}), 200
    return redirect(url_for('index'))

# API Routes
@api_bp.route('/quiz', methods=['POST'])
@login_required
def submit_quiz():
    """Submit quiz answers and get learning style prediction"""
    try:
        data = request.get_json()
        answers = data.get('answers')
        
        if not answers or len(answers) != 15:
            return jsonify({'error': 'Quiz must have exactly 15 answers'}), 400
        
        # Validate answer values
        for answer in answers:
            if answer not in [1, 2, 3]:
                return jsonify({'error': 'Answers must be 1, 2, or 3'}), 400
        
        # Save quiz response
        quiz_response = QuizResponse(user_id=current_user.id)
        for i, answer in enumerate(answers, 1):
            setattr(quiz_response, f'question_{i}', answer)
        
        db.session.add(quiz_response)
        
        # Load or train ML model if needed
        try:
            ml_predictor.load_models()
        except:
            # Generate training data and train model
            X, y = ml_predictor.generate_synthetic_dataset(1000)
            ml_predictor.train_models(X, y)
            ml_predictor.save_models()
        
        # Get prediction
        prediction = ml_predictor.predict_learning_style(answers)
        
        # Update learning profile
        profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
        profile.visual_score = prediction.get('visual', 0)
        profile.auditory_score = prediction.get('auditory', 0)
        profile.kinesthetic_score = prediction.get('kinesthetic', 0)
        profile.dominant_style = prediction.get('dominant_style')
        profile.last_updated = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'message': 'Quiz submitted successfully',
            'prediction': prediction,
            'style_breakdown': profile.get_style_breakdown()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/predict', methods=['POST'])
@login_required
def get_prediction():
    """Get learning style prediction for given answers"""
    try:
        data = request.get_json()
        answers = data.get('answers')
        
        if not answers or len(answers) != 15:
            return jsonify({'error': 'Must provide exactly 15 answers'}), 400
        
        # Load ML model
        try:
            ml_predictor.load_models()
        except:
            return jsonify({'error': 'ML model not trained yet'}), 503
        
        prediction = ml_predictor.predict_learning_style(answers)
        return jsonify(prediction), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/content', methods=['GET'])
@login_required
def get_personalized_content():
    """Get personalized content based on user's learning style"""
    try:
        profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
        
        if not profile or not profile.dominant_style:
            # Return general content if no profile
            content = ContentLibrary.query.limit(10).all()
        else:
            # Get content matching user's learning style
            style_filter = f'%{profile.dominant_style}%'
            content = ContentLibrary.query.filter(
                ContentLibrary.style_tags.like(style_filter)
            ).limit(10).all()
            
            # If not enough style-specific content, add general content
            if len(content) < 5:
                general_content = ContentLibrary.query.filter(
                    ~ContentLibrary.style_tags.like(style_filter)
                ).limit(10 - len(content)).all()
                content.extend(general_content)
        
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
            'user_style': profile.dominant_style if profile else None,
            'style_breakdown': profile.get_style_breakdown() if profile else None
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/progress', methods=['POST'])
@login_required
def update_progress():
    """Update user progress for content"""
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        completion_status = data.get('completion_status', 'started')
        time_spent = data.get('time_spent', 0)
        score = data.get('score')
        engagement_rating = data.get('engagement_rating')
        
        if not content_id:
            return jsonify({'error': 'content_id is required'}), 400
        
        # Check if progress record exists
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
        
        # Update progress
        progress.completion_status = completion_status
        progress.time_spent += time_spent
        if score is not None:
            progress.score = score
        if engagement_rating is not None:
            progress.engagement_rating = engagement_rating
        
        if completion_status == 'completed':
            progress.completed_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({'message': 'Progress updated successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/progress', methods=['GET'])
@login_required
def get_progress():
    """Get user progress data"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        progress_query = UserProgress.query.filter_by(user_id=current_user.id)
        
        # Get paginated results
        progress_paginated = progress_query.order_by(
            UserProgress.timestamp.desc()
        ).paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        progress_data = []
        for progress in progress_paginated.items:
            progress_data.append({
                'id': progress.id,
                'content_id': progress.content_id,
                'content_title': progress.content.title if progress.content else 'Unknown',
                'completion_status': progress.completion_status,
                'time_spent': progress.time_spent,
                'score': progress.score,
                'engagement_rating': progress.engagement_rating,
                'timestamp': progress.timestamp.isoformat(),
                'completed_at': progress.completed_at.isoformat() if progress.completed_at else None
            })
        
        return jsonify({
            'progress': progress_data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': progress_paginated.total,
                'pages': progress_paginated.pages,
                'has_next': progress_paginated.has_next,
                'has_prev': progress_paginated.has_prev
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/update-profile', methods=['POST'])
@login_required
def update_learning_profile():
    """Update user learning profile based on interactions"""
    try:
        from ml_models.learning_style_predictor import update_user_learning_profile
        
        # Get user interaction data
        interaction_data = request.get_json() or {}
        
        # Update learning profile
        success = update_user_learning_profile(current_user.id, interaction_data)
        
        if success:
            # Get updated profile
            profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
            return jsonify({
                'message': 'Learning profile updated successfully',
                'profile': {
                    'visual_score': profile.visual_score,
                    'auditory_score': profile.auditory_score,
                    'kinesthetic_score': profile.kinesthetic_score,
                    'dominant_style': profile.dominant_style,
                    'style_breakdown': profile.get_style_breakdown()
                }
            }), 200
        else:
            return jsonify({'error': 'Failed to update learning profile'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/generate-content', methods=['POST'])
@login_required
def generate_content():
    """Generate personalized content based on learning style"""
    try:
        from app.content_generator import ContentGenerator
        
        data = request.get_json()
        topic = data.get('topic', 'General Learning')
        content_type = data.get('content_type', 'article')
        
        # Get user's learning style
        profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
        learning_style = profile.dominant_style if profile else 'visual'
        
        # Generate content
        generator = ContentGenerator()
        generated_content = generator.generate_content(
            topic=topic,
            learning_style=learning_style,
            content_type=content_type,
            user_id=current_user.id
        )
        
        return jsonify({
            'success': True,
            'content': {
                'id': generated_content.id,
                'title': generated_content.title,
                'content': generated_content.content,
                'content_type': generated_content.content_type,
                'learning_style': generated_content.learning_style,
                'difficulty_level': generated_content.difficulty_level,
                'created_at': generated_content.created_at.isoformat()
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/chat', methods=['POST'])
@login_required
def ai_chat():
    """AI tutor chat with style-aware responses"""
    try:
        from app.utils.ai_tutor import AITutor
        
        data = request.get_json()
        user_message = data.get('message')
        conversation_history = data.get('conversation_history', [])
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get user's learning style for context
        profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
        learning_style = profile.dominant_style if profile else 'general'
        
        # Use AITutor for style-aware responses
        ai_tutor = AITutor()
        ai_response_data = ai_tutor.generate_response(
            user_message=user_message,
            learning_style=learning_style,
            conversation_history=conversation_history
        )
        
        # Save chat history
        chat_entry = ChatHistory(
            user_id=current_user.id,
            user_message=user_message,
            ai_response=ai_response_data['response'],
            learning_style_context=learning_style,
            topic_category=ai_response_data.get('topic', 'general'),
            response_type=ai_response_data.get('response_type', 'general')
        )
        db.session.add(chat_entry)
        db.session.commit()
        
        return jsonify({
            'response': ai_response_data['response'],
            'learning_style_context': learning_style,
            'topic': ai_response_data.get('topic', 'general'),
            'response_type': ai_response_data.get('response_type', 'general'),
            'timestamp': ai_response_data.get('timestamp'),
            'chat_id': chat_entry.id
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/chat-history', methods=['GET'])
@login_required
def get_chat_history():
    """Get user's chat history with AI tutor"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        chat_query = ChatHistory.query.filter_by(user_id=current_user.id)
        
        # Get paginated results
        chat_paginated = chat_query.order_by(
            ChatHistory.timestamp.desc()
        ).paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        chat_data = []
        for chat in chat_paginated.items:
            chat_data.append({
                'id': chat.id,
                'user_message': chat.user_message,
                'ai_response': chat.ai_response,
                'learning_style_context': chat.learning_style_context,
                'topic_category': chat.topic_category,
                'response_type': chat.response_type,
                'timestamp': chat.timestamp.isoformat()
            })
        
        return jsonify({
            'chat_history': chat_data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': chat_paginated.total,
                'pages': chat_paginated.pages,
                'has_next': chat_paginated.has_next,
                'has_prev': chat_paginated.has_prev
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/engagement', methods=['POST'])
@login_required
def track_engagement():
    """Track user engagement with content"""
    try:
        from ml_models.multimodal_fusion_engine import MultimodalFusionEngine, EngagementMetrics, LearningContext
        
        data = request.get_json()
        content_id = data.get('content_id')
        interaction_time = data.get('interaction_time', 0)
        completion_rate = data.get('completion_rate', 0)
        click_frequency = data.get('click_frequency', 0)
        scroll_velocity = data.get('scroll_velocity', 0)
        pause_duration = data.get('pause_duration', 0)
        context = data.get('context', 'general')
        
        if not content_id:
            return jsonify({'error': 'content_id is required'}), 400
        
        # Create engagement metrics
        engagement = EngagementMetrics(
            content_interaction_time=interaction_time,
            completion_rate=completion_rate,
            click_frequency=click_frequency,
            scroll_velocity=scroll_velocity,
            pause_duration=pause_duration,
            timestamp=datetime.now()
        )
        
        # Get learning context
        learning_context = LearningContext.GENERAL
        if context == 'mathematics':
            learning_context = LearningContext.MATHEMATICS
        elif context == 'languages':
            learning_context = LearningContext.LANGUAGES
        elif context == 'sciences':
            learning_context = LearningContext.SCIENCES
        elif context == 'programming':
            learning_context = LearningContext.PROGRAMMING
        
        # Update style weights using multimodal fusion
        fusion_engine = MultimodalFusionEngine()
        
        # Get content style
        content = ContentLibrary.query.get(content_id)
        content_style = 'general'
        if content and content.style_tags:
            style_tags = content.get_style_tags_list()
            if style_tags:
                content_style = style_tags[0]
        
        # Calculate performance score from completion rate and interaction time
        performance_score = (completion_rate + min(interaction_time / 300, 1)) / 2
        
        # Update style weights
        updated_weights = fusion_engine.update_style_weights(
            user_id=current_user.id,
            engagement_data=engagement,
            content_style=content_style,
            performance_score=performance_score,
            context=learning_context
        )
        
        # Update user's learning profile
        profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
        if profile:
            profile.visual_score = updated_weights.visual
            profile.auditory_score = updated_weights.auditory
            profile.kinesthetic_score = updated_weights.kinesthetic
            profile.dominant_style = max(
                ['visual', 'auditory', 'kinesthetic'], 
                key=lambda x: getattr(updated_weights, x)
            )
            db.session.commit()
        
        return jsonify({
            'message': 'Engagement tracked successfully',
            'updated_weights': {
                'visual': updated_weights.visual,
                'auditory': updated_weights.auditory,
                'kinesthetic': updated_weights.kinesthetic,
                'confidence': updated_weights.confidence
            },
            'dominant_style': profile.dominant_style if profile else None
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/engagement-analytics', methods=['GET'])
@login_required
def get_engagement_analytics():
    """Get user engagement analytics"""
    try:
        from ml_models.multimodal_fusion_engine import MultimodalFusionEngine, LearningContext
        
        fusion_engine = MultimodalFusionEngine()
        
        # Get style evolution patterns
        evolution_patterns = fusion_engine.detect_style_evolution_patterns(current_user.id)
        
        # Get recent progress for engagement analysis
        recent_progress = UserProgress.query.filter(
            UserProgress.user_id == current_user.id,
            UserProgress.timestamp >= datetime.now() - timedelta(days=30)
        ).all()
        
        # Calculate engagement metrics
        total_sessions = len(recent_progress)
        completed_sessions = len([p for p in recent_progress if p.completion_status == 'completed'])
        avg_score = sum(p.score for p in recent_progress if p.score) / len([p for p in recent_progress if p.score]) if recent_progress else 0
        avg_engagement = sum(p.engagement_rating for p in recent_progress if p.engagement_rating) / len([p for p in recent_progress if p.engagement_rating]) if recent_progress else 0
        
        return jsonify({
            'engagement_metrics': {
                'total_sessions': total_sessions,
                'completed_sessions': completed_sessions,
                'completion_rate': completed_sessions / total_sessions if total_sessions > 0 else 0,
                'average_score': avg_score,
                'average_engagement': avg_engagement
            },
            'style_evolution': evolution_patterns,
            'recommendations': fusion_engine.generate_hybrid_content_recommendation(
                fusion_engine._get_current_weights(current_user.id, LearningContext.GENERAL),
                []  # Empty content library for now
            )[:5]  # Top 5 recommendations
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/permissions', methods=['GET'])
@login_required
def get_permissions():
    """Get user's permission settings"""
    try:
        from app.models import UserPermissions
        
        permissions = UserPermissions.query.filter_by(user_id=current_user.id).first()
        
        if not permissions:
            # Create default permissions
            permissions = UserPermissions(
                user_id=current_user.id,
                camera_access=False,
                microphone_access=False,
                location_access=False,
                biometric_data=False,
                edu_sites_tracking=False,
                research_tracking=False,
                coding_tracking=False,
                video_tracking=False
            )
            db.session.add(permissions)
            db.session.commit()
        
        return jsonify({
            'permissions': {
                'camera': permissions.camera_access,
                'microphone': permissions.microphone_access,
                'location': permissions.location_access,
                'biometric': permissions.biometric_data,
                'eduSites': permissions.edu_sites_tracking,
                'research': permissions.research_tracking,
                'coding': permissions.coding_tracking,
                'video': permissions.video_tracking
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/permissions', methods=['POST'])
@login_required
def save_permissions():
    """Save user's permission settings"""
    try:
        from app.models import UserPermissions
        
        data = request.get_json()
        
        permissions = UserPermissions.query.filter_by(user_id=current_user.id).first()
        
        if not permissions:
            permissions = UserPermissions(user_id=current_user.id)
            db.session.add(permissions)
        
        # Update permissions
        permissions.camera_access = data.get('camera', False)
        permissions.microphone_access = data.get('microphone', False)
        permissions.location_access = data.get('location', False)
        permissions.biometric_data = data.get('biometric', False)
        permissions.edu_sites_tracking = data.get('eduSites', False)
        permissions.research_tracking = data.get('research', False)
        permissions.coding_tracking = data.get('coding', False)
        permissions.video_tracking = data.get('video', False)
        permissions.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'message': 'Permissions saved successfully',
            'permissions': {
                'camera': permissions.camera_access,
                'microphone': permissions.microphone_access,
                'location': permissions.location_access,
                'biometric': permissions.biometric_data,
                'eduSites': permissions.edu_sites_tracking,
                'research': permissions.research_tracking,
                'coding': permissions.coding_tracking,
                'video': permissions.video_tracking
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/learning-sites', methods=['POST'])
def track_learning_site():
    """Track user's learning site activity"""
    try:
        from app.models import LearningSiteActivity
        
        data = request.get_json()
        site_url = data.get('url')
        site_name = data.get('name')
        activity_type = data.get('activity_type', 'visit')
        time_spent = data.get('time_spent', 0)
        content_type = data.get('content_type', 'general')
        notes = data.get('notes', '')
        user_token = data.get('user_token')  # Token from browser extension
        
        if not site_url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Try to get user from session first (if logged in via web)
        user_id = None
        if current_user.is_authenticated:
            user_id = current_user.id
        elif user_token:
            # Try to validate token from browser extension
            # For now, we'll create a simple token system
            user_id = validate_extension_token(user_token)
        
        # If no valid user found, create a guest user
        if not user_id:
            user_id = get_or_create_guest_user()
        
        # Create learning site activity record
        activity = LearningSiteActivity(
            user_id=user_id,
            site_url=site_url,
            site_name=site_name,
            activity_type=activity_type,
            time_spent=time_spent,
            content_type=content_type,
            notes=notes
        )
        db.session.add(activity)
        db.session.commit()
        
        return jsonify({
            'message': 'Learning site activity tracked successfully',
            'activity_id': activity.id
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/extension-token', methods=['POST'])
@login_required
def generate_extension_token():
    """Generate a token for browser extension authentication"""
    try:
        # Generate a simple token (in production, use JWT)
        token = f"ext_{current_user.id}_{int(time.time())}"
        
        # Store token in session or database for validation
        session['extension_token'] = token
        
        return jsonify({
            'token': token,
            'user_id': current_user.id,
            'username': current_user.username,
            'expires_in': 3600  # 1 hour
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/learning-sites', methods=['GET'])
@login_required
def get_learning_sites():
    """Get user's learning site activities"""
    try:
        from app.models import LearningSiteActivity
        
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        user_token = request.args.get('user_token')
        
        # Determine user_id
        user_id = None
        if current_user.is_authenticated:
            user_id = current_user.id
            print(f"🔍 API: Authenticated user ID: {user_id}")
        elif user_token:
            user_id = validate_extension_token(user_token)
            print(f"🔍 API: Token-based user ID: {user_id}")
        
        # If no valid user, get guest user activities
        if not user_id:
            user_id = get_or_create_guest_user()
            print(f"🔍 API: Using guest user ID: {user_id}")
        
        if not user_id:
            return jsonify({'error': 'No valid user found'}), 400
        
        activities = LearningSiteActivity.query.filter_by(user_id=user_id).order_by(
            LearningSiteActivity.timestamp.desc()
        ).paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        activities_data = []
        for activity in activities.items:
            activities_data.append({
                'id': activity.id,
                'site_url': activity.site_url,
                'site_name': activity.site_name,
                'activity_type': activity.activity_type,
                'time_spent': activity.time_spent,
                'content_type': activity.content_type,
                'notes': activity.notes,
                'timestamp': activity.timestamp.isoformat()
            })
        
        return jsonify({
            'activities': activities_data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': activities.total,
                'pages': activities.pages,
                'has_next': activities.has_next,
                'has_prev': activities.has_prev
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/learning-sites/<int:activity_id>/content', methods=['GET'])
@login_required
def get_learning_site_content(activity_id):
    """Get detailed content information for a specific learning site activity"""
    try:
        from app.models import LearningSiteActivity
        
        # Get the activity
        activity = LearningSiteActivity.query.filter_by(
            id=activity_id, 
            user_id=current_user.id
        ).first()
        
        if not activity:
            return jsonify({'error': 'Activity not found'}), 404
        
        # Get additional content details (this could be enhanced with more detailed tracking)
        content_details = {
            'id': activity.id,
            'site_name': activity.site_name,
            'site_url': activity.site_url,
            'duration_minutes': activity.duration_minutes,
            'timestamp': activity.timestamp.isoformat(),
            'engagement_score': activity.engagement_score,
            'activity_type': activity.activity_type,
            'content_type': getattr(activity, 'content_type', 'Learning Material'),
            'subject': getattr(activity, 'subject', 'General'),
            'difficulty_level': getattr(activity, 'difficulty_level', 'Intermediate'),
            'progress_percentage': getattr(activity, 'progress_percentage', 0),
            'topics_covered': getattr(activity, 'topics_covered', 'Various'),
            'content_title': getattr(activity, 'content_title', None),
            'description': getattr(activity, 'description', None),
            'total_time': activity.duration_minutes,
            'completion_rate': getattr(activity, 'completion_rate', 0)
        }
        
        return jsonify(content_details)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/emotion-detection', methods=['POST'])
@login_required
def detect_emotions():
    """Detect emotions from camera/microphone data"""
    try:
        from app.models import UserPermissions, EmotionData
        from ml_models.emotion_ai.emotion_fusion_engine import EmotionFusionEngine
        
        # Check camera permission
        permissions = UserPermissions.query.filter_by(user_id=current_user.id).first()
        if not permissions or not permissions.camera_access:
            return jsonify({'error': 'Camera access not permitted'}), 403
        
        data = request.get_json()
        image_data = data.get('image_data')  # Base64 encoded image
        audio_data = data.get('audio_data')  # Audio data
        timestamp = data.get('timestamp', datetime.utcnow().isoformat())
        
        if not image_data and not audio_data:
            return jsonify({'error': 'Image or audio data required'}), 400
        
        # Initialize emotion fusion engine
        emotion_engine = EmotionFusionEngine()
        
        # Process emotion data
        emotion_result = emotion_engine.fuse_emotion_data(
            facial_data=image_data,
            voice_data=audio_data,
            context='learning'
        )
        
        # Save emotion data
        emotion_record = EmotionData(
            user_id=current_user.id,
            emotion_type=emotion_result.dominant_emotion.value,
            confidence_score=emotion_result.confidence,
            facial_emotion=emotion_result.facial_emotion.value if emotion_result.facial_emotion else None,
            voice_emotion=emotion_result.voice_emotion.value if emotion_result.voice_emotion else None,
            engagement_level=emotion_result.engagement_level,
            timestamp=datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        )
        db.session.add(emotion_record)
        db.session.commit()
        
        return jsonify({
            'emotion_detection': {
                'dominant_emotion': emotion_result.dominant_emotion.value,
                'confidence': emotion_result.confidence,
                'facial_emotion': emotion_result.facial_emotion.value if emotion_result.facial_emotion else None,
                'voice_emotion': emotion_result.voice_emotion.value if emotion_result.voice_emotion else None,
                'engagement_level': emotion_result.engagement_level,
                'recommendations': emotion_result.recommendations
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/admin/content-stats', methods=['GET'])
@login_required
def get_content_stats():
    """Get content statistics for admin dashboard"""
    try:
        if not current_user.is_admin():
            return jsonify({'error': 'Admin privileges required'}), 403
        
        from app.models import Content
        
        # Get content statistics
        total_content = Content.query.count()
        published_content = Content.query.filter_by(status='published').count()
        draft_content = Content.query.filter_by(status='draft').count()
        pending_content = Content.query.filter_by(status='pending').count()
        
        # Get total views and average rating
        total_views_result = db.session.query(db.func.sum(Content.views)).scalar() or 0
        avg_rating_result = db.session.query(db.func.avg(Content.rating)).scalar() or 0
        
        return jsonify({
            'total_content': total_content,
            'published_content': published_content,
            'draft_content': draft_content,
            'pending_content': pending_content,
            'total_views': total_views_result,
            'avg_rating': round(avg_rating_result, 1) if avg_rating_result else 0
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/admin/content', methods=['GET'])
@login_required
def get_admin_content():
    """Get content list for admin management"""
    try:
        if not current_user.is_admin():
            return jsonify({'error': 'Admin privileges required'}), 403
        
        from app.models import Content
        
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        search = request.args.get('search', '')
        content_type = request.args.get('type', '')
        status = request.args.get('status', '')
        learning_style = request.args.get('learning_style', '')
        
        # Build query
        query = Content.query
        
        if search:
            query = query.filter(
                or_(
                    Content.title.contains(search),
                    Content.content.contains(search),
                    Content.tags.contains(search)
                )
            )
        
        if content_type:
            query = query.filter_by(content_type=content_type)
        
        if status:
            query = query.filter_by(status=status)
        
        if learning_style:
            query = query.filter_by(learning_style=learning_style)
        
        # Paginate results
        content_paginated = query.order_by(Content.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        content_data = []
        for content in content_paginated.items:
            content_data.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'learning_style': content.learning_style,
                'difficulty_level': content.difficulty_level,
                'status': content.status,
                'views': content.views,
                'rating': content.rating,
                'tags': content.tags,
                'created_at': content.created_at.isoformat(),
                'updated_at': content.updated_at.isoformat()
            })
        
        return jsonify({
            'content': content_data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': content_paginated.total,
                'pages': content_paginated.pages,
                'has_next': content_paginated.has_next,
                'has_prev': content_paginated.has_prev
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/admin/content', methods=['POST'])
@login_required
def create_admin_content():
    """Create new content"""
    try:
        if not current_user.is_admin():
            return jsonify({'error': 'Admin privileges required'}), 403
        
        from app.models import Content
        
        data = request.get_json()
        
        # Create new content
        content = Content(
            title=data.get('title'),
            content_type=data.get('content_type'),
            content=data.get('content', ''),
            learning_style=data.get('learning_style'),
            difficulty_level=data.get('difficulty_level', 'beginner'),
            tags=data.get('tags', ''),
            author_id=current_user.id,
            status=data.get('status', 'draft'),
            views=0,
            rating=0.0
        )
        
        db.session.add(content)
        db.session.commit()
        
        return jsonify({
            'message': 'Content created successfully',
            'content_id': content.id
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/admin/content/<int:content_id>', methods=['GET'])
@login_required
def get_admin_content_detail(content_id):
    """Get specific content details"""
    try:
        if not current_user.is_admin():
            return jsonify({'error': 'Admin privileges required'}), 403
        
        from app.models import Content
        
        content = Content.query.get_or_404(content_id)
        
        return jsonify({
            'id': content.id,
            'title': content.title,
            'content_type': content.content_type,
            'content': content.content,
            'learning_style': content.learning_style,
            'difficulty_level': content.difficulty_level,
            'tags': content.tags,
            'status': content.status,
            'views': content.views,
            'rating': content.rating,
            'author_id': content.author_id,
            'created_at': content.created_at.isoformat(),
            'updated_at': content.updated_at.isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/admin/content/<int:content_id>', methods=['PUT'])
@login_required
def update_admin_content(content_id):
    """Update content"""
    try:
        if not current_user.is_admin():
            return jsonify({'error': 'Admin privileges required'}), 403
        
        from app.models import Content
        
        content = Content.query.get_or_404(content_id)
        data = request.get_json()
        
        # Update content fields
        content.title = data.get('title', content.title)
        content.content_type = data.get('content_type', content.content_type)
        content.content = data.get('content', content.content)
        content.learning_style = data.get('learning_style', content.learning_style)
        content.difficulty_level = data.get('difficulty_level', content.difficulty_level)
        content.tags = data.get('tags', content.tags)
        content.status = data.get('status', content.status)
        content.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'message': 'Content updated successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/admin/content/<int:content_id>', methods=['DELETE'])
@login_required
def delete_admin_content(content_id):
    """Delete content and remove from ContentLibrary"""
    try:
        if not current_user.is_admin():
            return jsonify({'error': 'Admin privileges required'}), 403
        
        from app.models import Content
        from app.utils.content_sync import ContentSyncManager
        
        content = Content.query.get_or_404(content_id)
        
        # Remove from ContentLibrary first
        sync_result = ContentSyncManager.remove_from_content_library(content_id)
        
        # Delete from Content table
        db.session.delete(content)
        db.session.commit()
        
        return jsonify({
            'message': 'Content deleted successfully',
            'sync_result': sync_result
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/admin/content/<int:content_id>/publish', methods=['POST'])
@login_required
def publish_admin_content(content_id):
    """Publish content and sync to ContentLibrary"""
    try:
        if not current_user.is_admin():
            return jsonify({'error': 'Admin privileges required'}), 403
        
        from app.models import Content
        from app.utils.content_sync import ContentSyncManager
        
        content = Content.query.get_or_404(content_id)
        content.status = 'published'
        content.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        # Sync to ContentLibrary
        sync_result = ContentSyncManager.sync_single_content(content_id)
        
        return jsonify({
            'message': 'Content published successfully',
            'sync_result': sync_result
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/admin/content/<int:content_id>/unpublish', methods=['POST'])
@login_required
def unpublish_admin_content(content_id):
    """Unpublish content and remove from ContentLibrary"""
    try:
        if not current_user.is_admin():
            return jsonify({'error': 'Admin privileges required'}), 403
        
        from app.models import Content
        from app.utils.content_sync import ContentSyncManager
        
        content = Content.query.get_or_404(content_id)
        content.status = 'draft'
        content.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        # Remove from ContentLibrary
        sync_result = ContentSyncManager.remove_from_content_library(content_id)
        
        return jsonify({
            'message': 'Content unpublished successfully',
            'sync_result': sync_result
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/admin/content-sync/status', methods=['GET'])
@login_required
def get_content_sync_status():
    """Get content synchronization status"""
    try:
        if not current_user.is_admin():
            return jsonify({'error': 'Admin privileges required'}), 403
        
        from app.utils.content_sync import ContentSyncManager
        status = ContentSyncManager.get_sync_status()
        return jsonify(status), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/admin/content-sync/sync-all', methods=['POST'])
@login_required
def sync_all_content():
    """Sync all published content to ContentLibrary"""
    try:
        if not current_user.is_admin():
            return jsonify({'error': 'Admin privileges required'}), 403
        
        from app.utils.content_sync import ContentSyncManager
        result = ContentSyncManager.sync_published_content()
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/admin/content-sync/force-sync', methods=['POST'])
@login_required
def force_sync_all_content():
    """Force sync all published content, updating existing entries"""
    try:
        if not current_user.is_admin():
            return jsonify({'error': 'Admin privileges required'}), 403
        
        from app.utils.content_sync import ContentSyncManager
        result = ContentSyncManager.force_sync_all()
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/admin/content-sync/sync/<int:content_id>', methods=['POST'])
@login_required
def sync_single_content(content_id):
    """Sync a single content item to ContentLibrary"""
    try:
        if not current_user.is_admin():
            return jsonify({'error': 'Admin privileges required'}), 403
        
        from app.utils.content_sync import ContentSyncManager
        result = ContentSyncManager.sync_single_content(content_id)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/admin/content-sync/remove/<int:content_id>', methods=['POST'])
@login_required
def remove_content_from_library(content_id):
    """Remove content from ContentLibrary"""
    try:
        if not current_user.is_admin():
            return jsonify({'error': 'Admin privileges required'}), 403
        
        from app.utils.content_sync import ContentSyncManager
        result = ContentSyncManager.remove_from_content_library(content_id)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/profile', methods=['GET'])
@login_required
def get_user_profile():
    """Get user's learning profile and statistics"""
    try:
        profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
        
        if not profile:
            return jsonify({'error': 'No learning profile found'}), 404
        
        # Get user statistics
        total_content = UserProgress.query.filter_by(user_id=current_user.id).count()
        completed_content = UserProgress.query.filter_by(
            user_id=current_user.id,
            completion_status='completed'
        ).count()
        
        total_time = db.session.query(
            db.func.sum(UserProgress.time_spent)
        ).filter_by(user_id=current_user.id).scalar() or 0
        
        return jsonify({
            'profile': {
                'visual_score': profile.visual_score,
                'auditory_score': profile.auditory_score,
                'kinesthetic_score': profile.kinesthetic_score,
                'dominant_style': profile.dominant_style,
                'last_updated': profile.last_updated.isoformat(),
                'style_breakdown': profile.get_style_breakdown()
            },
            'statistics': {
                'total_content_accessed': total_content,
                'completed_content': completed_content,
                'total_time_spent': total_time,
                'completion_rate': (completed_content / total_content * 100) if total_content > 0 else 0
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Register blueprints would be done in main app.py