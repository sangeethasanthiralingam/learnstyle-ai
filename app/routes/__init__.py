"""
LearnStyle AI - API Routes
RESTful API endpoints for all application functionality
"""

from flask import Blueprint, request, jsonify, session, render_template, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import sys
import os

# Add the parent directory to sys.path to import ml_models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.models import db, User, LearningProfile, QuizResponse, ContentLibrary, UserProgress, ChatHistory
from ml_models.learning_style_predictor import LearningStylePredictor

# Initialize ML predictor
ml_predictor = LearningStylePredictor()

# Create blueprints
api_bp = Blueprint('api', __name__, url_prefix='/api')
auth_bp = Blueprint('auth', __name__)

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

@api_bp.route('/chat', methods=['POST'])
@login_required
def ai_chat():
    """AI tutor chat with style-aware responses"""
    try:
        data = request.get_json()
        user_message = data.get('message')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get user's learning style for context
        profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
        learning_style = profile.dominant_style if profile else 'general'
        
        # Generate AI response based on learning style
        # This is a simplified implementation - in production you'd use OpenAI API or similar
        style_prompts = {
            'visual': "I'll explain this using visual concepts and suggest diagrams where helpful. ",
            'auditory': "Let me explain this in a conversational way with clear verbal descriptions. ",
            'kinesthetic': "I'll focus on practical examples and hands-on applications. "
        }
        
        style_context = style_prompts.get(learning_style, "")
        
        # Simple rule-based response (replace with actual AI API in production)
        if "help" in user_message.lower():
            ai_response = f"{style_context}I'm here to help you learn! What specific topic would you like to explore?"
        elif "explain" in user_message.lower():
            ai_response = f"{style_context}I'd be happy to explain that concept. Can you tell me more about what you'd like to understand?"
        else:
            ai_response = f"{style_context}That's an interesting question! Let me help you explore that topic."
        
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
            'learning_style_context': learning_style
        }), 200
        
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