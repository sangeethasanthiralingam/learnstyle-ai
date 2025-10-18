"""
Database Models for LearnStyle AI
"""

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

from app import db

class User(UserMixin, db.Model):
    """User model for authentication and profile management"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='user')  # 'user', 'admin', 'moderator'
    is_active = db.Column(db.Boolean, default=True)
    last_login = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    learning_profile = db.relationship('LearningProfile', backref='user', uselist=False)
    quiz_responses = db.relationship('QuizResponse', backref='user', lazy=True)
    progress_records = db.relationship('UserProgress', backref='user', lazy=True)
    chat_history = db.relationship('ChatHistory', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        """Check if user is an admin"""
        return self.role == 'admin'
    
    def is_moderator(self):
        """Check if user is a moderator"""
        return self.role in ['admin', 'moderator']
    
    def can_manage_users(self):
        """Check if user can manage other users"""
        return self.role == 'admin'
    
    def can_view_analytics(self):
        """Check if user can view system analytics"""
        return self.role in ['admin', 'moderator']
    
    def can_manage_content(self):
        """Check if user can manage content"""
        return self.role in ['admin', 'moderator']
    
    def __repr__(self):
        return f'<User {self.username} ({self.role})>'

class LearningProfile(db.Model):
    """Learning profile with style preferences and scores"""
    __tablename__ = 'learning_profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    visual_score = db.Column(db.Float, default=0.0)
    auditory_score = db.Column(db.Float, default=0.0)
    kinesthetic_score = db.Column(db.Float, default=0.0)
    dominant_style = db.Column(db.String(20))
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    
    def get_style_breakdown(self):
        """Return style breakdown as percentages"""
        total = self.visual_score + self.auditory_score + self.kinesthetic_score
        if total == 0:
            return {'visual': 0, 'auditory': 0, 'kinesthetic': 0}
        return {
            'visual': round((self.visual_score / total) * 100, 1),
            'auditory': round((self.auditory_score / total) * 100, 1),
            'kinesthetic': round((self.kinesthetic_score / total) * 100, 1)
        }
    
    def __repr__(self):
        return f'<LearningProfile User:{self.user_id} Style:{self.dominant_style}>'

class QuizResponse(db.Model):
    """Quiz responses for learning style assessment"""
    __tablename__ = 'quiz_responses'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    question_1 = db.Column(db.Integer)
    question_2 = db.Column(db.Integer)
    question_3 = db.Column(db.Integer)
    question_4 = db.Column(db.Integer)
    question_5 = db.Column(db.Integer)
    question_6 = db.Column(db.Integer)
    question_7 = db.Column(db.Integer)
    question_8 = db.Column(db.Integer)
    question_9 = db.Column(db.Integer)
    question_10 = db.Column(db.Integer)
    question_11 = db.Column(db.Integer)
    question_12 = db.Column(db.Integer)
    question_13 = db.Column(db.Integer)
    question_14 = db.Column(db.Integer)
    question_15 = db.Column(db.Integer)
    submission_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    def get_responses_list(self):
        """Return quiz responses as a list"""
        return [
            self.question_1, self.question_2, self.question_3, self.question_4, self.question_5,
            self.question_6, self.question_7, self.question_8, self.question_9, self.question_10,
            self.question_11, self.question_12, self.question_13, self.question_14, self.question_15
        ]
    
    def __repr__(self):
        return f'<QuizResponse User:{self.user_id} Date:{self.submission_date}>'

class ContentLibrary(db.Model):
    """Content library for educational materials"""
    __tablename__ = 'content_library'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    content_type = db.Column(db.String(20), nullable=False)  # 'video', 'audio', 'interactive', 'text'
    style_tags = db.Column(db.String(100))  # 'visual', 'auditory', 'kinesthetic'
    difficulty_level = db.Column(db.String(20))  # 'beginner', 'intermediate', 'advanced'
    url_path = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    progress_records = db.relationship('UserProgress', backref='content', lazy=True)
    
    def get_style_tags_list(self):
        """Return style tags as a list"""
        if self.style_tags:
            return [tag.strip() for tag in self.style_tags.split(',')]
        return []
    
    def __repr__(self):
        return f'<Content {self.title} - {self.content_type}>'

class UserProgress(db.Model):
    """User progress tracking for content consumption"""
    __tablename__ = 'user_progress'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content_library.id'), nullable=False)
    completion_status = db.Column(db.String(20), default='started')  # 'started', 'in_progress', 'completed'
    time_spent = db.Column(db.Integer, default=0)  # in seconds
    score = db.Column(db.Float)
    engagement_rating = db.Column(db.Integer)  # 1-5 scale
    completed_at = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<Progress User:{self.user_id} Content:{self.content_id} Status:{self.completion_status}>'

class ChatHistory(db.Model):
    """AI tutor chat history"""
    __tablename__ = 'chat_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    ai_response = db.Column(db.Text, nullable=False)
    learning_style_context = db.Column(db.String(20))  # dominant learning style at time of chat
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Chat User:{self.user_id} Time:{self.timestamp}>'