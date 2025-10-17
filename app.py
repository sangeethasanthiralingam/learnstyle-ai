"""
LearnStyle AI - Main Flask Application
An Intelligent Adaptive Learning System with Personalized Content Delivery
"""

from flask import render_template
from flask_login import login_required, current_user
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import app factory and extensions
from app import create_app, db, login_manager

# Create Flask app
app = create_app()

# Import models after app is created
from app.models import User, LearningProfile

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with personalized content"""
    user_profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
    return render_template('dashboard.html', user_profile=user_profile)

@app.route('/quiz')
@login_required
def quiz():
    """Learning style assessment quiz"""
    return render_template('quiz.html')

@app.route('/chat')
@login_required
def chat():
    """AI tutor chat interface"""
    return render_template('chat.html')

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('errors/500.html'), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)