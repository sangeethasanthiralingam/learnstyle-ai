"""
Enhanced API endpoints for LearnStyle AI
Additional backend features to support the enhanced frontend
"""

from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from app.models import db, User, LearningProfile, ContentLibrary, UserProgress
from sqlalchemy import func, desc
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Create blueprint for enhanced API
enhanced_api = Blueprint('enhanced_api', __name__)

@enhanced_api.route('/api/dashboard/statistics', methods=['GET'])
@login_required
def get_dashboard_statistics():
    """Get comprehensive dashboard statistics"""
    try:
        # Get basic user stats
        total_users = User.query.count()
        active_users = User.query.filter(
            User.last_login >= datetime.now() - timedelta(days=30)
        ).count()
        
        # Get learning sessions
        learning_sessions = UserProgress.query.filter(
            UserProgress.timestamp >= datetime.now() - timedelta(days=30)
        ).count()
        
        # Get AI predictions (learning style assessments)
        ai_predictions = LearningProfile.query.filter(
            LearningProfile.dominant_style.isnot(None)
        ).count()
        
        # Get user's personal stats
        user_profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
        user_stats = {
            'content_completed': UserProgress.query.filter_by(
                user_id=current_user.id, 
                completion_status='completed'
            ).count(),
            'hours_learned': db.session.query(
                func.sum(UserProgress.time_spent)
            ).filter_by(user_id=current_user.id).scalar() or 0,
            'average_score': db.session.query(
                func.avg(UserProgress.score)
            ).filter_by(user_id=current_user.id).scalar() or 0
        }
        
        # Convert hours from seconds
        user_stats['hours_learned'] = round(user_stats['hours_learned'] / 3600, 1)
        user_stats['average_score'] = round(user_stats['average_score'], 1)
        
        return jsonify({
            'total_users': total_users,
            'active_users': active_users,
            'learning_sessions': learning_sessions,
            'ai_predictions': ai_predictions,
            'user_stats': user_stats,
            'user_profile': {
                'dominant_style': user_profile.dominant_style if user_profile else None,
                'visual_score': user_profile.visual_score if user_profile else 0,
                'auditory_score': user_profile.auditory_score if user_profile else 0,
                'kinesthetic_score': user_profile.kinesthetic_score if user_profile else 0
            } if user_profile else None
        })
        
    except Exception as e:
        logger.error(f"Error getting dashboard statistics: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/api/content', methods=['GET'])
@login_required
def get_content_recommendations():
    """Get personalized content recommendations"""
    try:
        # Get user's learning profile
        profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
        
        if not profile or not profile.dominant_style:
            # Return general content if no profile
            content = ContentLibrary.query.limit(6).all()
        else:
            # Get content based on learning style
            content = ContentLibrary.query.filter(
                ContentLibrary.style_tags.contains(profile.dominant_style)
            ).limit(6).all()
            
            # If not enough content, fill with general content
            if len(content) < 6:
                general_content = ContentLibrary.query.filter(
                    ~ContentLibrary.id.in_([c.id for c in content])
                ).limit(6 - len(content)).all()
                content.extend(general_content)
        
        content_list = []
        for item in content:
            content_list.append({
                'id': item.id,
                'title': item.title,
                'description': item.description,
                'content_type': item.content_type,
                'style_tags': item.style_tags,
                'difficulty_level': item.difficulty_level,
                'url': item.url_path or '#',
                'created_at': item.created_at.isoformat()
            })
        
        return jsonify({
            'content': content_list,
            'learning_style': profile.dominant_style if profile else 'general'
        })
        
    except Exception as e:
        logger.error(f"Error getting content recommendations: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/api/save-content', methods=['POST'])
@login_required
def save_content():
    """Save content to user's library"""
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        
        if not content_id:
            return jsonify({'error': 'Content ID required'}), 400
        
        # Check if content exists
        content = ContentLibrary.query.get(content_id)
        if not content:
            return jsonify({'error': 'Content not found'}), 404
        
        # For now, just return success (in a real app, you'd save to user's library)
        return jsonify({
            'success': True,
            'message': 'Content saved to your library'
        })
        
    except Exception as e:
        logger.error(f"Error saving content: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/api/save-quiz-progress', methods=['POST'])
@login_required
def save_quiz_progress():
    """Save quiz progress (auto-save)"""
    try:
        data = request.get_json()
        answers = data.get('answers', {})
        
        # Store in session or temporary storage
        # In a real app, you might save to a temporary table
        return jsonify({
            'success': True,
            'message': 'Progress saved'
        })
        
    except Exception as e:
        logger.error(f"Error saving quiz progress: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/api/learning-analytics', methods=['GET'])
@login_required
def get_learning_analytics():
    """Get detailed learning analytics for charts"""
    try:
        # Get user's progress over time
        progress_data = []
        for i in range(6):  # Last 6 weeks
            week_start = datetime.now() - timedelta(weeks=5-i, days=datetime.now().weekday())
            week_end = week_start + timedelta(days=6)
            
            week_sessions = UserProgress.query.filter(
                UserProgress.user_id == current_user.id,
                UserProgress.timestamp >= week_start,
                UserProgress.timestamp <= week_end
            ).count()
            
            progress_data.append(week_sessions)
        
        # Get learning style distribution
        style_distribution = db.session.query(
            LearningProfile.dominant_style,
            func.count(LearningProfile.id)
        ).group_by(LearningProfile.dominant_style).all()
        
        style_data = {}
        for style, count in style_distribution:
            style_data[style or 'Unknown'] = count
        
        # Get content engagement by type
        content_engagement = db.session.query(
            ContentLibrary.content_type,
            func.count(ContentLibrary.id)
        ).group_by(ContentLibrary.content_type).all()
        
        engagement_data = {}
        for content_type, count in content_engagement:
            engagement_data[content_type or 'Unknown'] = count
        
        return jsonify({
            'progress_data': progress_data,
            'style_distribution': style_data,
            'content_engagement': engagement_data
        })
        
    except Exception as e:
        logger.error(f"Error getting learning analytics: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/api/notifications', methods=['GET'])
@login_required
def get_notifications():
    """Get user notifications"""
    try:
        # Get recent activities and notifications
        recent_progress = UserProgress.query.filter_by(
            user_id=current_user.id
        ).order_by(desc(UserProgress.timestamp)).limit(5).all()
        
        notifications = []
        for progress in recent_progress:
            notifications.append({
                'type': 'progress',
                'message': f'Completed: {progress.content.title if progress.content else "Content"}',
                'timestamp': progress.timestamp.isoformat(),
                'icon': 'fas fa-check-circle'
            })
        
        return jsonify({
            'notifications': notifications,
            'unread_count': len(notifications)
        })
        
    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/api/achievements', methods=['GET'])
@login_required
def get_achievements():
    """Get user achievements and badges"""
    try:
        # Calculate achievements based on user progress
        user_progress = UserProgress.query.filter_by(user_id=current_user.id).all()
        
        achievements = []
        
        # Content completion achievements
        completed_count = len([p for p in user_progress if p.completion_status == 'completed'])
        if completed_count >= 1:
            achievements.append({
                'id': 'first_completion',
                'name': 'First Steps',
                'description': 'Completed your first learning content',
                'icon': 'fas fa-star',
                'unlocked': True
            })
        
        if completed_count >= 5:
            achievements.append({
                'id': 'regular_learner',
                'name': 'Regular Learner',
                'description': 'Completed 5 learning sessions',
                'icon': 'fas fa-trophy',
                'unlocked': True
            })
        
        if completed_count >= 10:
            achievements.append({
                'id': 'dedicated_learner',
                'name': 'Dedicated Learner',
                'description': 'Completed 10 learning sessions',
                'icon': 'fas fa-medal',
                'unlocked': True
            })
        
        # Time-based achievements
        total_time = sum(p.time_spent or 0 for p in user_progress) / 3600  # Convert to hours
        if total_time >= 10:
            achievements.append({
                'id': 'time_invested',
                'name': 'Time Investor',
                'description': 'Spent 10+ hours learning',
                'icon': 'fas fa-clock',
                'unlocked': True
            })
        
        return jsonify({
            'achievements': achievements,
            'total_achievements': len(achievements),
            'completed_count': completed_count,
            'total_time': round(total_time, 1)
        })
        
    except Exception as e:
        logger.error(f"Error getting achievements: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_api.route('/api/learning-insights', methods=['GET'])
@login_required
def get_learning_insights():
    """Get personalized learning insights"""
    try:
        profile = LearningProfile.query.filter_by(user_id=current_user.id).first()
        if not profile:
            return jsonify({'error': 'No learning profile found'}), 404
        
        # Get learning patterns
        recent_progress = UserProgress.query.filter_by(
            user_id=current_user.id
        ).order_by(desc(UserProgress.timestamp)).limit(10).all()
        
        # Calculate insights
        insights = []
        
        # Learning style insights
        if profile.dominant_style:
            insights.append({
                'type': 'learning_style',
                'title': f'Your {profile.dominant_style.title()} Learning Style',
                'description': f'You learn best through {profile.dominant_style} methods. Try to focus on content that matches this style.',
                'icon': 'fas fa-brain',
                'priority': 'high'
            })
        
        # Progress insights
        if recent_progress:
            avg_score = sum(p.score or 0 for p in recent_progress) / len(recent_progress)
            if avg_score > 80:
                insights.append({
                    'type': 'performance',
                    'title': 'Excellent Performance!',
                    'description': f'Your average score is {avg_score:.1f}%. Keep up the great work!',
                    'icon': 'fas fa-star',
                    'priority': 'medium'
                })
            elif avg_score < 60:
                insights.append({
                    'type': 'performance',
                    'title': 'Room for Improvement',
                    'description': f'Your average score is {avg_score:.1f}%. Consider reviewing previous content or trying different learning methods.',
                    'icon': 'fas fa-lightbulb',
                    'priority': 'high'
                })
        
        # Time insights
        total_time = sum(p.time_spent or 0 for p in recent_progress) / 3600
        if total_time > 5:
            insights.append({
                'type': 'time',
                'title': 'Consistent Learner',
                'description': f'You\'ve spent {total_time:.1f} hours learning recently. Consistency is key to success!',
                'icon': 'fas fa-clock',
                'priority': 'low'
            })
        
        return jsonify({
            'insights': insights,
            'profile': {
                'dominant_style': profile.dominant_style,
                'visual_score': profile.visual_score,
                'auditory_score': profile.auditory_score,
                'kinesthetic_score': profile.kinesthetic_score
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting learning insights: {e}")
        return jsonify({'error': str(e)}), 500
