"""
LearnStyle AI - Content Management System
Personalized content delivery based on learning styles
"""

from typing import Dict, List, Optional
from app.models import ContentLibrary, UserProgress, LearningProfile, db
from sqlalchemy import and_, or_

class ContentManager:
    """
    Manages personalized content delivery based on learning styles
    """
    
    def __init__(self):
        self.style_weights = {
            'visual': ['visual', 'diagram', 'infographic', 'video'],
            'auditory': ['auditory', 'audio', 'podcast', 'lecture'],
            'kinesthetic': ['kinesthetic', 'interactive', 'hands-on', 'practice']
        }
    
    def get_personalized_content(self, user_id: int, style_weights: Dict[str, float], 
                               progress_data: Optional[List] = None, limit: int = 10) -> List[Dict]:
        """
        Algorithm for personalized content recommendation
        
        Args:
            user_id: User ID for personalization
            style_weights: Dictionary with visual, auditory, kinesthetic scores
            progress_data: User's past content consumption data
            limit: Maximum number of content items to return
            
        Returns:
            List of recommended content items
        """
        # Get user's learning profile
        profile = LearningProfile.query.filter_by(user_id=user_id).first()
        
        if not profile or not profile.dominant_style:
            # Return general content if no profile
            return self._get_general_content(limit)
        
        # Get content based on style preferences
        recommended_content = []
        
        # 1. Filter content by user's style preferences
        primary_style = profile.dominant_style
        primary_content = self._get_content_by_style(primary_style, limit // 2)
        recommended_content.extend(primary_content)
        
        # 2. Consider past engagement and performance
        if progress_data:
            high_engagement_types = self._analyze_engagement_patterns(user_id)
            for content_type in high_engagement_types:
                additional_content = self._get_content_by_type(content_type, limit // 4)
                recommended_content.extend(additional_content)
        
        # 3. Balance content types based on style blend
        style_breakdown = profile.get_style_breakdown()
        for style, percentage in style_breakdown.items():
            if style != primary_style and percentage > 20:  # Secondary styles with >20%
                secondary_content = self._get_content_by_style(style, max(1, int(limit * percentage / 100)))
                recommended_content.extend(secondary_content)
        
        # 4. Adjust difficulty based on progress
        user_progress = self._get_user_difficulty_level(user_id)
        recommended_content = self._filter_by_difficulty(recommended_content, user_progress)
        
        # 5. Include new/unseen content for diversity
        unseen_content = self._get_unseen_content(user_id, limit // 4)
        recommended_content.extend(unseen_content)
        
        # Remove duplicates and limit results
        seen_ids = set()
        unique_content = []
        for content in recommended_content:
            if content['id'] not in seen_ids:
                seen_ids.add(content['id'])
                unique_content.append(content)
        
        return unique_content[:limit]
    
    def _get_content_by_style(self, style: str, limit: int) -> List[Dict]:
        """Get content matching a specific learning style"""
        style_filter = f'%{style}%'
        content_items = ContentLibrary.query.filter(
            ContentLibrary.style_tags.like(style_filter)
        ).limit(limit).all()
        
        return [self._content_to_dict(item) for item in content_items]
    
    def _get_content_by_type(self, content_type: str, limit: int) -> List[Dict]:
        """Get content by content type (video, audio, etc.)"""
        content_items = ContentLibrary.query.filter_by(
            content_type=content_type
        ).limit(limit).all()
        
        return [self._content_to_dict(item) for item in content_items]
    
    def _get_general_content(self, limit: int) -> List[Dict]:
        """Get general content when no user profile exists"""
        content_items = ContentLibrary.query.limit(limit).all()
        return [self._content_to_dict(item) for item in content_items]
    
    def _analyze_engagement_patterns(self, user_id: int) -> List[str]:
        """Analyze user engagement patterns to find preferred content types"""
        # Get user progress with high engagement ratings
        high_engagement = UserProgress.query.join(ContentLibrary).filter(
            and_(
                UserProgress.user_id == user_id,
                UserProgress.engagement_rating >= 4
            )
        ).all()
        
        # Count content types with high engagement
        type_counts = {}
        for progress in high_engagement:
            content_type = progress.content.content_type
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        
        # Return content types sorted by engagement
        return sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)
    
    def _get_user_difficulty_level(self, user_id: int) -> str:
        """Determine user's appropriate difficulty level based on progress"""
        completed_progress = UserProgress.query.filter(
            and_(
                UserProgress.user_id == user_id,
                UserProgress.completion_status == 'completed'
            )
        ).all()
        
        if not completed_progress:
            return 'beginner'
        
        # Analyze completion rates and scores
        total_completed = len(completed_progress)
        high_scores = sum(1 for p in completed_progress if p.score and p.score >= 80)
        
        if total_completed >= 10 and high_scores / total_completed >= 0.8:
            return 'advanced'
        elif total_completed >= 5 and high_scores / total_completed >= 0.6:
            return 'intermediate'
        else:
            return 'beginner'
    
    def _filter_by_difficulty(self, content_list: List[Dict], user_level: str) -> List[Dict]:
        """Filter content based on user's difficulty level"""
        # Allow content at user's level and one level below/above
        level_hierarchy = ['beginner', 'intermediate', 'advanced']
        user_index = level_hierarchy.index(user_level) if user_level in level_hierarchy else 0
        
        allowed_levels = set()
        allowed_levels.add(user_level)
        if user_index > 0:
            allowed_levels.add(level_hierarchy[user_index - 1])
        if user_index < len(level_hierarchy) - 1:
            allowed_levels.add(level_hierarchy[user_index + 1])
        
        return [content for content in content_list 
                if not content.get('difficulty_level') or content['difficulty_level'] in allowed_levels]
    
    def _get_unseen_content(self, user_id: int, limit: int) -> List[Dict]:
        """Get content the user hasn't seen before"""
        # Get IDs of content user has already accessed
        seen_content_ids = db.session.query(UserProgress.content_id).filter_by(user_id=user_id).subquery()
        
        # Get content not in seen list
        unseen_content = ContentLibrary.query.filter(
            ~ContentLibrary.id.in_(seen_content_ids)
        ).limit(limit).all()
        
        return [self._content_to_dict(item) for item in unseen_content]
    
    def _content_to_dict(self, content_item: ContentLibrary) -> Dict:
        """Convert ContentLibrary object to dictionary"""
        return {
            'id': content_item.id,
            'title': content_item.title,
            'description': content_item.description,
            'content_type': content_item.content_type,
            'style_tags': content_item.get_style_tags_list(),
            'difficulty_level': content_item.difficulty_level,
            'url_path': content_item.url_path,
            'created_at': content_item.created_at.isoformat()
        }
    
    def add_content(self, title: str, description: str, content_type: str, 
                   style_tags: List[str], difficulty_level: str, url_path: str) -> ContentLibrary:
        """Add new content to the library"""
        content = ContentLibrary(
            title=title,
            description=description,
            content_type=content_type,
            style_tags=','.join(style_tags),
            difficulty_level=difficulty_level,
            url_path=url_path
        )
        
        db.session.add(content)
        db.session.commit()
        
        return content
    
    def update_content_effectiveness(self, content_id: int, user_feedback: Dict):
        """Update content effectiveness based on user feedback"""
        # This would be used to improve content recommendations over time
        # Could implement A/B testing, rating systems, etc.
        pass
    
    def get_content_analytics(self, content_id: int) -> Dict:
        """Get analytics for specific content"""
        content = ContentLibrary.query.get(content_id)
        if not content:
            return {}
        
        progress_records = UserProgress.query.filter_by(content_id=content_id).all()
        
        total_users = len(progress_records)
        completed_users = len([p for p in progress_records if p.completion_status == 'completed'])
        avg_time = sum(p.time_spent for p in progress_records) / total_users if total_users > 0 else 0
        avg_rating = sum(p.engagement_rating for p in progress_records if p.engagement_rating) / total_users if total_users > 0 else 0
        avg_score = sum(p.score for p in progress_records if p.score) / total_users if total_users > 0 else 0
        
        return {
            'content_id': content_id,
            'title': content.title,
            'total_users': total_users,
            'completion_rate': (completed_users / total_users * 100) if total_users > 0 else 0,
            'average_time_spent': avg_time,
            'average_engagement_rating': avg_rating,
            'average_score': avg_score
        }

# Content seeding functions
def seed_sample_content():
    """Seed the database with sample content for different learning styles"""
    sample_content = [
        # Visual Content
        {
            'title': 'Introduction to Data Structures with Diagrams',
            'description': 'Learn data structures through visual diagrams and flowcharts.',
            'content_type': 'video',
            'style_tags': ['visual', 'beginner'],
            'difficulty_level': 'beginner',
            'url_path': '/content/data-structures-visual'
        },
        {
            'title': 'Mind Maps for Algorithm Design',
            'description': 'Use mind mapping techniques to understand algorithm design patterns.',
            'content_type': 'interactive',
            'style_tags': ['visual', 'intermediate'],
            'difficulty_level': 'intermediate',
            'url_path': '/content/algorithm-mindmaps'
        },
        {
            'title': 'Infographic: Machine Learning Overview',
            'description': 'Visual overview of machine learning concepts and applications.',
            'content_type': 'text',
            'style_tags': ['visual', 'beginner'],
            'difficulty_level': 'beginner',
            'url_path': '/content/ml-infographic'
        },
        
        # Auditory Content
        {
            'title': 'Programming Concepts Podcast Series',
            'description': 'Learn programming through engaging audio discussions.',
            'content_type': 'audio',
            'style_tags': ['auditory', 'beginner'],
            'difficulty_level': 'beginner',
            'url_path': '/content/programming-podcast'
        },
        {
            'title': 'Database Design Audio Lectures',
            'description': 'Comprehensive audio lectures on database design principles.',
            'content_type': 'audio',
            'style_tags': ['auditory', 'intermediate'],
            'difficulty_level': 'intermediate',
            'url_path': '/content/database-lectures'
        },
        {
            'title': 'Tech Talk: AI and Ethics',
            'description': 'Thought-provoking discussion on AI ethics and implications.',
            'content_type': 'audio',
            'style_tags': ['auditory', 'advanced'],
            'difficulty_level': 'advanced',
            'url_path': '/content/ai-ethics-talk'
        },
        
        # Kinesthetic Content
        {
            'title': 'Hands-on Python Coding Workshop',
            'description': 'Interactive coding exercises to learn Python programming.',
            'content_type': 'interactive',
            'style_tags': ['kinesthetic', 'beginner'],
            'difficulty_level': 'beginner',
            'url_path': '/content/python-workshop'
        },
        {
            'title': 'Build a Web App from Scratch',
            'description': 'Step-by-step project to create a complete web application.',
            'content_type': 'interactive',
            'style_tags': ['kinesthetic', 'intermediate'],
            'difficulty_level': 'intermediate',
            'url_path': '/content/webapp-project'
        },
        {
            'title': 'Machine Learning Lab Experiments',
            'description': 'Hands-on experiments with real datasets and ML algorithms.',
            'content_type': 'interactive',
            'style_tags': ['kinesthetic', 'advanced'],
            'difficulty_level': 'advanced',
            'url_path': '/content/ml-lab'
        },
        
        # Mixed Content
        {
            'title': 'Software Engineering Best Practices',
            'description': 'Comprehensive guide to software development practices.',
            'content_type': 'text',
            'style_tags': ['visual', 'auditory'],
            'difficulty_level': 'intermediate',
            'url_path': '/content/software-practices'
        }
    ]
    
    content_manager = ContentManager()
    
    for content_data in sample_content:
        existing_content = ContentLibrary.query.filter_by(title=content_data['title']).first()
        if not existing_content:
            content_manager.add_content(
                title=content_data['title'],
                description=content_data['description'],
                content_type=content_data['content_type'],
                style_tags=content_data['style_tags'],
                difficulty_level=content_data['difficulty_level'],
                url_path=content_data['url_path']
            )
    
    print(f"Seeded {len(sample_content)} sample content items")

if __name__ == "__main__":
    # Example usage
    content_manager = ContentManager()
    
    # Example: Get personalized content for a user
    # user_recommendations = content_manager.get_personalized_content(
    #     user_id=1,
    #     style_weights={'visual': 0.6, 'auditory': 0.3, 'kinesthetic': 0.1}
    # )
    # print(f"Recommended content: {user_recommendations}")
    
    seed_sample_content()