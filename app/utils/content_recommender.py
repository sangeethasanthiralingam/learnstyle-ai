"""
LearnStyle AI - Advanced Content Recommendation System
Analyzes user interests and suggests next content based on multiple factors
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import func, desc, and_, or_
from app.models import ContentLibrary, UserProgress, LearningProfile, ChatHistory, QuestionHistory, db
import logging

logger = logging.getLogger(__name__)

class ContentRecommender:
    """
    Advanced content recommendation system that analyzes user interests,
    learning patterns, and engagement to suggest personalized next content
    """
    
    def __init__(self):
        self.interest_weights = {
            'machine_learning': ['ml', 'machine learning', 'neural', 'ai', 'artificial intelligence', 'deep learning'],
            'programming': ['python', 'programming', 'code', 'algorithm', 'software', 'development'],
            'data_science': ['data', 'analysis', 'statistics', 'visualization', 'analytics', 'database'],
            'mathematics': ['math', 'mathematics', 'calculus', 'linear algebra', 'statistics', 'probability'],
            'web_development': ['web', 'html', 'css', 'javascript', 'frontend', 'backend', 'react', 'vue'],
            'mobile_development': ['mobile', 'ios', 'android', 'flutter', 'react native', 'swift', 'kotlin'],
            'cybersecurity': ['security', 'cybersecurity', 'hacking', 'encryption', 'vulnerability', 'penetration'],
            'cloud_computing': ['cloud', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'microservices'],
            'blockchain': ['blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'smart contracts', 'defi'],
            'ui_ux': ['ui', 'ux', 'design', 'user interface', 'user experience', 'wireframe', 'prototype']
        }
        
        self.content_type_preferences = {
            'video': 1.0,
            'interactive': 0.9,
            'text': 0.8,
            'audio': 0.7
        }
        
        self.difficulty_progression = {
            'beginner': ['beginner', 'intermediate'],
            'intermediate': ['beginner', 'intermediate', 'advanced'],
            'advanced': ['intermediate', 'advanced']
        }
    
    def get_next_content_recommendations(self, user_id: int, limit: int = 5) -> List[Dict]:
        """
        Get personalized next content recommendations based on user interests and patterns
        
        Args:
            user_id: User ID for personalization
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended content with reasoning
        """
        try:
            # Get user profile and learning style
            profile = LearningProfile.query.filter_by(user_id=user_id).first()
            if not profile:
                return self._get_fallback_recommendations(limit)
            
            # Analyze user interests from various sources
            user_interests = self._analyze_user_interests(user_id)
            
            # Get user's learning patterns
            learning_patterns = self._analyze_learning_patterns(user_id)
            
            # Get content consumption history
            consumption_history = self._get_consumption_history(user_id)
            
            # Generate recommendations based on multiple factors
            recommendations = []
            
            # 1. Interest-based recommendations (40% weight)
            interest_recs = self._get_interest_based_recommendations(
                user_interests, profile.dominant_style, limit // 2
            )
            recommendations.extend(interest_recs)
            
            # 2. Learning progression recommendations (30% weight)
            progression_recs = self._get_progression_recommendations(
                user_id, learning_patterns, limit // 3
            )
            recommendations.extend(progression_recs)
            
            # 3. Collaborative filtering (20% weight)
            collaborative_recs = self._get_collaborative_recommendations(
                user_id, profile.dominant_style, limit // 4
            )
            recommendations.extend(collaborative_recs)
            
            # 4. Diversity injection (10% weight)
            diversity_recs = self._get_diversity_recommendations(
                user_id, profile.dominant_style, limit // 5
            )
            recommendations.extend(diversity_recs)
            
            # Score and rank recommendations
            scored_recommendations = self._score_recommendations(
                recommendations, user_interests, learning_patterns, profile
            )
            
            # Remove duplicates and return top recommendations
            unique_recommendations = self._remove_duplicates(scored_recommendations)
            
            return unique_recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return self._get_fallback_recommendations(limit)
    
    def _analyze_user_interests(self, user_id: int) -> Dict[str, float]:
        """Analyze user interests from chat history, questions, and content consumption"""
        interests = {}
        
        try:
            # Analyze chat history for interests
            chat_interests = self._extract_interests_from_chat(user_id)
            
            # Analyze question history for interests
            question_interests = self._extract_interests_from_questions(user_id)
            
            # Analyze content consumption patterns
            content_interests = self._extract_interests_from_content(user_id)
            
            # Combine all interest sources
            all_interests = {**chat_interests, **question_interests, **content_interests}
            
            # Normalize interest scores
            total_score = sum(all_interests.values())
            if total_score > 0:
                interests = {k: v / total_score for k, v in all_interests.items()}
            
            return interests
            
        except Exception as e:
            logger.error(f"Error analyzing user interests: {e}")
            return {}
    
    def _extract_interests_from_chat(self, user_id: int) -> Dict[str, float]:
        """Extract interests from chat history"""
        interests = {}
        
        try:
            # Get recent chat messages
            recent_chats = ChatHistory.query.filter_by(user_id=user_id).filter(
                ChatHistory.timestamp >= datetime.now() - timedelta(days=30)
            ).all()
            
            for chat in recent_chats:
                message = (chat.user_message + " " + chat.ai_response).lower()
                
                for interest, keywords in self.interest_weights.items():
                    score = sum(1 for keyword in keywords if keyword in message)
                    if score > 0:
                        interests[interest] = interests.get(interest, 0) + score
            
            return interests
            
        except Exception as e:
            logger.error(f"Error extracting interests from chat: {e}")
            return {}
    
    def _extract_interests_from_questions(self, user_id: int) -> Dict[str, float]:
        """Extract interests from question history"""
        interests = {}
        
        try:
            # Get recent questions
            recent_questions = QuestionHistory.query.filter_by(user_id=user_id).filter(
                QuestionHistory.timestamp >= datetime.now() - timedelta(days=30)
            ).all()
            
            for qa in recent_questions:
                question = qa.question.lower()
                
                for interest, keywords in self.interest_weights.items():
                    score = sum(1 for keyword in keywords if keyword in question)
                    if score > 0:
                        interests[interest] = interests.get(interest, 0) + score * 1.5  # Questions weighted higher
            
            return interests
            
        except Exception as e:
            logger.error(f"Error extracting interests from questions: {e}")
            return {}
    
    def _extract_interests_from_content(self, user_id: int) -> Dict[str, float]:
        """Extract interests from content consumption patterns"""
        interests = {}
        
        try:
            # Get user's content consumption
            progress_records = UserProgress.query.filter_by(user_id=user_id).all()
            
            for progress in progress_records:
                if progress.content:
                    content_text = (progress.content.title + " " + progress.content.description).lower()
                    
                    for interest, keywords in self.interest_weights.items():
                        score = sum(1 for keyword in keywords if keyword in content_text)
                        if score > 0:
                            # Weight by engagement (completion, rating, score)
                            engagement_multiplier = 1.0
                            if progress.completion_status == 'completed':
                                engagement_multiplier += 0.5
                            if progress.engagement_rating:
                                engagement_multiplier += progress.engagement_rating / 5.0
                            if progress.score:
                                engagement_multiplier += progress.score / 100.0
                            
                            interests[interest] = interests.get(interest, 0) + score * engagement_multiplier
            
            return interests
            
        except Exception as e:
            logger.error(f"Error extracting interests from content: {e}")
            return {}
    
    def _analyze_learning_patterns(self, user_id: int) -> Dict[str, any]:
        """Analyze user's learning patterns and preferences"""
        patterns = {
            'preferred_content_types': {},
            'preferred_difficulty': 'beginner',
            'learning_velocity': 1.0,
            'engagement_trend': 'stable',
            'time_preferences': {},
            'completion_rate': 0.0
        }
        
        try:
            # Get user progress data
            progress_records = UserProgress.query.filter_by(user_id=user_id).all()
            
            if not progress_records:
                return patterns
            
            # Analyze content type preferences
            content_type_scores = {}
            difficulty_scores = {}
            completion_count = 0
            total_time = 0
            
            for progress in progress_records:
                if progress.content:
                    # Content type preferences
                    content_type = progress.content.content_type
                    engagement_score = (progress.engagement_rating or 0) + (progress.score or 0) / 20
                    content_type_scores[content_type] = content_type_scores.get(content_type, 0) + engagement_score
                    
                    # Difficulty preferences
                    difficulty = progress.content.difficulty_level
                    if progress.completion_status == 'completed':
                        difficulty_scores[difficulty] = difficulty_scores.get(difficulty, 0) + 1
                    
                    # Completion rate
                    if progress.completion_status == 'completed':
                        completion_count += 1
                    
                    # Time tracking
                    if progress.time_spent:
                        total_time += progress.time_spent
            
            # Set preferred content types
            if content_type_scores:
                patterns['preferred_content_types'] = dict(sorted(
                    content_type_scores.items(), key=lambda x: x[1], reverse=True
                ))
            
            # Set preferred difficulty
            if difficulty_scores:
                patterns['preferred_difficulty'] = max(difficulty_scores.items(), key=lambda x: x[1])[0]
            
            # Calculate completion rate
            patterns['completion_rate'] = completion_count / len(progress_records) if progress_records else 0
            
            # Calculate learning velocity (content per week)
            if progress_records:
                first_content = min(progress_records, key=lambda x: x.timestamp)
                weeks_learning = (datetime.now() - first_content.timestamp).days / 7
                patterns['learning_velocity'] = len(progress_records) / max(weeks_learning, 1)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing learning patterns: {e}")
            return patterns
    
    def _get_consumption_history(self, user_id: int) -> List[Dict]:
        """Get user's content consumption history"""
        try:
            progress_records = UserProgress.query.filter_by(user_id=user_id).order_by(
                desc(UserProgress.timestamp)
            ).limit(50).all()
            
            history = []
            for progress in progress_records:
                if progress.content:
                    history.append({
                        'content_id': progress.content.id,
                        'title': progress.content.title,
                        'content_type': progress.content.content_type,
                        'difficulty_level': progress.content.difficulty_level,
                        'style_tags': progress.content.get_style_tags_list(),
                        'completion_status': progress.completion_status,
                        'engagement_rating': progress.engagement_rating,
                        'score': progress.score,
                        'time_spent': progress.time_spent,
                        'timestamp': progress.timestamp
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting consumption history: {e}")
            return []
    
    def _get_interest_based_recommendations(self, interests: Dict[str, float], 
                                          learning_style: str, limit: int) -> List[Dict]:
        """Get recommendations based on user interests"""
        recommendations = []
        
        try:
            # Sort interests by score
            sorted_interests = sorted(interests.items(), key=lambda x: x[1], reverse=True)
            
            for interest, score in sorted_interests[:3]:  # Top 3 interests
                # Find content matching this interest
                keywords = self.interest_weights.get(interest, [])
                if not keywords:
                    continue
                
                # Build search query
                search_conditions = []
                for keyword in keywords:
                    search_conditions.append(
                        or_(
                            ContentLibrary.title.ilike(f'%{keyword}%'),
                            ContentLibrary.description.ilike(f'%{keyword}%')
                        )
                    )
                
                # Add learning style filter
                style_condition = ContentLibrary.style_tags.like(f'%{learning_style}%')
                
                # Query content
                content_items = ContentLibrary.query.filter(
                    and_(
                        or_(*search_conditions),
                        style_condition
                    )
                ).limit(limit).all()
                
                for item in content_items:
                    recommendations.append({
                        'content': self._content_to_dict(item),
                        'reason': f"Matches your interest in {interest.replace('_', ' ').title()}",
                        'confidence': score,
                        'type': 'interest_based'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting interest-based recommendations: {e}")
            return []
    
    def _get_progression_recommendations(self, user_id: int, patterns: Dict, limit: int) -> List[Dict]:
        """Get recommendations based on learning progression"""
        recommendations = []
        
        try:
            # Get user's current difficulty level
            current_difficulty = patterns.get('preferred_difficulty', 'beginner')
            next_difficulties = self.difficulty_progression.get(current_difficulty, [current_difficulty])
            
            # Get user's preferred content types
            preferred_types = list(patterns.get('preferred_content_types', {}).keys())[:2]
            
            # Query content for progression
            query_conditions = [
                ContentLibrary.difficulty_level.in_(next_difficulties)
            ]
            
            if preferred_types:
                query_conditions.append(ContentLibrary.content_type.in_(preferred_types))
            
            content_items = ContentLibrary.query.filter(
                and_(*query_conditions)
            ).limit(limit).all()
            
            for item in content_items:
                recommendations.append({
                    'content': self._content_to_dict(item),
                    'reason': f"Next step in your learning progression ({current_difficulty} → {item.difficulty_level})",
                    'confidence': 0.8,
                    'type': 'progression'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting progression recommendations: {e}")
            return []
    
    def _get_collaborative_recommendations(self, user_id: int, learning_style: str, limit: int) -> List[Dict]:
        """Get recommendations using collaborative filtering"""
        recommendations = []
        
        try:
            # Find users with similar learning styles and preferences
            similar_users = self._find_similar_users(user_id, learning_style)
            
            if not similar_users:
                return recommendations
            
            # Get content liked by similar users
            similar_user_ids = [user['user_id'] for user in similar_users]
            
            # Query content with high engagement from similar users
            content_items = db.session.query(ContentLibrary).join(UserProgress).filter(
                and_(
                    UserProgress.user_id.in_(similar_user_ids),
                    UserProgress.completion_status == 'completed',
                    or_(
                        UserProgress.engagement_rating >= 4,
                        UserProgress.score >= 80
                    ),
                    ContentLibrary.style_tags.like(f'%{learning_style}%')
                )
            ).distinct().limit(limit).all()
            
            for item in content_items:
                recommendations.append({
                    'content': self._content_to_dict(item),
                    'reason': f"Popular among learners with similar preferences",
                    'confidence': 0.7,
                    'type': 'collaborative'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting collaborative recommendations: {e}")
            return []
    
    def _get_diversity_recommendations(self, user_id: int, learning_style: str, limit: int) -> List[Dict]:
        """Get diverse recommendations to prevent filter bubbles"""
        recommendations = []
        
        try:
            # Get content from different categories and difficulty levels
            content_items = ContentLibrary.query.filter(
                ContentLibrary.style_tags.like(f'%{learning_style}%')
            ).order_by(func.random()).limit(limit).all()
            
            for item in content_items:
                recommendations.append({
                    'content': self._content_to_dict(item),
                    'reason': "Diverse content to expand your learning horizons",
                    'confidence': 0.5,
                    'type': 'diversity'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting diversity recommendations: {e}")
            return []
    
    def _find_similar_users(self, user_id: int, learning_style: str) -> List[Dict]:
        """Find users with similar learning patterns"""
        try:
            # Get current user's preferences
            user_progress = UserProgress.query.filter_by(user_id=user_id).all()
            user_content_types = set()
            user_difficulties = set()
            
            for progress in user_progress:
                if progress.content:
                    user_content_types.add(progress.content.content_type)
                    user_difficulties.add(progress.content.difficulty_level)
            
            # Find users with similar preferences
            similar_users = db.session.query(
                UserProgress.user_id,
                func.count(UserProgress.id).label('common_content'),
                func.avg(UserProgress.engagement_rating).label('avg_engagement')
            ).join(ContentLibrary).filter(
                and_(
                    UserProgress.user_id != user_id,
                    ContentLibrary.content_type.in_(user_content_types),
                    ContentLibrary.difficulty_level.in_(user_difficulties),
                    ContentLibrary.style_tags.like(f'%{learning_style}%')
                )
            ).group_by(UserProgress.user_id).having(
                func.count(UserProgress.id) >= 2
            ).order_by(desc('common_content')).limit(10).all()
            
            return [
                {
                    'user_id': user.user_id,
                    'common_content': user.common_content,
                    'avg_engagement': float(user.avg_engagement or 0)
                }
                for user in similar_users
            ]
            
        except Exception as e:
            logger.error(f"Error finding similar users: {e}")
            return []
    
    def _score_recommendations(self, recommendations: List[Dict], interests: Dict[str, float],
                             patterns: Dict, profile: LearningProfile) -> List[Dict]:
        """Score and rank recommendations"""
        try:
            for rec in recommendations:
                content = rec['content']
                score = 0.0
                
                # Base score from recommendation type
                type_weights = {
                    'interest_based': 0.4,
                    'progression': 0.3,
                    'collaborative': 0.2,
                    'diversity': 0.1
                }
                score += type_weights.get(rec['type'], 0.1) * rec['confidence']
                
                # Boost for learning style match
                if profile.dominant_style in content.get('style_tags', []):
                    score += 0.2
                
                # Boost for preferred content type
                preferred_types = patterns.get('preferred_content_types', {})
                if content['content_type'] in preferred_types:
                    score += 0.1
                
                # Boost for appropriate difficulty
                preferred_difficulty = patterns.get('preferred_difficulty', 'beginner')
                if content['difficulty_level'] == preferred_difficulty:
                    score += 0.1
                
                # Boost for high completion rate content
                content_analytics = self._get_content_analytics(content['id'])
                if content_analytics.get('completion_rate', 0) > 0.8:
                    score += 0.1
                
                rec['final_score'] = score
            
            # Sort by final score
            return sorted(recommendations, key=lambda x: x['final_score'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error scoring recommendations: {e}")
            return recommendations
    
    def _get_content_analytics(self, content_id: int) -> Dict:
        """Get analytics for specific content"""
        try:
            progress_records = UserProgress.query.filter_by(content_id=content_id).all()
            
            if not progress_records:
                return {'completion_rate': 0, 'avg_engagement': 0, 'avg_score': 0}
            
            completed = len([p for p in progress_records if p.completion_status == 'completed'])
            avg_engagement = sum(p.engagement_rating for p in progress_records if p.engagement_rating) / len(progress_records)
            avg_score = sum(p.score for p in progress_records if p.score) / len(progress_records)
            
            return {
                'completion_rate': completed / len(progress_records),
                'avg_engagement': avg_engagement,
                'avg_score': avg_score
            }
            
        except Exception as e:
            logger.error(f"Error getting content analytics: {e}")
            return {'completion_rate': 0, 'avg_engagement': 0, 'avg_score': 0}
    
    def _remove_duplicates(self, recommendations: List[Dict]) -> List[Dict]:
        """Remove duplicate recommendations"""
        seen_ids = set()
        unique_recommendations = []
        
        for rec in recommendations:
            content_id = rec['content']['id']
            if content_id not in seen_ids:
                seen_ids.add(content_id)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
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
    
    def _get_fallback_recommendations(self, limit: int) -> List[Dict]:
        """Get fallback recommendations when user analysis fails"""
        try:
            content_items = ContentLibrary.query.limit(limit).all()
            
            return [
                {
                    'content': self._content_to_dict(item),
                    'reason': "Popular content to get you started",
                    'confidence': 0.5,
                    'type': 'fallback',
                    'final_score': 0.5
                }
                for item in content_items
            ]
            
        except Exception as e:
            logger.error(f"Error getting fallback recommendations: {e}")
            return []
