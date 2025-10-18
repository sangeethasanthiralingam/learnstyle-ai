"""
Advanced Gamification System for LearnStyle AI
Style-specific achievements, social learning features, and progress visualization
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import math

class AchievementType(Enum):
    """Types of achievements available"""
    STYLE_SPECIFIC = "style_specific"
    LEARNING_MILESTONE = "learning_milestone"
    SOCIAL_LEARNING = "social_learning"
    CONSISTENCY = "consistency"
    MASTERY = "mastery"
    EXPLORATION = "exploration"

class BadgeRarity(Enum):
    """Rarity levels for badges"""
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"

@dataclass
class Achievement:
    """Individual achievement definition"""
    id: str
    name: str
    description: str
    icon: str
    rarity: BadgeRarity
    category: AchievementType
    style_requirement: Optional[str]  # visual, auditory, kinesthetic, or None
    requirements: Dict
    points: int
    unlocked_at: Optional[datetime] = None

@dataclass
class UserProgress:
    """User's gamification progress"""
    user_id: int
    total_points: int
    level: int
    achievements: List[str]
    current_streak: int
    longest_streak: int
    style_mastery: Dict[str, float]  # Style -> mastery percentage
    social_score: int
    last_activity: datetime

class GamificationEngine:
    """
    Advanced gamification system with style-specific achievements
    and social learning features
    """
    
    def __init__(self):
        self.achievements = self._initialize_achievements()
        self.levels = self._initialize_levels()
        self.social_features = SocialLearningFeatures()
    
    def _initialize_achievements(self) -> Dict[str, Achievement]:
        """Initialize all available achievements"""
        achievements = {}
        
        # Visual Learning Achievements
        achievements['visual_reader'] = Achievement(
            id='visual_reader',
            name='Infographic Master',
            description='Read 50 visual content pieces',
            icon='📊',
            rarity=BadgeRarity.COMMON,
            category=AchievementType.STYLE_SPECIFIC,
            style_requirement='visual',
            requirements={'visual_content_read': 50},
            points=100
        )
        
        achievements['diagram_expert'] = Achievement(
            id='diagram_expert',
            name='Diagram Expert',
            description='Create 10 visual summaries',
            icon='📈',
            rarity=BadgeRarity.UNCOMMON,
            category=AchievementType.STYLE_SPECIFIC,
            style_requirement='visual',
            requirements={'diagrams_created': 10},
            points=250
        )
        
        achievements['visual_artist'] = Achievement(
            id='visual_artist',
            name='Visual Artist',
            description='Achieve 90% visual learning mastery',
            icon='🎨',
            rarity=BadgeRarity.RARE,
            category=AchievementType.MASTERY,
            style_requirement='visual',
            requirements={'visual_mastery': 0.9},
            points=500
        )
        
        # Auditory Learning Achievements
        achievements['active_listener'] = Achievement(
            id='active_listener',
            name='Active Listener',
            description='Complete 30 audio content pieces',
            icon='🎧',
            rarity=BadgeRarity.COMMON,
            category=AchievementType.STYLE_SPECIFIC,
            style_requirement='auditory',
            requirements={'audio_content_completed': 30},
            points=100
        )
        
        achievements['discussion_leader'] = Achievement(
            id='discussion_leader',
            name='Discussion Leader',
            description='Participate in 20 group discussions',
            icon='💬',
            rarity=BadgeRarity.UNCOMMON,
            category=AchievementType.SOCIAL_LEARNING,
            style_requirement='auditory',
            requirements={'discussions_participated': 20},
            points=300
        )
        
        achievements['audio_master'] = Achievement(
            id='audio_master',
            name='Audio Master',
            description='Achieve 90% auditory learning mastery',
            icon='🎵',
            rarity=BadgeRarity.RARE,
            category=AchievementType.MASTERY,
            style_requirement='auditory',
            requirements={'auditory_mastery': 0.9},
            points=500
        )
        
        # Kinesthetic Learning Achievements
        achievements['hands_on_explorer'] = Achievement(
            id='hands_on_explorer',
            name='Hands-On Explorer',
            description='Complete 25 interactive exercises',
            icon='🛠️',
            rarity=BadgeRarity.COMMON,
            category=AchievementType.STYLE_SPECIFIC,
            style_requirement='kinesthetic',
            requirements={'interactive_exercises': 25},
            points=100
        )
        
        achievements['project_builder'] = Achievement(
            id='project_builder',
            name='Project Builder',
            description='Complete 5 hands-on projects',
            icon='🏗️',
            rarity=BadgeRarity.UNCOMMON,
            category=AchievementType.STYLE_SPECIFIC,
            style_requirement='kinesthetic',
            requirements={'projects_completed': 5},
            points=300
        )
        
        achievements['kinesthetic_master'] = Achievement(
            id='kinesthetic_master',
            name='Kinesthetic Master',
            description='Achieve 90% kinesthetic learning mastery',
            icon='🤲',
            rarity=BadgeRarity.RARE,
            category=AchievementType.MASTERY,
            style_requirement='kinesthetic',
            requirements={'kinesthetic_mastery': 0.9},
            points=500
        )
        
        # Cross-Style Achievements
        achievements['style_versatile'] = Achievement(
            id='style_versatile',
            name='Style Versatile',
            description='Achieve 70% mastery in all three learning styles',
            icon='🌈',
            rarity=BadgeRarity.EPIC,
            category=AchievementType.MASTERY,
            style_requirement=None,
            requirements={'visual_mastery': 0.7, 'auditory_mastery': 0.7, 'kinesthetic_mastery': 0.7},
            points=1000
        )
        
        achievements['learning_marathon'] = Achievement(
            id='learning_marathon',
            name='Learning Marathon',
            description='Maintain a 30-day learning streak',
            icon='🏃',
            rarity=BadgeRarity.RARE,
            category=AchievementType.CONSISTENCY,
            style_requirement=None,
            requirements={'streak_days': 30},
            points=750
        )
        
        achievements['social_butterfly'] = Achievement(
            id='social_butterfly',
            name='Social Butterfly',
            description='Help 10 other learners with their studies',
            icon='🦋',
            rarity=BadgeRarity.UNCOMMON,
            category=AchievementType.SOCIAL_LEARNING,
            style_requirement=None,
            requirements={'learners_helped': 10},
            points=400
        )
        
        achievements['explorer'] = Achievement(
            id='explorer',
            name='Learning Explorer',
            description='Try content from all three learning styles',
            icon='🗺️',
            rarity=BadgeRarity.COMMON,
            category=AchievementType.EXPLORATION,
            style_requirement=None,
            requirements={'styles_explored': 3},
            points=150
        )
        
        achievements['quiz_master'] = Achievement(
            id='quiz_master',
            name='Quiz Master',
            description='Score 95% or higher on 10 quizzes',
            icon='🧠',
            rarity=BadgeRarity.UNCOMMON,
            category=AchievementType.LEARNING_MILESTONE,
            style_requirement=None,
            requirements={'high_score_quizzes': 10, 'min_score': 0.95},
            points=350
        )
        
        achievements['early_bird'] = Achievement(
            id='early_bird',
            name='Early Bird',
            description='Complete learning activities before 8 AM for 7 days',
            icon='🌅',
            rarity=BadgeRarity.RARE,
            category=AchievementType.CONSISTENCY,
            style_requirement=None,
            requirements={'early_morning_days': 7},
            points=500
        )
        
        achievements['night_owl'] = Achievement(
            id='night_owl',
            name='Night Owl',
            description='Complete learning activities after 10 PM for 7 days',
            icon='🦉',
            rarity=BadgeRarity.RARE,
            category=AchievementType.CONSISTENCY,
            style_requirement=None,
            requirements={'late_night_days': 7},
            points=500
        )
        
        return achievements
    
    def _initialize_levels(self) -> List[Dict]:
        """Initialize level progression system"""
        levels = []
        for i in range(1, 101):  # 100 levels
            required_points = int(1000 * (i ** 1.5))  # Exponential growth
            levels.append({
                'level': i,
                'required_points': required_points,
                'title': self._get_level_title(i),
                'color': self._get_level_color(i),
                'perks': self._get_level_perks(i)
            })
        return levels
    
    def _get_level_title(self, level: int) -> str:
        """Get title for level"""
        if level <= 10:
            return "Learning Novice"
        elif level <= 25:
            return "Knowledge Seeker"
        elif level <= 50:
            return "Learning Enthusiast"
        elif level <= 75:
            return "Knowledge Master"
        elif level <= 90:
            return "Learning Sage"
        else:
            return "Learning Legend"
    
    def _get_level_color(self, level: int) -> str:
        """Get color for level"""
        if level <= 10:
            return "#6c757d"  # Gray
        elif level <= 25:
            return "#28a745"  # Green
        elif level <= 50:
            return "#007bff"  # Blue
        elif level <= 75:
            return "#6f42c1"  # Purple
        elif level <= 90:
            return "#fd7e14"  # Orange
        else:
            return "#dc3545"  # Red
    
    def _get_level_perks(self, level: int) -> List[str]:
        """Get perks for level"""
        perks = []
        
        if level >= 5:
            perks.append("Unlock advanced content")
        if level >= 10:
            perks.append("Access to study groups")
        if level >= 15:
            perks.append("Priority AI tutor responses")
        if level >= 20:
            perks.append("Custom learning paths")
        if level >= 30:
            perks.append("Mentor other learners")
        if level >= 40:
            perks.append("Create content")
        if level >= 50:
            perks.append("Beta feature access")
        if level >= 75:
            perks.append("Learning analytics dashboard")
        if level >= 90:
            perks.append("Exclusive content library")
        
        return perks
    
    def check_achievements(self, user_id: int, user_stats: Dict) -> List[Achievement]:
        """Check and unlock new achievements for user"""
        unlocked_achievements = []
        
        for achievement_id, achievement in self.achievements.items():
            if self._is_achievement_unlocked(achievement, user_stats):
                unlocked_achievements.append(achievement)
        
        return unlocked_achievements
    
    def _is_achievement_unlocked(self, achievement: Achievement, user_stats: Dict) -> bool:
        """Check if specific achievement is unlocked"""
        for requirement, threshold in achievement.requirements.items():
            if user_stats.get(requirement, 0) < threshold:
                return False
        return True
    
    def calculate_user_level(self, total_points: int) -> int:
        """Calculate user level based on total points"""
        for level_info in reversed(self.levels):
            if total_points >= level_info['required_points']:
                return level_info['level']
        return 1
    
    def get_progress_to_next_level(self, current_level: int, current_points: int) -> Dict:
        """Get progress information for next level"""
        if current_level >= len(self.levels):
            return {
                'current_level': current_level,
                'next_level': current_level,
                'points_needed': 0,
                'progress_percentage': 100,
                'is_max_level': True
            }
        
        current_level_info = self.levels[current_level - 1]
        next_level_info = self.levels[current_level]
        
        points_needed = next_level_info['required_points'] - current_points
        progress_points = current_points - current_level_info['required_points']
        total_progress_points = next_level_info['required_points'] - current_level_info['required_points']
        progress_percentage = (progress_points / total_progress_points) * 100
        
        return {
            'current_level': current_level,
            'next_level': current_level + 1,
            'points_needed': points_needed,
            'progress_percentage': min(progress_percentage, 100),
            'is_max_level': False
        }
    
    def generate_style_specific_recommendations(self, 
                                             user_style: str, 
                                             user_level: int) -> List[Dict]:
        """Generate style-specific learning recommendations"""
        
        recommendations = []
        
        # Visual learners
        if user_style == 'visual':
            recommendations.extend([
                {
                    'type': 'achievement',
                    'title': 'Create Visual Summaries',
                    'description': 'Turn your notes into infographics and mind maps',
                    'points': 50,
                    'difficulty': 'easy'
                },
                {
                    'type': 'challenge',
                    'title': 'Diagram Master Challenge',
                    'description': 'Create 5 diagrams this week',
                    'points': 200,
                    'difficulty': 'medium'
                }
            ])
        
        # Auditory learners
        elif user_style == 'auditory':
            recommendations.extend([
                {
                    'type': 'achievement',
                    'title': 'Join Study Groups',
                    'description': 'Participate in group discussions',
                    'points': 75,
                    'difficulty': 'easy'
                },
                {
                    'type': 'challenge',
                    'title': 'Teaching Challenge',
                    'description': 'Explain a concept to another learner',
                    'points': 150,
                    'difficulty': 'medium'
                }
            ])
        
        # Kinesthetic learners
        elif user_style == 'kinesthetic':
            recommendations.extend([
                {
                    'type': 'achievement',
                    'title': 'Build Something',
                    'description': 'Complete a hands-on project',
                    'points': 100,
                    'difficulty': 'medium'
                },
                {
                    'type': 'challenge',
                    'title': 'Maker Challenge',
                    'description': 'Create 3 interactive exercises',
                    'points': 250,
                    'difficulty': 'hard'
                }
            ])
        
        return recommendations

class SocialLearningFeatures:
    """Social learning and collaboration features"""
    
    def __init__(self):
        self.study_groups = {}
        self.mentorship_pairs = {}
        self.collaborative_challenges = {}
    
    def create_study_group(self, creator_id: int, style_focus: str, max_members: int = 6) -> str:
        """Create a new study group"""
        group_id = f"group_{len(self.study_groups) + 1}"
        
        self.study_groups[group_id] = {
            'id': group_id,
            'creator_id': creator_id,
            'style_focus': style_focus,
            'members': [creator_id],
            'max_members': max_members,
            'created_at': datetime.now(),
            'active': True
        }
        
        return group_id
    
    def join_study_group(self, user_id: int, group_id: str) -> bool:
        """Join an existing study group"""
        if group_id not in self.study_groups:
            return False
        
        group = self.study_groups[group_id]
        if len(group['members']) >= group['max_members']:
            return False
        
        if user_id not in group['members']:
            group['members'].append(user_id)
            return True
        
        return False
    
    def find_compatible_study_partners(self, user_id: int, user_style: str) -> List[Dict]:
        """Find study partners with complementary or similar learning styles"""
        # This would integrate with user database in real implementation
        compatible_partners = []
        
        # Example logic for finding partners
        compatibility_scores = {
            'visual': {'visual': 0.8, 'kinesthetic': 0.6, 'auditory': 0.4},
            'auditory': {'auditory': 0.8, 'visual': 0.4, 'kinesthetic': 0.6},
            'kinesthetic': {'kinesthetic': 0.8, 'visual': 0.6, 'auditory': 0.4}
        }
        
        # In real implementation, this would query the database
        # for users with compatible styles and similar learning goals
        
        return compatible_partners
    
    def create_collaborative_challenge(self, 
                                     creator_id: int, 
                                     challenge_type: str,
                                     description: str,
                                     points: int) -> str:
        """Create a collaborative learning challenge"""
        challenge_id = f"challenge_{len(self.collaborative_challenges) + 1}"
        
        self.collaborative_challenges[challenge_id] = {
            'id': challenge_id,
            'creator_id': creator_id,
            'type': challenge_type,
            'description': description,
            'points': points,
            'participants': [],
            'created_at': datetime.now(),
            'status': 'active'
        }
        
        return challenge_id
    
    def get_social_leaderboard(self, timeframe: str = 'weekly') -> List[Dict]:
        """Get social learning leaderboard"""
        # This would integrate with user progress database
        leaderboard = []
        
        # Example leaderboard data
        if timeframe == 'weekly':
            leaderboard = [
                {'user_id': 1, 'username': 'VisualLearner', 'points': 1250, 'rank': 1},
                {'user_id': 2, 'username': 'AudioMaster', 'points': 1180, 'rank': 2},
                {'user_id': 3, 'username': 'HandsOnHero', 'points': 1100, 'rank': 3}
            ]
        
        return leaderboard

# Example usage
if __name__ == "__main__":
    # Initialize gamification engine
    gamification = GamificationEngine()
    
    # Example user stats
    user_stats = {
        'visual_content_read': 60,
        'audio_content_completed': 25,
        'interactive_exercises': 30,
        'discussions_participated': 15,
        'projects_completed': 3,
        'visual_mastery': 0.85,
        'auditory_mastery': 0.70,
        'kinesthetic_mastery': 0.75,
        'streak_days': 15,
        'learners_helped': 5,
        'styles_explored': 3,
        'high_score_quizzes': 8
    }
    
    # Check for new achievements
    new_achievements = gamification.check_achievements(1, user_stats)
    print(f"New achievements unlocked: {len(new_achievements)}")
    for achievement in new_achievements:
        print(f"- {achievement.name}: {achievement.description}")
    
    # Calculate level
    total_points = sum(achievement.points for achievement in new_achievements)
    level = gamification.calculate_user_level(total_points)
    print(f"Current level: {level}")
    
    # Get progress to next level
    progress = gamification.get_progress_to_next_level(level, total_points)
    print(f"Progress to next level: {progress['progress_percentage']:.1f}%")
