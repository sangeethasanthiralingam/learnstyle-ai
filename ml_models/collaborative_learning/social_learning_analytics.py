"""
Social Learning Analytics Module

This module provides comprehensive social learning analytics including:
- Collaboration effectiveness measurement
- Social learning network analysis
- Peer influence tracking
- Learning community insights
- Social engagement metrics

Author: LearnStyle AI Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import networkx as nx

logger = logging.getLogger(__name__)

class SocialMetric(Enum):
    """Types of social learning metrics"""
    COLLABORATION_STRENGTH = "collaboration_strength"
    PEER_INFLUENCE = "peer_influence"
    LEARNING_CENTRALITY = "learning_centrality"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    SOCIAL_ENGAGEMENT = "social_engagement"

class NetworkPosition(Enum):
    """Network position classifications"""
    CENTRAL = "central"
    BRIDGE = "bridge"
    PERIPHERAL = "peripheral"
    ISOLATED = "isolated"

@dataclass
class SocialLearningProfile:
    """Individual social learning profile"""
    user_id: str
    collaboration_score: float
    peer_influence_score: float
    learning_centrality: float
    knowledge_sharing_score: float
    social_engagement_score: float
    network_position: NetworkPosition
    active_connections: int
    learning_community_size: int
    contribution_quality: float
    help_seeking_frequency: float
    help_providing_frequency: float

@dataclass
class SocialLearningMetrics:
    """Comprehensive social learning metrics"""
    total_users: int
    total_connections: int
    network_density: float
    average_clustering: float
    network_centralization: float
    knowledge_flow_rate: float
    collaboration_effectiveness: float
    peer_learning_impact: float
    social_engagement_level: float
    community_cohesion: float
    learning_acceleration: float
    recommendations: List[str]

class SocialLearningAnalytics:
    """
    Advanced social learning analytics system
    """
    
    def __init__(self):
        """Initialize social learning analytics"""
        self.interaction_history = []
        self.learning_networks = defaultdict(list)
        self.collaboration_patterns = defaultdict(list)
        self.knowledge_sharing_events = []
        
        logger.info("Social Learning Analytics initialized")
    
    def analyze_social_learning(self, interaction_data: List[Dict], 
                              learning_data: List[Dict]) -> SocialLearningMetrics:
        """
        Analyze social learning patterns and metrics
        
        Args:
            interaction_data: Social interaction data
            learning_data: Learning performance data
            
        Returns:
            SocialLearningMetrics object with comprehensive analysis
        """
        try:
            # Build learning network
            network = self._build_learning_network(interaction_data)
            
            # Calculate network metrics
            total_users = len(network.nodes())
            total_connections = len(network.edges())
            network_density = nx.density(network)
            average_clustering = nx.average_clustering(network)
            network_centralization = self._calculate_network_centralization(network)
            
            # Calculate social learning metrics
            knowledge_flow_rate = self._calculate_knowledge_flow_rate(interaction_data)
            collaboration_effectiveness = self._calculate_collaboration_effectiveness(interaction_data)
            peer_learning_impact = self._calculate_peer_learning_impact(learning_data)
            social_engagement_level = self._calculate_social_engagement_level(interaction_data)
            community_cohesion = self._calculate_community_cohesion(network)
            learning_acceleration = self._calculate_learning_acceleration(learning_data)
            
            # Generate recommendations
            recommendations = self._generate_social_learning_recommendations(
                network_density, collaboration_effectiveness, peer_learning_impact,
                social_engagement_level, community_cohesion
            )
            
            return SocialLearningMetrics(
                total_users=total_users,
                total_connections=total_connections,
                network_density=network_density,
                average_clustering=average_clustering,
                network_centralization=network_centralization,
                knowledge_flow_rate=knowledge_flow_rate,
                collaboration_effectiveness=collaboration_effectiveness,
                peer_learning_impact=peer_learning_impact,
                social_engagement_level=social_engagement_level,
                community_cohesion=community_cohesion,
                learning_acceleration=learning_acceleration,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error analyzing social learning: {str(e)}")
            return self._get_default_metrics()
    
    def create_social_learning_profile(self, user_id: str, 
                                     interaction_data: List[Dict],
                                     learning_data: List[Dict]) -> SocialLearningProfile:
        """
        Create social learning profile for individual user
        
        Args:
            user_id: User identifier
            interaction_data: User interaction data
            learning_data: User learning data
            
        Returns:
            SocialLearningProfile object
        """
        try:
            # Calculate individual metrics
            collaboration_score = self._calculate_user_collaboration_score(user_id, interaction_data)
            peer_influence_score = self._calculate_peer_influence_score(user_id, interaction_data)
            learning_centrality = self._calculate_learning_centrality(user_id, interaction_data)
            knowledge_sharing_score = self._calculate_knowledge_sharing_score(user_id, interaction_data)
            social_engagement_score = self._calculate_social_engagement_score(user_id, interaction_data)
            
            # Determine network position
            network_position = self._determine_network_position(
                user_id, learning_centrality, collaboration_score
            )
            
            # Calculate connection metrics
            active_connections = self._count_active_connections(user_id, interaction_data)
            learning_community_size = self._calculate_learning_community_size(user_id, interaction_data)
            
            # Calculate contribution and help metrics
            contribution_quality = self._calculate_contribution_quality(user_id, interaction_data)
            help_seeking_frequency = self._calculate_help_seeking_frequency(user_id, interaction_data)
            help_providing_frequency = self._calculate_help_providing_frequency(user_id, interaction_data)
            
            return SocialLearningProfile(
                user_id=user_id,
                collaboration_score=collaboration_score,
                peer_influence_score=peer_influence_score,
                learning_centrality=learning_centrality,
                knowledge_sharing_score=knowledge_sharing_score,
                social_engagement_score=social_engagement_score,
                network_position=network_position,
                active_connections=active_connections,
                learning_community_size=learning_community_size,
                contribution_quality=contribution_quality,
                help_seeking_frequency=help_seeking_frequency,
                help_providing_frequency=help_providing_frequency
            )
            
        except Exception as e:
            logger.error(f"Error creating social learning profile: {str(e)}")
            return self._get_default_profile(user_id)
    
    def _build_learning_network(self, interaction_data: List[Dict]) -> nx.Graph:
        """Build learning network from interaction data"""
        try:
            G = nx.Graph()
            
            # Add nodes and edges based on interactions
            for interaction in interaction_data:
                user1 = interaction.get('user1_id')
                user2 = interaction.get('user2_id')
                interaction_strength = interaction.get('interaction_strength', 1.0)
                
                if user1 and user2:
                    if G.has_edge(user1, user2):
                        # Increase edge weight for existing connections
                        G[user1][user2]['weight'] += interaction_strength
                    else:
                        G.add_edge(user1, user2, weight=interaction_strength)
            
            return G
            
        except Exception as e:
            logger.error(f"Error building learning network: {str(e)}")
            return nx.Graph()
    
    def _calculate_network_centralization(self, network: nx.Graph) -> float:
        """Calculate network centralization"""
        try:
            if len(network.nodes()) < 2:
                return 0.0
            
            # Calculate degree centralization
            degree_centrality = nx.degree_centrality(network)
            max_centrality = max(degree_centrality.values())
            
            # Calculate centralization
            centralization = sum(max_centrality - centrality for centrality in degree_centrality.values())
            centralization = centralization / (len(network.nodes()) - 1)
            
            return centralization
            
        except Exception as e:
            logger.error(f"Error calculating network centralization: {str(e)}")
            return 0.0
    
    def _calculate_knowledge_flow_rate(self, interaction_data: List[Dict]) -> float:
        """Calculate knowledge flow rate in the network"""
        try:
            if not interaction_data:
                return 0.0
            
            # Count knowledge sharing interactions
            knowledge_interactions = [
                interaction for interaction in interaction_data
                if interaction.get('interaction_type') == 'knowledge_sharing'
            ]
            
            # Calculate rate per day
            total_days = self._calculate_time_span(interaction_data)
            if total_days == 0:
                return 0.0
            
            flow_rate = len(knowledge_interactions) / total_days
            return min(1.0, flow_rate / 10.0)  # Normalize
            
        except Exception as e:
            logger.error(f"Error calculating knowledge flow rate: {str(e)}")
            return 0.0
    
    def _calculate_collaboration_effectiveness(self, interaction_data: List[Dict]) -> float:
        """Calculate collaboration effectiveness"""
        try:
            if not interaction_data:
                return 0.0
            
            # Analyze collaboration patterns
            collaboration_interactions = [
                interaction for interaction in interaction_data
                if interaction.get('interaction_type') in ['collaboration', 'peer_learning']
            ]
            
            if not collaboration_interactions:
                return 0.0
            
            # Calculate effectiveness based on outcomes
            successful_collaborations = [
                interaction for interaction in collaboration_interactions
                if interaction.get('outcome_quality', 0) > 0.7
            ]
            
            effectiveness = len(successful_collaborations) / len(collaboration_interactions)
            return effectiveness
            
        except Exception as e:
            logger.error(f"Error calculating collaboration effectiveness: {str(e)}")
            return 0.0
    
    def _calculate_peer_learning_impact(self, learning_data: List[Dict]) -> float:
        """Calculate peer learning impact on learning outcomes"""
        try:
            if not learning_data:
                return 0.0
            
            # Compare learning outcomes with and without peer interaction
            peer_learning_outcomes = []
            individual_learning_outcomes = []
            
            for data in learning_data:
                if data.get('peer_interaction', False):
                    peer_learning_outcomes.append(data.get('learning_outcome', 0))
                else:
                    individual_learning_outcomes.append(data.get('learning_outcome', 0))
            
            if not peer_learning_outcomes or not individual_learning_outcomes:
                return 0.0
            
            avg_peer_outcome = np.mean(peer_learning_outcomes)
            avg_individual_outcome = np.mean(individual_learning_outcomes)
            
            # Calculate impact as improvement ratio
            if avg_individual_outcome > 0:
                impact = (avg_peer_outcome - avg_individual_outcome) / avg_individual_outcome
            else:
                impact = 0.0
            
            return max(0.0, min(1.0, impact))
            
        except Exception as e:
            logger.error(f"Error calculating peer learning impact: {str(e)}")
            return 0.0
    
    def _calculate_social_engagement_level(self, interaction_data: List[Dict]) -> float:
        """Calculate social engagement level"""
        try:
            if not interaction_data:
                return 0.0
            
            # Count different types of social interactions
            interaction_types = [interaction.get('interaction_type', '') for interaction in interaction_data]
            interaction_counts = Counter(interaction_types)
            
            # Weight different interaction types
            weights = {
                'discussion': 1.0,
                'collaboration': 1.2,
                'peer_learning': 1.5,
                'knowledge_sharing': 1.3,
                'help_providing': 1.1,
                'help_seeking': 0.8
            }
            
            weighted_engagement = sum(
                interaction_counts.get(interaction_type, 0) * weight
                for interaction_type, weight in weights.items()
            )
            
            # Normalize by time span
            time_span = self._calculate_time_span(interaction_data)
            if time_span == 0:
                return 0.0
            
            engagement_level = weighted_engagement / time_span
            return min(1.0, engagement_level / 5.0)  # Normalize
            
        except Exception as e:
            logger.error(f"Error calculating social engagement level: {str(e)}")
            return 0.0
    
    def _calculate_community_cohesion(self, network: nx.Graph) -> float:
        """Calculate community cohesion"""
        try:
            if len(network.nodes()) < 2:
                return 0.0
            
            # Use clustering coefficient as cohesion measure
            clustering_coefficient = nx.average_clustering(network)
            
            # Also consider network density
            density = nx.density(network)
            
            # Combined cohesion measure
            cohesion = (clustering_coefficient * 0.7 + density * 0.3)
            
            return cohesion
            
        except Exception as e:
            logger.error(f"Error calculating community cohesion: {str(e)}")
            return 0.0
    
    def _calculate_learning_acceleration(self, learning_data: List[Dict]) -> float:
        """Calculate learning acceleration from social interactions"""
        try:
            if not learning_data:
                return 0.0
            
            # Group learning data by time periods
            time_periods = defaultdict(list)
            for data in learning_data:
                period = data.get('time_period', 0)
                time_periods[period].append(data)
            
            if len(time_periods) < 2:
                return 0.0
            
            # Calculate learning progress over time
            progress_rates = []
            for period in sorted(time_periods.keys()):
                period_data = time_periods[period]
                avg_outcome = np.mean([d.get('learning_outcome', 0) for d in period_data])
                progress_rates.append(avg_outcome)
            
            # Calculate acceleration (second derivative)
            if len(progress_rates) >= 3:
                acceleration = np.diff(progress_rates, 2)
                avg_acceleration = np.mean(acceleration)
            else:
                avg_acceleration = 0.0
            
            return max(0.0, min(1.0, avg_acceleration))
            
        except Exception as e:
            logger.error(f"Error calculating learning acceleration: {str(e)}")
            return 0.0
    
    def _calculate_user_collaboration_score(self, user_id: str, interaction_data: List[Dict]) -> float:
        """Calculate collaboration score for individual user"""
        try:
            user_interactions = [
                interaction for interaction in interaction_data
                if interaction.get('user1_id') == user_id or interaction.get('user2_id') == user_id
            ]
            
            if not user_interactions:
                return 0.0
            
            # Count collaboration interactions
            collaboration_count = sum(
                1 for interaction in user_interactions
                if interaction.get('interaction_type') in ['collaboration', 'peer_learning']
            )
            
            # Calculate score
            total_interactions = len(user_interactions)
            collaboration_score = collaboration_count / total_interactions
            
            return collaboration_score
            
        except Exception as e:
            logger.error(f"Error calculating user collaboration score: {str(e)}")
            return 0.0
    
    def _calculate_peer_influence_score(self, user_id: str, interaction_data: List[Dict]) -> float:
        """Calculate peer influence score for user"""
        try:
            # Count how often user's interactions lead to positive outcomes
            user_interactions = [
                interaction for interaction in interaction_data
                if interaction.get('user1_id') == user_id or interaction.get('user2_id') == user_id
            ]
            
            if not user_interactions:
                return 0.0
            
            # Calculate influence based on outcome quality
            outcome_qualities = [
                interaction.get('outcome_quality', 0) for interaction in user_interactions
            ]
            
            influence_score = np.mean(outcome_qualities)
            return influence_score
            
        except Exception as e:
            logger.error(f"Error calculating peer influence score: {str(e)}")
            return 0.0
    
    def _calculate_learning_centrality(self, user_id: str, interaction_data: List[Dict]) -> float:
        """Calculate learning centrality for user"""
        try:
            # Build user's local network
            user_network = nx.Graph()
            
            for interaction in interaction_data:
                user1 = interaction.get('user1_id')
                user2 = interaction.get('user2_id')
                
                if user1 == user_id or user2 == user_id:
                    if user1 != user2:  # Don't add self-loops
                        user_network.add_edge(user1, user2)
            
            if len(user_network.nodes()) < 2:
                return 0.0
            
            # Calculate centrality
            centrality = nx.degree_centrality(user_network).get(user_id, 0)
            return centrality
            
        except Exception as e:
            logger.error(f"Error calculating learning centrality: {str(e)}")
            return 0.0
    
    def _calculate_knowledge_sharing_score(self, user_id: str, interaction_data: List[Dict]) -> float:
        """Calculate knowledge sharing score for user"""
        try:
            user_interactions = [
                interaction for interaction in interaction_data
                if interaction.get('user1_id') == user_id or interaction.get('user2_id') == user_id
            ]
            
            if not user_interactions:
                return 0.0
            
            # Count knowledge sharing interactions
            knowledge_sharing_count = sum(
                1 for interaction in user_interactions
                if interaction.get('interaction_type') == 'knowledge_sharing'
            )
            
            # Calculate score
            total_interactions = len(user_interactions)
            sharing_score = knowledge_sharing_count / total_interactions
            
            return sharing_score
            
        except Exception as e:
            logger.error(f"Error calculating knowledge sharing score: {str(e)}")
            return 0.0
    
    def _calculate_social_engagement_score(self, user_id: str, interaction_data: List[Dict]) -> float:
        """Calculate social engagement score for user"""
        try:
            user_interactions = [
                interaction for interaction in interaction_data
                if interaction.get('user1_id') == user_id or interaction.get('user2_id') == user_id
            ]
            
            if not user_interactions:
                return 0.0
            
            # Calculate engagement based on interaction frequency and diversity
            interaction_types = [interaction.get('interaction_type', '') for interaction in user_interactions]
            type_diversity = len(set(interaction_types))
            
            # Normalize by time span
            time_span = self._calculate_time_span(user_interactions)
            if time_span == 0:
                return 0.0
            
            frequency_score = len(user_interactions) / time_span
            diversity_score = type_diversity / 10.0  # Normalize
            
            engagement_score = (frequency_score * 0.6 + diversity_score * 0.4)
            return min(1.0, engagement_score)
            
        except Exception as e:
            logger.error(f"Error calculating social engagement score: {str(e)}")
            return 0.0
    
    def _determine_network_position(self, user_id: str, centrality: float, 
                                  collaboration_score: float) -> NetworkPosition:
        """Determine user's network position"""
        try:
            if centrality > 0.7 and collaboration_score > 0.6:
                return NetworkPosition.CENTRAL
            elif centrality > 0.5 and collaboration_score > 0.4:
                return NetworkPosition.BRIDGE
            elif centrality > 0.2 or collaboration_score > 0.2:
                return NetworkPosition.PERIPHERAL
            else:
                return NetworkPosition.ISOLATED
                
        except Exception as e:
            logger.error(f"Error determining network position: {str(e)}")
            return NetworkPosition.PERIPHERAL
    
    def _count_active_connections(self, user_id: str, interaction_data: List[Dict]) -> int:
        """Count active connections for user"""
        try:
            connections = set()
            
            for interaction in interaction_data:
                user1 = interaction.get('user1_id')
                user2 = interaction.get('user2_id')
                
                if user1 == user_id and user2 != user_id:
                    connections.add(user2)
                elif user2 == user_id and user1 != user_id:
                    connections.add(user1)
            
            return len(connections)
            
        except Exception as e:
            logger.error(f"Error counting active connections: {str(e)}")
            return 0
    
    def _calculate_learning_community_size(self, user_id: str, interaction_data: List[Dict]) -> int:
        """Calculate size of user's learning community"""
        try:
            # Find all users connected through learning interactions
            community = {user_id}
            to_explore = {user_id}
            
            while to_explore:
                current_user = to_explore.pop()
                
                for interaction in interaction_data:
                    user1 = interaction.get('user1_id')
                    user2 = interaction.get('user2_id')
                    
                    if current_user == user1 and user2 not in community:
                        community.add(user2)
                        to_explore.add(user2)
                    elif current_user == user2 and user1 not in community:
                        community.add(user1)
                        to_explore.add(user1)
            
            return len(community) - 1  # Exclude the user themselves
            
        except Exception as e:
            logger.error(f"Error calculating learning community size: {str(e)}")
            return 0
    
    def _calculate_contribution_quality(self, user_id: str, interaction_data: List[Dict]) -> float:
        """Calculate contribution quality for user"""
        try:
            user_interactions = [
                interaction for interaction in interaction_data
                if interaction.get('user1_id') == user_id or interaction.get('user2_id') == user_id
            ]
            
            if not user_interactions:
                return 0.0
            
            # Calculate average quality of user's contributions
            qualities = [
                interaction.get('contribution_quality', 0) for interaction in user_interactions
            ]
            
            return np.mean(qualities)
            
        except Exception as e:
            logger.error(f"Error calculating contribution quality: {str(e)}")
            return 0.0
    
    def _calculate_help_seeking_frequency(self, user_id: str, interaction_data: List[Dict]) -> float:
        """Calculate help seeking frequency for user"""
        try:
            help_seeking_interactions = [
                interaction for interaction in interaction_data
                if (interaction.get('user1_id') == user_id and 
                    interaction.get('interaction_type') == 'help_seeking')
            ]
            
            time_span = self._calculate_time_span(interaction_data)
            if time_span == 0:
                return 0.0
            
            frequency = len(help_seeking_interactions) / time_span
            return min(1.0, frequency)
            
        except Exception as e:
            logger.error(f"Error calculating help seeking frequency: {str(e)}")
            return 0.0
    
    def _calculate_help_providing_frequency(self, user_id: str, interaction_data: List[Dict]) -> float:
        """Calculate help providing frequency for user"""
        try:
            help_providing_interactions = [
                interaction for interaction in interaction_data
                if (interaction.get('user1_id') == user_id and 
                    interaction.get('interaction_type') == 'help_providing')
            ]
            
            time_span = self._calculate_time_span(interaction_data)
            if time_span == 0:
                return 0.0
            
            frequency = len(help_providing_interactions) / time_span
            return min(1.0, frequency)
            
        except Exception as e:
            logger.error(f"Error calculating help providing frequency: {str(e)}")
            return 0.0
    
    def _calculate_time_span(self, interaction_data: List[Dict]) -> float:
        """Calculate time span of interactions in days"""
        try:
            if not interaction_data:
                return 0.0
            
            timestamps = [
                interaction.get('timestamp', datetime.now()) for interaction in interaction_data
            ]
            
            if len(timestamps) < 2:
                return 1.0
            
            min_time = min(timestamps)
            max_time = max(timestamps)
            
            time_span = (max_time - min_time).total_seconds() / (24 * 3600)  # Convert to days
            return max(1.0, time_span)  # Minimum 1 day
            
        except Exception as e:
            logger.error(f"Error calculating time span: {str(e)}")
            return 1.0
    
    def _generate_social_learning_recommendations(self, network_density: float,
                                                collaboration_effectiveness: float,
                                                peer_learning_impact: float,
                                                social_engagement_level: float,
                                                community_cohesion: float) -> List[str]:
        """Generate recommendations for improving social learning"""
        try:
            recommendations = []
            
            # Network density recommendations
            if network_density < 0.3:
                recommendations.append("Encourage more connections between learners to increase network density")
            elif network_density > 0.8:
                recommendations.append("Consider creating smaller, focused learning groups to avoid over-connection")
            
            # Collaboration effectiveness recommendations
            if collaboration_effectiveness < 0.5:
                recommendations.append("Provide training on effective collaboration techniques")
                recommendations.append("Implement structured collaboration protocols")
            
            # Peer learning impact recommendations
            if peer_learning_impact < 0.3:
                recommendations.append("Increase opportunities for peer-to-peer learning")
                recommendations.append("Create mentorship programs to facilitate knowledge transfer")
            
            # Social engagement recommendations
            if social_engagement_level < 0.4:
                recommendations.append("Introduce more interactive and social learning activities")
                recommendations.append("Create discussion forums and peer interaction opportunities")
            
            # Community cohesion recommendations
            if community_cohesion < 0.5:
                recommendations.append("Organize community-building activities")
                recommendations.append("Create shared goals and collaborative projects")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Continue current social learning practices")
                recommendations.append("Monitor and adjust social learning strategies as needed")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating social learning recommendations: {str(e)}")
            return ["Monitor social learning patterns and provide support as needed"]
    
    def _get_default_metrics(self) -> SocialLearningMetrics:
        """Return default metrics when analysis fails"""
        return SocialLearningMetrics(
            total_users=0,
            total_connections=0,
            network_density=0.0,
            average_clustering=0.0,
            network_centralization=0.0,
            knowledge_flow_rate=0.0,
            collaboration_effectiveness=0.0,
            peer_learning_impact=0.0,
            social_engagement_level=0.0,
            community_cohesion=0.0,
            learning_acceleration=0.0,
            recommendations=["Insufficient data for analysis"]
        )
    
    def _get_default_profile(self, user_id: str) -> SocialLearningProfile:
        """Return default profile when analysis fails"""
        return SocialLearningProfile(
            user_id=user_id,
            collaboration_score=0.0,
            peer_influence_score=0.0,
            learning_centrality=0.0,
            knowledge_sharing_score=0.0,
            social_engagement_score=0.0,
            network_position=NetworkPosition.ISOLATED,
            active_connections=0,
            learning_community_size=0,
            contribution_quality=0.0,
            help_seeking_frequency=0.0,
            help_providing_frequency=0.0
        )
    
    def get_social_learning_statistics(self) -> Dict[str, float]:
        """Get social learning statistics from historical data"""
        try:
            # This would typically query a database
            # For now, return basic statistics
            return {
                'total_interactions': len(self.interaction_history),
                'active_networks': len(self.learning_networks),
                'collaboration_patterns': len(self.collaboration_patterns),
                'knowledge_sharing_events': len(self.knowledge_sharing_events)
            }
            
        except Exception as e:
            logger.error(f"Error getting social learning statistics: {str(e)}")
            return {
                'total_interactions': 0,
                'active_networks': 0,
                'collaboration_patterns': 0,
                'knowledge_sharing_events': 0
            }
