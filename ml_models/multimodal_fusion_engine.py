"""
Multimodal Learning Style Fusion Engine
Advanced style blending system with dynamic weight adjustment and context-aware adaptation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from enum import Enum

class LearningContext(Enum):
    """Different learning contexts that may affect style preferences"""
    MATHEMATICS = "mathematics"
    LANGUAGES = "languages"
    SCIENCES = "sciences"
    ARTS = "arts"
    HISTORY = "history"
    PROGRAMMING = "programming"
    GENERAL = "general"

@dataclass
class StyleWeights:
    """Dynamic style weights with confidence scores"""
    visual: float
    auditory: float
    kinesthetic: float
    confidence: float
    last_updated: datetime
    context: LearningContext

@dataclass
class EngagementMetrics:
    """Real-time engagement tracking"""
    content_interaction_time: float
    scroll_velocity: float
    click_frequency: float
    pause_duration: float
    completion_rate: float
    timestamp: datetime

class MultimodalFusionEngine:
    """
    Advanced learning style fusion engine that adapts to user behavior
    and provides context-aware style recommendations
    """
    
    def __init__(self):
        self.style_evolution_history = []
        self.engagement_patterns = []
        self.context_adaptations = {}
        self.bayesian_prior = {
            'visual': 0.33,
            'auditory': 0.33,
            'kinesthetic': 0.34
        }
        
    def update_style_weights(self, 
                           user_id: int,
                           engagement_data: EngagementMetrics,
                           content_style: str,
                           performance_score: float,
                           context: LearningContext = LearningContext.GENERAL) -> StyleWeights:
        """
        Dynamically update style weights based on real-time engagement
        
        Args:
            user_id: User identifier
            engagement_data: Real-time engagement metrics
            content_style: Style of content being consumed
            performance_score: User's performance on the content (0-1)
            context: Learning context/subject area
            
        Returns:
            Updated StyleWeights object
        """
        
        # Calculate engagement score (0-1)
        engagement_score = self._calculate_engagement_score(engagement_data)
        
        # Bayesian update based on engagement and performance
        likelihood = self._calculate_style_likelihood(
            content_style, engagement_score, performance_score
        )
        
        # Get current weights or initialize
        current_weights = self._get_current_weights(user_id, context)
        
        # Update weights using Bayesian inference
        updated_weights = self._bayesian_update(
            current_weights, likelihood, engagement_score
        )
        
        # Apply context-specific adjustments
        context_adjusted_weights = self._apply_context_adaptation(
            updated_weights, context
        )
        
        # Normalize weights
        normalized_weights = self._normalize_weights(context_adjusted_weights)
        
        # Create new StyleWeights object
        new_weights = StyleWeights(
            visual=normalized_weights['visual'],
            auditory=normalized_weights['auditory'],
            kinesthetic=normalized_weights['kinesthetic'],
            confidence=self._calculate_confidence(engagement_score, performance_score),
            last_updated=datetime.now(),
            context=context
        )
        
        # Store evolution history
        self._store_evolution(user_id, new_weights, engagement_data)
        
        return new_weights
    
    def _calculate_engagement_score(self, engagement: EngagementMetrics) -> float:
        """Calculate overall engagement score from multiple metrics"""
        
        # Weighted combination of engagement indicators
        weights = {
            'interaction_time': 0.3,
            'completion_rate': 0.25,
            'click_frequency': 0.2,
            'scroll_velocity': 0.15,
            'pause_duration': 0.1
        }
        
        # Normalize each metric to 0-1 scale
        normalized_metrics = {
            'interaction_time': min(engagement.content_interaction_time / 300, 1.0),  # 5 min max
            'completion_rate': engagement.completion_rate,
            'click_frequency': min(engagement.click_frequency / 10, 1.0),  # 10 clicks max
            'scroll_velocity': min(engagement.scroll_velocity / 1000, 1.0),  # 1000px/s max
            'pause_duration': max(0, 1 - engagement.pause_duration / 60)  # 60s pause max
        }
        
        # Calculate weighted score
        engagement_score = sum(
            weights[metric] * normalized_metrics[metric] 
            for metric in weights
        )
        
        return min(max(engagement_score, 0), 1)
    
    def _calculate_style_likelihood(self, 
                                  content_style: str, 
                                  engagement: float, 
                                  performance: float) -> Dict[str, float]:
        """Calculate likelihood of each style based on content interaction"""
        
        # Base likelihood from content style preference
        base_likelihood = {
            'visual': 0.33,
            'auditory': 0.33,
            'kinesthetic': 0.34
        }
        
        # Adjust based on content style
        if content_style == 'visual':
            base_likelihood['visual'] += 0.4
            base_likelihood['auditory'] -= 0.1
            base_likelihood['kinesthetic'] -= 0.1
        elif content_style == 'auditory':
            base_likelihood['auditory'] += 0.4
            base_likelihood['visual'] -= 0.1
            base_likelihood['kinesthetic'] -= 0.1
        elif content_style == 'kinesthetic':
            base_likelihood['kinesthetic'] += 0.4
            base_likelihood['visual'] -= 0.1
            base_likelihood['auditory'] -= 0.1
        
        # Adjust based on engagement and performance
        engagement_factor = engagement * 0.3
        performance_factor = performance * 0.2
        
        for style in base_likelihood:
            base_likelihood[style] += engagement_factor + performance_factor
        
        return base_likelihood
    
    def _bayesian_update(self, 
                        prior: Dict[str, float], 
                        likelihood: Dict[str, float],
                        evidence_strength: float) -> Dict[str, float]:
        """Update prior beliefs using Bayesian inference"""
        
        # Calculate posterior = (likelihood * prior) / evidence
        posterior = {}
        evidence = sum(likelihood[style] * prior[style] for style in prior)
        
        for style in prior:
            posterior[style] = (likelihood[style] * prior[style]) / evidence
        
        # Blend with prior based on evidence strength
        alpha = evidence_strength  # Learning rate
        for style in posterior:
            posterior[style] = alpha * posterior[style] + (1 - alpha) * prior[style]
        
        return posterior
    
    def _apply_context_adaptation(self, 
                                weights: Dict[str, float], 
                                context: LearningContext) -> Dict[str, float]:
        """Apply context-specific style adaptations"""
        
        context_adjustments = {
            LearningContext.MATHEMATICS: {'visual': 0.1, 'kinesthetic': 0.05, 'auditory': -0.15},
            LearningContext.LANGUAGES: {'auditory': 0.15, 'kinesthetic': 0.05, 'visual': -0.1},
            LearningContext.SCIENCES: {'visual': 0.1, 'kinesthetic': 0.1, 'auditory': -0.1},
            LearningContext.ARTS: {'visual': 0.2, 'kinesthetic': 0.1, 'auditory': -0.1},
            LearningContext.HISTORY: {'auditory': 0.1, 'visual': 0.05, 'kinesthetic': -0.05},
            LearningContext.PROGRAMMING: {'kinesthetic': 0.15, 'visual': 0.05, 'auditory': -0.1},
            LearningContext.GENERAL: {'visual': 0, 'auditory': 0, 'kinesthetic': 0}
        }
        
        adjustments = context_adjustments.get(context, context_adjustments[LearningContext.GENERAL])
        
        adapted_weights = {}
        for style in weights:
            adapted_weights[style] = max(0, min(1, weights[style] + adjustments[style]))
        
        return adapted_weights
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0"""
        total = sum(weights.values())
        if total == 0:
            return {'visual': 0.33, 'auditory': 0.33, 'kinesthetic': 0.34}
        
        return {style: weight / total for style, weight in weights.items()}
    
    def _calculate_confidence(self, engagement: float, performance: float) -> float:
        """Calculate confidence in style assessment"""
        # Higher confidence with consistent high engagement and performance
        consistency = 1 - abs(engagement - performance)
        data_quality = (engagement + performance) / 2
        return (consistency + data_quality) / 2
    
    def _get_current_weights(self, user_id: int, context: LearningContext) -> Dict[str, float]:
        """Get current style weights for user and context"""
        # In a real implementation, this would query the database
        # For now, return default weights
        return {
            'visual': 0.33,
            'auditory': 0.33,
            'kinesthetic': 0.34
        }
    
    def _store_evolution(self, user_id: int, weights: StyleWeights, engagement: EngagementMetrics):
        """Store style evolution history for analysis"""
        evolution_record = {
            'user_id': user_id,
            'timestamp': datetime.now(),
            'weights': {
                'visual': weights.visual,
                'auditory': weights.auditory,
                'kinesthetic': weights.kinesthetic
            },
            'confidence': weights.confidence,
            'context': weights.context.value,
            'engagement': {
                'interaction_time': engagement.content_interaction_time,
                'completion_rate': engagement.completion_rate,
                'click_frequency': engagement.click_frequency
            }
        }
        
        self.style_evolution_history.append(evolution_record)
    
    def get_style_evolution_timeline(self, user_id: int, days: int = 30) -> List[Dict]:
        """Get style evolution timeline for visualization"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        user_evolution = [
            record for record in self.style_evolution_history
            if record['user_id'] == user_id and record['timestamp'] >= cutoff_date
        ]
        
        return sorted(user_evolution, key=lambda x: x['timestamp'])
    
    def generate_hybrid_content_recommendation(self, 
                                             weights: StyleWeights,
                                             content_library: List[Dict]) -> List[Dict]:
        """Generate hybrid content recommendations based on style weights"""
        
        recommendations = []
        
        for content in content_library:
            # Calculate content-style affinity
            content_affinity = self._calculate_content_affinity(content, weights)
            
            # Generate hybrid content package
            hybrid_package = self._create_hybrid_package(content, weights, content_affinity)
            
            recommendations.append(hybrid_package)
        
        # Sort by affinity score
        return sorted(recommendations, key=lambda x: x['affinity_score'], reverse=True)
    
    def _calculate_content_affinity(self, content: Dict, weights: StyleWeights) -> float:
        """Calculate how well content matches user's style preferences"""
        
        content_styles = content.get('style_tags', '').split(',')
        content_styles = [style.strip().lower() for style in content_styles]
        
        affinity = 0
        if 'visual' in content_styles:
            affinity += weights.visual
        if 'auditory' in content_styles:
            affinity += weights.auditory
        if 'kinesthetic' in content_styles:
            affinity += weights.kinesthetic
        
        return affinity / len(content_styles) if content_styles else 0
    
    def _create_hybrid_package(self, 
                             base_content: Dict, 
                             weights: StyleWeights,
                             affinity: float) -> Dict:
        """Create hybrid content package with multiple style elements"""
        
        package = base_content.copy()
        package['affinity_score'] = affinity
        package['hybrid_elements'] = []
        
        # Add visual elements if visual weight is high
        if weights.visual > 0.4:
            package['hybrid_elements'].append({
                'type': 'visual_summary',
                'description': 'Key concepts as infographic',
                'weight': weights.visual
            })
        
        # Add auditory elements if auditory weight is high
        if weights.auditory > 0.4:
            package['hybrid_elements'].append({
                'type': 'audio_explanation',
                'description': 'Spoken explanation of concepts',
                'weight': weights.auditory
            })
        
        # Add kinesthetic elements if kinesthetic weight is high
        if weights.kinesthetic > 0.4:
            package['hybrid_elements'].append({
                'type': 'interactive_exercise',
                'description': 'Hands-on practice activity',
                'weight': weights.kinesthetic
            })
        
        return package
    
    def detect_style_evolution_patterns(self, user_id: int) -> Dict:
        """Detect patterns in style evolution over time"""
        
        user_evolution = self.get_style_evolution_timeline(user_id, days=90)
        
        if len(user_evolution) < 3:
            return {'pattern': 'insufficient_data', 'confidence': 0}
        
        # Analyze trends
        visual_trend = self._calculate_trend([r['weights']['visual'] for r in user_evolution])
        auditory_trend = self._calculate_trend([r['weights']['auditory'] for r in user_evolution])
        kinesthetic_trend = self._calculate_trend([r['weights']['kinesthetic'] for r in user_evolution])
        
        # Detect patterns
        patterns = {
            'increasing_visual': visual_trend > 0.1,
            'increasing_auditory': auditory_trend > 0.1,
            'increasing_kinesthetic': kinesthetic_trend > 0.1,
            'style_stabilization': max(abs(visual_trend), abs(auditory_trend), abs(kinesthetic_trend)) < 0.05,
            'context_adaptation': self._detect_context_adaptation(user_evolution)
        }
        
        return {
            'patterns': patterns,
            'trends': {
                'visual': visual_trend,
                'auditory': auditory_trend,
                'kinesthetic': kinesthetic_trend
            },
            'confidence': self._calculate_pattern_confidence(user_evolution)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction and strength"""
        if len(values) < 2:
            return 0
        
        # Simple linear trend calculation
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _detect_context_adaptation(self, evolution: List[Dict]) -> bool:
        """Detect if user adapts style based on learning context"""
        if len(evolution) < 5:
            return False
        
        # Group by context and check for variations
        context_weights = {}
        for record in evolution:
            context = record['context']
            if context not in context_weights:
                context_weights[context] = []
            context_weights[context].append(record['weights'])
        
        # Check for significant variations across contexts
        if len(context_weights) < 2:
            return False
        
        # Calculate variance across contexts
        context_means = {}
        for context, weights_list in context_weights.items():
            context_means[context] = {
                'visual': np.mean([w['visual'] for w in weights_list]),
                'auditory': np.mean([w['auditory'] for w in weights_list]),
                'kinesthetic': np.mean([w['kinesthetic'] for w in weights_list])
            }
        
        # Calculate variance
        visual_vars = [means['visual'] for means in context_means.values()]
        auditory_vars = [means['auditory'] for means in context_means.values()]
        kinesthetic_vars = [means['kinesthetic'] for means in context_means.values()]
        
        total_variance = np.var(visual_vars) + np.var(auditory_vars) + np.var(kinesthetic_vars)
        
        return total_variance > 0.05  # Threshold for context adaptation
    
    def _calculate_pattern_confidence(self, evolution: List[Dict]) -> float:
        """Calculate confidence in detected patterns"""
        if len(evolution) < 3:
            return 0
        
        # Confidence based on data consistency and volume
        data_consistency = 1 - np.std([r['confidence'] for r in evolution])
        data_volume = min(len(evolution) / 30, 1)  # Normalize to 30 data points
        
        return (data_consistency + data_volume) / 2

# Example usage and testing
if __name__ == "__main__":
    # Initialize the fusion engine
    fusion_engine = MultimodalFusionEngine()
    
    # Simulate engagement data
    engagement = EngagementMetrics(
        content_interaction_time=180,  # 3 minutes
        scroll_velocity=500,  # pixels per second
        click_frequency=8,  # clicks per minute
        pause_duration=15,  # seconds
        completion_rate=0.85,
        timestamp=datetime.now()
    )
    
    # Update style weights
    updated_weights = fusion_engine.update_style_weights(
        user_id=1,
        engagement_data=engagement,
        content_style='visual',
        performance_score=0.9,
        context=LearningContext.MATHEMATICS
    )
    
    print(f"Updated Style Weights:")
    print(f"Visual: {updated_weights.visual:.3f}")
    print(f"Auditory: {updated_weights.auditory:.3f}")
    print(f"Kinesthetic: {updated_weights.kinesthetic:.3f}")
    print(f"Confidence: {updated_weights.confidence:.3f}")
    
    # Detect evolution patterns
    patterns = fusion_engine.detect_style_evolution_patterns(1)
    print(f"\nEvolution Patterns: {patterns}")
