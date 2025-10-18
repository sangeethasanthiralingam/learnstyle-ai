"""
Attention Mapping Module

This module provides advanced attention mapping capabilities including:
- Real-time attention heatmap generation
- Attention distribution analysis
- Visual hierarchy effectiveness measurement
- Content engagement scoring
- Attention pattern recognition

Author: LearnStyle AI Team
Version: 1.0.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

class AttentionPattern(Enum):
    """Types of attention patterns"""
    FOCUSED = "focused"              # Concentrated attention
    SCATTERED = "scattered"          # Distributed attention
    LINEAR = "linear"                # Sequential attention
    CLUSTERED = "clustered"          # Grouped attention
    RANDOM = "random"                # Random attention

class VisualHierarchyLevel(Enum):
    """Visual hierarchy effectiveness levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class AttentionRegion:
    """Attention region data"""
    x: float
    y: float
    width: float
    height: float
    intensity: float
    duration: float
    importance: float
    content_type: str

@dataclass
class AttentionMap:
    """Comprehensive attention mapping data"""
    heatmap: np.ndarray
    attention_regions: List[AttentionRegion]
    pattern_type: AttentionPattern
    hierarchy_effectiveness: VisualHierarchyLevel
    engagement_distribution: Dict[str, float]
    visual_flow_score: float
    content_effectiveness: Dict[str, float]
    recommendations: List[str]

class AttentionMapper:
    """
    Advanced attention mapping and analysis system
    """
    
    def __init__(self, 
                 heatmap_resolution: Tuple[int, int] = (200, 200),
                 gaussian_sigma: float = 2.0,
                 attention_threshold: float = 0.3):
        """
        Initialize attention mapper
        
        Args:
            heatmap_resolution: Resolution for attention heatmap
            gaussian_sigma: Sigma for Gaussian smoothing
            attention_threshold: Threshold for attention region detection
        """
        self.heatmap_resolution = heatmap_resolution
        self.gaussian_sigma = gaussian_sigma
        self.attention_threshold = attention_threshold
        
        logger.info("Attention Mapper initialized")
    
    def create_attention_map(self, gaze_points: List[Dict], 
                           content_layout: Dict[str, any],
                           content_bounds: Dict[str, float]) -> AttentionMap:
        """
        Create comprehensive attention map from gaze data
        
        Args:
            gaze_points: List of gaze point dictionaries
            content_layout: Content layout information
            content_bounds: Content area bounds
            
        Returns:
            AttentionMap object with comprehensive analysis
        """
        try:
            # Generate base heatmap
            heatmap = self._generate_base_heatmap(gaze_points, content_bounds)
            
            # Apply Gaussian smoothing
            smoothed_heatmap = gaussian_filter(heatmap, sigma=self.gaussian_sigma)
            
            # Identify attention regions
            attention_regions = self._identify_attention_regions(
                smoothed_heatmap, content_layout, content_bounds
            )
            
            # Analyze attention pattern
            pattern_type = self._analyze_attention_pattern(gaze_points, attention_regions)
            
            # Assess visual hierarchy effectiveness
            hierarchy_effectiveness = self._assess_visual_hierarchy(
                attention_regions, content_layout
            )
            
            # Calculate engagement distribution
            engagement_distribution = self._calculate_engagement_distribution(
                attention_regions, content_layout
            )
            
            # Calculate visual flow score
            visual_flow_score = self._calculate_visual_flow_score(
                gaze_points, attention_regions
            )
            
            # Assess content effectiveness
            content_effectiveness = self._assess_content_effectiveness(
                attention_regions, content_layout
            )
            
            # Generate recommendations
            recommendations = self._generate_attention_recommendations(
                pattern_type, hierarchy_effectiveness, content_effectiveness
            )
            
            return AttentionMap(
                heatmap=smoothed_heatmap,
                attention_regions=attention_regions,
                pattern_type=pattern_type,
                hierarchy_effectiveness=hierarchy_effectiveness,
                engagement_distribution=engagement_distribution,
                visual_flow_score=visual_flow_score,
                content_effectiveness=content_effectiveness,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error creating attention map: {str(e)}")
            return self._get_default_attention_map()
    
    def _generate_base_heatmap(self, gaze_points: List[Dict], 
                             content_bounds: Dict[str, float]) -> np.ndarray:
        """
        Generate base attention heatmap from gaze points
        
        Args:
            gaze_points: List of gaze point data
            content_bounds: Content area bounds
            
        Returns:
            2D numpy array representing attention heatmap
        """
        width, height = self.heatmap_resolution
        content_width = content_bounds['width']
        content_height = content_bounds['height']
        
        # Initialize heatmap
        heatmap = np.zeros((height, width))
        
        for point in gaze_points:
            # Convert to heatmap coordinates
            x = int((point['x'] / content_width) * width)
            y = int((point['y'] / content_height) * height)
            
            # Ensure coordinates are within bounds
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            
            # Weight by duration and confidence
            weight = point.get('duration', 100) * point.get('confidence', 1.0)
            heatmap[y, x] += weight
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap
    
    def _identify_attention_regions(self, heatmap: np.ndarray, 
                                  content_layout: Dict[str, any],
                                  content_bounds: Dict[str, float]) -> List[AttentionRegion]:
        """
        Identify regions of high attention
        
        Args:
            heatmap: Smoothed attention heatmap
            content_layout: Content layout information
            content_bounds: Content area bounds
            
        Returns:
            List of attention regions
        """
        regions = []
        height, width = heatmap.shape
        content_width = content_bounds['width']
        content_height = content_bounds['height']
        
        # Find local maxima above threshold
        from scipy.ndimage import maximum_filter
        
        local_maxima = maximum_filter(heatmap, size=10) == heatmap
        local_maxima = local_maxima & (heatmap > self.attention_threshold)
        
        # Get coordinates of local maxima
        y_coords, x_coords = np.where(local_maxima)
        
        for x, y in zip(x_coords, y_coords):
            # Convert back to content coordinates
            content_x = (x / width) * content_width
            content_y = (y / height) * content_height
            
            # Calculate region properties
            intensity = heatmap[y, x]
            region_width = 100  # Approximate region width
            region_height = 100  # Approximate region height
            
            # Determine content type at this location
            content_type = self._identify_content_type_at_location(
                content_x, content_y, content_layout
            )
            
            # Calculate importance based on intensity and content type
            importance = self._calculate_region_importance(intensity, content_type)
            
            # Estimate duration (simplified)
            duration = intensity * 1000  # Convert to milliseconds
            
            region = AttentionRegion(
                x=content_x,
                y=content_y,
                width=region_width,
                height=region_height,
                intensity=intensity,
                duration=duration,
                importance=importance,
                content_type=content_type
            )
            
            regions.append(region)
        
        return regions
    
    def _identify_content_type_at_location(self, x: float, y: float, 
                                         content_layout: Dict[str, any]) -> str:
        """
        Identify content type at specific location
        
        Args:
            x: X coordinate
            y: Y coordinate
            content_layout: Content layout information
            
        Returns:
            Content type string
        """
        # This is a simplified implementation
        # In practice, you'd have more sophisticated content detection
        
        if 'text_regions' in content_layout:
            for text_region in content_layout['text_regions']:
                if (text_region['x'] <= x <= text_region['x'] + text_region['width'] and
                    text_region['y'] <= y <= text_region['y'] + text_region['height']):
                    return 'text'
        
        if 'image_regions' in content_layout:
            for image_region in content_layout['image_regions']:
                if (image_region['x'] <= x <= image_region['x'] + image_region['width'] and
                    image_region['y'] <= y <= image_region['y'] + image_region['height']):
                    return 'image'
        
        if 'button_regions' in content_layout:
            for button_region in content_layout['button_regions']:
                if (button_region['x'] <= x <= button_region['x'] + button_region['width'] and
                    button_region['y'] <= y <= button_region['y'] + button_region['height']):
                    return 'button'
        
        return 'unknown'
    
    def _calculate_region_importance(self, intensity: float, content_type: str) -> float:
        """
        Calculate importance of attention region
        
        Args:
            intensity: Attention intensity
            content_type: Type of content
            
        Returns:
            Importance score (0-1)
        """
        # Base importance from intensity
        base_importance = intensity
        
        # Adjust based on content type
        content_weights = {
            'text': 0.8,
            'image': 0.6,
            'button': 0.9,
            'unknown': 0.5
        }
        
        content_weight = content_weights.get(content_type, 0.5)
        
        # Combined importance
        importance = base_importance * content_weight
        
        return max(0, min(1, importance))
    
    def _analyze_attention_pattern(self, gaze_points: List[Dict], 
                                 attention_regions: List[AttentionRegion]) -> AttentionPattern:
        """
        Analyze overall attention pattern
        
        Args:
            gaze_points: List of gaze points
            attention_regions: List of attention regions
            
        Returns:
            Attention pattern classification
        """
        if not gaze_points or not attention_regions:
            return AttentionPattern.RANDOM
        
        # Calculate spatial distribution
        region_positions = [(r.x, r.y) for r in attention_regions]
        
        if len(region_positions) < 2:
            return AttentionPattern.FOCUSED
        
        # Calculate spatial spread
        positions = np.array(region_positions)
        x_std = np.std(positions[:, 0])
        y_std = np.std(positions[:, 1])
        spatial_spread = np.sqrt(x_std**2 + y_std**2)
        
        # Calculate temporal distribution
        gaze_times = [p.get('timestamp', 0) for p in gaze_points]
        if len(gaze_times) > 1:
            time_span = max(gaze_times) - min(gaze_times)
            time_distribution = len(gaze_points) / max(time_span, 1)
        else:
            time_distribution = 0
        
        # Classify pattern
        if spatial_spread < 100 and len(attention_regions) < 3:
            return AttentionPattern.FOCUSED
        elif spatial_spread > 300:
            return AttentionPattern.SCATTERED
        elif self._is_linear_pattern(gaze_points):
            return AttentionPattern.LINEAR
        elif len(attention_regions) > 5 and spatial_spread < 200:
            return AttentionPattern.CLUSTERED
        else:
            return AttentionPattern.RANDOM
    
    def _is_linear_pattern(self, gaze_points: List[Dict]) -> bool:
        """
        Check if gaze pattern is linear
        
        Args:
            gaze_points: List of gaze points
            
        Returns:
            True if pattern is linear
        """
        if len(gaze_points) < 3:
            return False
        
        # Extract coordinates
        x_coords = [p['x'] for p in gaze_points]
        y_coords = [p['y'] for p in gaze_points]
        
        # Check for linear trend
        x_correlation = np.corrcoef(range(len(x_coords)), x_coords)[0, 1]
        y_correlation = np.corrcoef(range(len(y_coords)), y_coords)[0, 1]
        
        # Linear if strong correlation in either direction
        return abs(x_correlation) > 0.7 or abs(y_correlation) > 0.7
    
    def _assess_visual_hierarchy(self, attention_regions: List[AttentionRegion], 
                               content_layout: Dict[str, any]) -> VisualHierarchyLevel:
        """
        Assess visual hierarchy effectiveness
        
        Args:
            attention_regions: List of attention regions
            content_layout: Content layout information
            
        Returns:
            Visual hierarchy effectiveness level
        """
        if not attention_regions:
            return VisualHierarchyLevel.POOR
        
        # Calculate attention distribution across content types
        content_attention = {}
        for region in attention_regions:
            content_type = region.content_type
            if content_type not in content_attention:
                content_attention[content_type] = []
            content_attention[content_type].append(region.intensity)
        
        # Calculate hierarchy score
        hierarchy_score = 0.0
        
        # Text should have highest attention
        if 'text' in content_attention:
            text_attention = np.mean(content_attention['text'])
            hierarchy_score += text_attention * 0.4
        
        # Images should have moderate attention
        if 'image' in content_attention:
            image_attention = np.mean(content_attention['image'])
            hierarchy_score += image_attention * 0.3
        
        # Buttons should have high attention when present
        if 'button' in content_attention:
            button_attention = np.mean(content_attention['button'])
            hierarchy_score += button_attention * 0.3
        
        # Classify hierarchy effectiveness
        if hierarchy_score >= 0.8:
            return VisualHierarchyLevel.EXCELLENT
        elif hierarchy_score >= 0.6:
            return VisualHierarchyLevel.GOOD
        elif hierarchy_score >= 0.4:
            return VisualHierarchyLevel.FAIR
        else:
            return VisualHierarchyLevel.POOR
    
    def _calculate_engagement_distribution(self, attention_regions: List[AttentionRegion], 
                                        content_layout: Dict[str, any]) -> Dict[str, float]:
        """
        Calculate engagement distribution across content types
        
        Args:
            attention_regions: List of attention regions
            content_layout: Content layout information
            
        Returns:
            Dictionary of engagement by content type
        """
        content_engagement = {}
        total_engagement = 0.0
        
        for region in attention_regions:
            content_type = region.content_type
            if content_type not in content_engagement:
                content_engagement[content_type] = 0.0
            content_engagement[content_type] += region.intensity
            total_engagement += region.intensity
        
        # Normalize to percentages
        if total_engagement > 0:
            for content_type in content_engagement:
                content_engagement[content_type] = content_engagement[content_type] / total_engagement
        
        return content_engagement
    
    def _calculate_visual_flow_score(self, gaze_points: List[Dict], 
                                   attention_regions: List[AttentionRegion]) -> float:
        """
        Calculate visual flow score
        
        Args:
            gaze_points: List of gaze points
            attention_regions: List of attention regions
            
        Returns:
            Visual flow score (0-1)
        """
        if len(gaze_points) < 2:
            return 0.5
        
        # Calculate gaze path smoothness
        gaze_path = [(p['x'], p['y']) for p in gaze_points]
        path_smoothness = self._calculate_path_smoothness(gaze_path)
        
        # Calculate attention region connectivity
        region_connectivity = self._calculate_region_connectivity(attention_regions)
        
        # Calculate flow efficiency
        flow_efficiency = self._calculate_flow_efficiency(gaze_path, attention_regions)
        
        # Combine scores
        visual_flow_score = (
            path_smoothness * 0.4 +
            region_connectivity * 0.3 +
            flow_efficiency * 0.3
        )
        
        return max(0, min(1, visual_flow_score))
    
    def _calculate_path_smoothness(self, gaze_path: List[Tuple[float, float]]) -> float:
        """Calculate smoothness of gaze path"""
        if len(gaze_path) < 3:
            return 0.5
        
        # Calculate angle changes between consecutive segments
        angles = []
        for i in range(1, len(gaze_path) - 1):
            p1 = gaze_path[i-1]
            p2 = gaze_path[i]
            p3 = gaze_path[i+1]
            
            # Calculate vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Calculate angle between vectors
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        if not angles:
            return 0.5
        
        # Smoothness is inverse of angle variance
        angle_variance = np.var(angles)
        smoothness = 1.0 / (1.0 + angle_variance)
        
        return max(0, min(1, smoothness))
    
    def _calculate_region_connectivity(self, attention_regions: List[AttentionRegion]) -> float:
        """Calculate connectivity between attention regions"""
        if len(attention_regions) < 2:
            return 0.5
        
        # Calculate distances between regions
        region_positions = [(r.x, r.y) for r in attention_regions]
        distances = cdist(region_positions, region_positions)
        
        # Calculate average distance (excluding self-distances)
        mask = distances > 0
        if np.any(mask):
            avg_distance = np.mean(distances[mask])
            # Connectivity is inverse of average distance
            connectivity = 1.0 / (1.0 + avg_distance / 100.0)
        else:
            connectivity = 0.5
        
        return max(0, min(1, connectivity))
    
    def _calculate_flow_efficiency(self, gaze_path: List[Tuple[float, float]], 
                                 attention_regions: List[AttentionRegion]) -> float:
        """Calculate flow efficiency through attention regions"""
        if not attention_regions or len(gaze_path) < 2:
            return 0.5
        
        # Calculate how many regions are visited
        visited_regions = 0
        for region in attention_regions:
            for point in gaze_path:
                distance = np.sqrt(
                    (point[0] - region.x)**2 + (point[1] - region.y)**2
                )
                if distance < 50:  # Within region radius
                    visited_regions += 1
                    break
        
        # Efficiency is ratio of visited to total regions
        efficiency = visited_regions / len(attention_regions)
        
        return max(0, min(1, efficiency))
    
    def _assess_content_effectiveness(self, attention_regions: List[AttentionRegion], 
                                    content_layout: Dict[str, any]) -> Dict[str, float]:
        """
        Assess effectiveness of different content types
        
        Args:
            attention_regions: List of attention regions
            content_layout: Content layout information
            
        Returns:
            Dictionary of effectiveness scores by content type
        """
        effectiveness = {}
        
        # Group regions by content type
        content_regions = {}
        for region in attention_regions:
            content_type = region.content_type
            if content_type not in content_regions:
                content_regions[content_type] = []
            content_regions[content_type].append(region)
        
        # Calculate effectiveness for each content type
        for content_type, regions in content_regions.items():
            if regions:
                # Effectiveness based on attention intensity and importance
                avg_intensity = np.mean([r.intensity for r in regions])
                avg_importance = np.mean([r.importance for r in regions])
                region_count = len(regions)
                
                # Weighted effectiveness score
                effectiveness[content_type] = (
                    avg_intensity * 0.5 +
                    avg_importance * 0.3 +
                    min(1.0, region_count / 5.0) * 0.2
                )
            else:
                effectiveness[content_type] = 0.0
        
        return effectiveness
    
    def _generate_attention_recommendations(self, pattern_type: AttentionPattern,
                                          hierarchy_effectiveness: VisualHierarchyLevel,
                                          content_effectiveness: Dict[str, float]) -> List[str]:
        """
        Generate recommendations based on attention analysis
        
        Args:
            pattern_type: Attention pattern classification
            hierarchy_effectiveness: Visual hierarchy effectiveness
            content_effectiveness: Content effectiveness scores
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Pattern-based recommendations
        if pattern_type == AttentionPattern.SCATTERED:
            recommendations.append("Consider reducing visual clutter to improve focus")
        elif pattern_type == AttentionPattern.RANDOM:
            recommendations.append("Improve visual hierarchy to guide user attention")
        elif pattern_type == AttentionPattern.FOCUSED:
            recommendations.append("Good focus! Consider adding more interactive elements")
        
        # Hierarchy-based recommendations
        if hierarchy_effectiveness == VisualHierarchyLevel.POOR:
            recommendations.append("Improve visual hierarchy - make important content more prominent")
        elif hierarchy_effectiveness == VisualHierarchyLevel.FAIR:
            recommendations.append("Visual hierarchy could be improved with better contrast and sizing")
        
        # Content effectiveness recommendations
        for content_type, effectiveness in content_effectiveness.items():
            if effectiveness < 0.3:
                recommendations.append(f"Improve {content_type} content visibility and engagement")
            elif effectiveness > 0.8:
                recommendations.append(f"Excellent {content_type} content effectiveness!")
        
        return recommendations
    
    def _get_default_attention_map(self) -> AttentionMap:
        """Return default attention map when analysis fails"""
        return AttentionMap(
            heatmap=np.zeros(self.heatmap_resolution),
            attention_regions=[],
            pattern_type=AttentionPattern.RANDOM,
            hierarchy_effectiveness=VisualHierarchyLevel.POOR,
            engagement_distribution={},
            visual_flow_score=0.5,
            content_effectiveness={},
            recommendations=["Unable to analyze attention patterns"]
        )
