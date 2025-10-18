"""
Gaze Pattern Analysis Module

This module provides comprehensive gaze pattern analysis including:
- Fixation detection and clustering
- Saccade analysis
- Attention heatmap generation
- Engagement scoring
- Visual search pattern recognition

Author: LearnStyle AI Team
Version: 1.0.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import spatial
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

class GazeEventType(Enum):
    """Types of gaze events"""
    FIXATION = "fixation"
    SACCADE = "saccade"
    SMOOTH_PURSUIT = "smooth_pursuit"
    BLINK = "blink"

class AttentionLevel(Enum):
    """Attention level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class GazePoint:
    """Individual gaze point data"""
    x: float
    y: float
    timestamp: float
    duration: float
    pupil_diameter: Optional[float] = None
    confidence: Optional[float] = None

@dataclass
class FixationCluster:
    """Fixation cluster data"""
    center_x: float
    center_y: float
    duration: float
    point_count: int
    start_time: float
    end_time: float
    intensity: float

@dataclass
class GazeMetrics:
    """Comprehensive gaze analysis metrics"""
    attention_level: AttentionLevel
    engagement_score: float
    fixation_count: int
    average_fixation_duration: float
    saccade_count: int
    average_saccade_amplitude: float
    scanpath_length: float
    scanpath_efficiency: float
    attention_heatmap: np.ndarray
    areas_of_interest: List[Dict]
    reading_pattern: str
    visual_search_efficiency: float
    confidence: float

class GazeAnalyzer:
    """
    Advanced gaze pattern analysis system
    """
    
    def __init__(self, 
                 fixation_threshold: float = 100,  # milliseconds
                 saccade_threshold: float = 30,    # degrees
                 heatmap_resolution: Tuple[int, int] = (100, 100)):
        """
        Initialize gaze analyzer
        
        Args:
            fixation_threshold: Minimum duration for fixation classification (ms)
            saccade_threshold: Minimum amplitude for saccade classification (degrees)
            heatmap_resolution: Resolution for attention heatmap (width, height)
        """
        self.fixation_threshold = fixation_threshold
        self.saccade_threshold = saccade_threshold
        self.heatmap_resolution = heatmap_resolution
        
        # Attention level thresholds
        self.attention_thresholds = {
            AttentionLevel.VERY_LOW: 0.2,
            AttentionLevel.LOW: 0.4,
            AttentionLevel.MEDIUM: 0.6,
            AttentionLevel.HIGH: 0.8,
            AttentionLevel.VERY_HIGH: 0.9
        }
        
        logger.info("Gaze Analyzer initialized")
    
    def analyze_gaze_patterns(self, gaze_points: List[GazePoint], 
                            content_bounds: Dict[str, float]) -> GazeMetrics:
        """
        Analyze gaze patterns and generate comprehensive metrics
        
        Args:
            gaze_points: List of gaze point data
            content_bounds: Content area bounds {'width': float, 'height': float}
            
        Returns:
            GazeMetrics object with comprehensive analysis
        """
        try:
            if len(gaze_points) < 2:
                return self._get_default_metrics()
            
            # Detect gaze events
            gaze_events = self._detect_gaze_events(gaze_points)
            
            # Analyze fixations
            fixations = self._analyze_fixations(gaze_events, content_bounds)
            
            # Analyze saccades
            saccades = self._analyze_saccades(gaze_events)
            
            # Generate attention heatmap
            heatmap = self._generate_attention_heatmap(gaze_points, content_bounds)
            
            # Identify areas of interest
            areas_of_interest = self._identify_areas_of_interest(heatmap, content_bounds)
            
            # Analyze reading pattern
            reading_pattern = self._analyze_reading_pattern(gaze_points, content_bounds)
            
            # Calculate engagement score
            engagement_score = self._calculate_engagement_score(
                fixations, saccades, heatmap, gaze_points
            )
            
            # Calculate visual search efficiency
            search_efficiency = self._calculate_search_efficiency(gaze_points, areas_of_interest)
            
            # Classify attention level
            attention_level = self._classify_attention_level(engagement_score)
            
            # Calculate confidence
            confidence = self._calculate_confidence(gaze_points, content_bounds)
            
            return GazeMetrics(
                attention_level=attention_level,
                engagement_score=engagement_score,
                fixation_count=len(fixations),
                average_fixation_duration=np.mean([f.duration for f in fixations]) if fixations else 0,
                saccade_count=len(saccades),
                average_saccade_amplitude=np.mean([s['amplitude'] for s in saccades]) if saccades else 0,
                scanpath_length=self._calculate_scanpath_length(gaze_points),
                scanpath_efficiency=self._calculate_scanpath_efficiency(gaze_points, areas_of_interest),
                attention_heatmap=heatmap,
                areas_of_interest=areas_of_interest,
                reading_pattern=reading_pattern,
                visual_search_efficiency=search_efficiency,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing gaze patterns: {str(e)}")
            return self._get_default_metrics()
    
    def _detect_gaze_events(self, gaze_points: List[GazePoint]) -> List[Dict]:
        """
        Detect different types of gaze events
        
        Args:
            gaze_points: List of gaze points
            
        Returns:
            List of gaze events with type and properties
        """
        events = []
        
        for i, point in enumerate(gaze_points):
            event = {
                'type': GazeEventType.FIXATION,
                'x': point.x,
                'y': point.y,
                'timestamp': point.timestamp,
                'duration': point.duration,
                'pupil_diameter': point.pupil_diameter,
                'confidence': point.confidence
            }
            
            # Classify event type based on duration and movement
            if point.duration < self.fixation_threshold:
                if i > 0:
                    prev_point = gaze_points[i-1]
                    distance = np.sqrt((point.x - prev_point.x)**2 + (point.y - prev_point.y)**2)
                    if distance > self.saccade_threshold:
                        event['type'] = GazeEventType.SACCADE
                        event['amplitude'] = distance
                    else:
                        event['type'] = GazeEventType.SMOOTH_PURSUIT
                else:
                    event['type'] = GazeEventType.SACCADE
            
            events.append(event)
        
        return events
    
    def _analyze_fixations(self, gaze_events: List[Dict], 
                          content_bounds: Dict[str, float]) -> List[FixationCluster]:
        """
        Analyze and cluster fixations
        
        Args:
            gaze_events: List of gaze events
            content_bounds: Content area bounds
            
        Returns:
            List of fixation clusters
        """
        # Filter fixation events
        fixation_events = [e for e in gaze_events if e['type'] == GazeEventType.FIXATION]
        
        if not fixation_events:
            return []
        
        # Prepare data for clustering
        fixation_coords = np.array([[e['x'], e['y']] for e in fixation_events])
        
        # Use DBSCAN for fixation clustering
        clustering = DBSCAN(eps=50, min_samples=2).fit(fixation_coords)
        
        clusters = []
        unique_labels = set(clustering.labels_)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
                
            cluster_mask = clustering.labels_ == label
            cluster_events = [fixation_events[i] for i in range(len(fixation_events)) if cluster_mask[i]]
            
            if cluster_events:
                # Calculate cluster properties
                center_x = np.mean([e['x'] for e in cluster_events])
                center_y = np.mean([e['y'] for e in cluster_events])
                duration = sum([e['duration'] for e in cluster_events])
                start_time = min([e['timestamp'] for e in cluster_events])
                end_time = max([e['timestamp'] for e in cluster_events])
                
                # Calculate intensity based on duration and point count
                intensity = duration / (end_time - start_time + 1e-6)
                
                cluster = FixationCluster(
                    center_x=center_x,
                    center_y=center_y,
                    duration=duration,
                    point_count=len(cluster_events),
                    start_time=start_time,
                    end_time=end_time,
                    intensity=intensity
                )
                
                clusters.append(cluster)
        
        return clusters
    
    def _analyze_saccades(self, gaze_events: List[Dict]) -> List[Dict]:
        """
        Analyze saccadic eye movements
        
        Args:
            gaze_events: List of gaze events
            
        Returns:
            List of saccade properties
        """
        saccade_events = [e for e in gaze_events if e['type'] == GazeEventType.SACCADE]
        
        saccades = []
        for event in saccade_events:
            saccade = {
                'amplitude': event.get('amplitude', 0),
                'duration': event['duration'],
                'velocity': event.get('amplitude', 0) / (event['duration'] / 1000.0) if event['duration'] > 0 else 0,
                'start_x': event['x'],
                'start_y': event['y'],
                'timestamp': event['timestamp']
            }
            saccades.append(saccade)
        
        return saccades
    
    def _generate_attention_heatmap(self, gaze_points: List[GazePoint], 
                                  content_bounds: Dict[str, float]) -> np.ndarray:
        """
        Generate attention heatmap from gaze points
        
        Args:
            gaze_points: List of gaze points
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
            x = int((point.x / content_width) * width)
            y = int((point.y / content_height) * height)
            
            # Ensure coordinates are within bounds
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            
            # Weight by duration and confidence
            weight = point.duration * (point.confidence or 1.0)
            heatmap[y, x] += weight
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap
    
    def _identify_areas_of_interest(self, heatmap: np.ndarray, 
                                  content_bounds: Dict[str, float]) -> List[Dict]:
        """
        Identify areas of interest from attention heatmap
        
        Args:
            heatmap: Attention heatmap
            content_bounds: Content area bounds
            
        Returns:
            List of areas of interest with properties
        """
        areas = []
        height, width = heatmap.shape
        
        # Find local maxima in heatmap
        from scipy.ndimage import maximum_filter
        
        local_maxima = maximum_filter(heatmap, size=5) == heatmap
        local_maxima = local_maxima & (heatmap > 0.3)  # Threshold for significant attention
        
        # Get coordinates of local maxima
        y_coords, x_coords = np.where(local_maxima)
        
        for x, y in zip(x_coords, y_coords):
            # Convert back to content coordinates
            content_x = (x / width) * content_bounds['width']
            content_y = (y / height) * content_bounds['height']
            
            # Calculate area properties
            intensity = heatmap[y, x]
            radius = 50  # Approximate radius in pixels
            
            area = {
                'center_x': content_x,
                'center_y': content_y,
                'intensity': float(intensity),
                'radius': radius,
                'type': 'attention_hotspot'
            }
            
            areas.append(area)
        
        return areas
    
    def _analyze_reading_pattern(self, gaze_points: List[GazePoint], 
                               content_bounds: Dict[str, float]) -> str:
        """
        Analyze reading pattern from gaze data
        
        Args:
            gaze_points: List of gaze points
            content_bounds: Content area bounds
            
        Returns:
            Reading pattern classification
        """
        if len(gaze_points) < 3:
            return "insufficient_data"
        
        # Calculate movement vectors
        x_movements = []
        y_movements = []
        
        for i in range(1, len(gaze_points)):
            prev_point = gaze_points[i-1]
            curr_point = gaze_points[i]
            
            x_movements.append(curr_point.x - prev_point.x)
            y_movements.append(curr_point.y - prev_point.y)
        
        # Analyze movement patterns
        avg_x_movement = np.mean(np.abs(x_movements))
        avg_y_movement = np.mean(np.abs(y_movements))
        
        # Determine primary reading direction
        if avg_x_movement > avg_y_movement * 1.5:
            return "horizontal_reading"
        elif avg_y_movement > avg_x_movement * 1.5:
            return "vertical_reading"
        else:
            return "mixed_reading"
    
    def _calculate_engagement_score(self, fixations: List[FixationCluster], 
                                  saccades: List[Dict], heatmap: np.ndarray,
                                  gaze_points: List[GazePoint]) -> float:
        """
        Calculate engagement score from gaze data
        
        Args:
            fixations: List of fixation clusters
            saccades: List of saccade data
            heatmap: Attention heatmap
            gaze_points: Original gaze points
            
        Returns:
            Engagement score (0-1)
        """
        # Factors contributing to engagement
        fixation_density = len(fixations) / max(1, len(gaze_points))
        average_fixation_duration = np.mean([f.duration for f in fixations]) if fixations else 0
        heatmap_entropy = self._calculate_heatmap_entropy(heatmap)
        saccade_frequency = len(saccades) / max(1, len(gaze_points))
        
        # Normalize factors
        fixation_score = min(1.0, fixation_density * 2)
        duration_score = min(1.0, average_fixation_duration / 500.0)  # 500ms baseline
        entropy_score = min(1.0, heatmap_entropy)
        saccade_score = min(1.0, saccade_frequency * 3)
        
        # Weighted combination
        engagement_score = (
            fixation_score * 0.3 +
            duration_score * 0.3 +
            entropy_score * 0.2 +
            saccade_score * 0.2
        )
        
        return max(0, min(1, engagement_score))
    
    def _calculate_heatmap_entropy(self, heatmap: np.ndarray) -> float:
        """Calculate entropy of attention heatmap"""
        # Flatten heatmap and normalize
        flat_heatmap = heatmap.flatten()
        flat_heatmap = flat_heatmap / (np.sum(flat_heatmap) + 1e-10)
        
        # Calculate entropy
        entropy = -np.sum(flat_heatmap * np.log(flat_heatmap + 1e-10))
        
        # Normalize entropy
        max_entropy = np.log(len(flat_heatmap))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def _calculate_search_efficiency(self, gaze_points: List[GazePoint], 
                                   areas_of_interest: List[Dict]) -> float:
        """
        Calculate visual search efficiency
        
        Args:
            gaze_points: List of gaze points
            areas_of_interest: List of areas of interest
            
        Returns:
            Search efficiency score (0-1)
        """
        if not areas_of_interest or len(gaze_points) < 2:
            return 0.5
        
        # Calculate how quickly user found areas of interest
        total_time = gaze_points[-1].timestamp - gaze_points[0].timestamp
        if total_time <= 0:
            return 0.5
        
        # Count how many areas of interest were visited
        visited_areas = 0
        for area in areas_of_interest:
            for point in gaze_points:
                distance = np.sqrt(
                    (point.x - area['center_x'])**2 + 
                    (point.y - area['center_y'])**2
                )
                if distance < area['radius']:
                    visited_areas += 1
                    break
        
        # Calculate efficiency
        efficiency = visited_areas / len(areas_of_interest)
        
        return max(0, min(1, efficiency))
    
    def _calculate_scanpath_length(self, gaze_points: List[GazePoint]) -> float:
        """Calculate total scanpath length"""
        if len(gaze_points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(gaze_points)):
            prev_point = gaze_points[i-1]
            curr_point = gaze_points[i]
            
            distance = np.sqrt(
                (curr_point.x - prev_point.x)**2 + 
                (curr_point.y - prev_point.y)**2
            )
            total_length += distance
        
        return total_length
    
    def _calculate_scanpath_efficiency(self, gaze_points: List[GazePoint], 
                                     areas_of_interest: List[Dict]) -> float:
        """
        Calculate scanpath efficiency
        
        Args:
            gaze_points: List of gaze points
            areas_of_interest: List of areas of interest
            
        Returns:
            Scanpath efficiency (0-1)
        """
        if not areas_of_interest or len(gaze_points) < 2:
            return 0.5
        
        # Calculate optimal path length (straight line to each area)
        optimal_length = 0.0
        for area in areas_of_interest:
            # Find closest gaze point to this area
            min_distance = float('inf')
            for point in gaze_points:
                distance = np.sqrt(
                    (point.x - area['center_x'])**2 + 
                    (point.y - area['center_y'])**2
                )
                min_distance = min(min_distance, distance)
            optimal_length += min_distance
        
        # Calculate actual scanpath length
        actual_length = self._calculate_scanpath_length(gaze_points)
        
        # Efficiency is ratio of optimal to actual
        if actual_length > 0:
            efficiency = optimal_length / actual_length
        else:
            efficiency = 0.5
        
        return max(0, min(1, efficiency))
    
    def _classify_attention_level(self, engagement_score: float) -> AttentionLevel:
        """Classify attention level from engagement score"""
        if engagement_score >= self.attention_thresholds[AttentionLevel.VERY_HIGH]:
            return AttentionLevel.VERY_HIGH
        elif engagement_score >= self.attention_thresholds[AttentionLevel.HIGH]:
            return AttentionLevel.HIGH
        elif engagement_score >= self.attention_thresholds[AttentionLevel.MEDIUM]:
            return AttentionLevel.MEDIUM
        elif engagement_score >= self.attention_thresholds[AttentionLevel.LOW]:
            return AttentionLevel.LOW
        else:
            return AttentionLevel.VERY_LOW
    
    def _calculate_confidence(self, gaze_points: List[GazePoint], 
                            content_bounds: Dict[str, float]) -> float:
        """Calculate confidence in gaze analysis"""
        if not gaze_points:
            return 0.0
        
        # Factors affecting confidence
        data_quality = np.mean([p.confidence or 1.0 for p in gaze_points])
        data_quantity = min(1.0, len(gaze_points) / 100.0)  # 100 points baseline
        coverage = self._calculate_coverage(gaze_points, content_bounds)
        
        # Weighted combination
        confidence = (data_quality * 0.4 + data_quantity * 0.3 + coverage * 0.3)
        
        return max(0.1, min(1.0, confidence))
    
    def _calculate_coverage(self, gaze_points: List[GazePoint], 
                          content_bounds: Dict[str, float]) -> float:
        """Calculate how well gaze points cover the content area"""
        if not gaze_points:
            return 0.0
        
        # Create a grid and count covered cells
        grid_size = 20
        grid_x = int(content_bounds['width'] / grid_size)
        grid_y = int(content_bounds['height'] / grid_size)
        
        covered_cells = set()
        for point in gaze_points:
            cell_x = int(point.x / grid_size)
            cell_y = int(point.y / grid_size)
            if 0 <= cell_x < grid_x and 0 <= cell_y < grid_y:
                covered_cells.add((cell_x, cell_y))
        
        total_cells = grid_x * grid_y
        coverage = len(covered_cells) / total_cells if total_cells > 0 else 0.0
        
        return coverage
    
    def _get_default_metrics(self) -> GazeMetrics:
        """Return default metrics when analysis fails"""
        return GazeMetrics(
            attention_level=AttentionLevel.MEDIUM,
            engagement_score=0.5,
            fixation_count=0,
            average_fixation_duration=0.0,
            saccade_count=0,
            average_saccade_amplitude=0.0,
            scanpath_length=0.0,
            scanpath_efficiency=0.5,
            attention_heatmap=np.zeros(self.heatmap_resolution),
            areas_of_interest=[],
            reading_pattern="insufficient_data",
            visual_search_efficiency=0.5,
            confidence=0.1
        )
