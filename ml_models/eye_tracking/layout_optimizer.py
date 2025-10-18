"""
Layout Optimization Module

This module provides real-time content layout optimization based on eye-tracking data:
- Dynamic font size adjustment
- Spacing optimization
- Color scheme adaptation
- Content positioning optimization
- Visual hierarchy improvement

Author: LearnStyle AI Team
Version: 1.0.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    """Types of layout optimizations"""
    FONT_SIZE = "font_size"
    LINE_SPACING = "line_spacing"
    MARGIN_SIZE = "margin_size"
    COLOR_CONTRAST = "color_contrast"
    CONTENT_POSITION = "content_position"
    VISUAL_HIERARCHY = "visual_hierarchy"

class OptimizationPriority(Enum):
    """Optimization priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class LayoutOptimization:
    """Individual layout optimization"""
    type: OptimizationType
    priority: OptimizationPriority
    current_value: float
    recommended_value: float
    confidence: float
    expected_improvement: float
    description: str

@dataclass
class LayoutOptimizationResult:
    """Complete layout optimization result"""
    optimizations: List[LayoutOptimization]
    overall_confidence: float
    expected_improvement: float
    implementation_priority: List[OptimizationType]
    css_changes: Dict[str, str]
    recommendations: List[str]

class LayoutOptimizer:
    """
    Real-time layout optimization system based on eye-tracking data
    """
    
    def __init__(self, 
                 min_font_size: int = 12,
                 max_font_size: int = 24,
                 min_line_spacing: float = 1.2,
                 max_line_spacing: float = 2.0,
                 min_margin: int = 10,
                 max_margin: int = 50):
        """
        Initialize layout optimizer
        
        Args:
            min_font_size: Minimum font size in pixels
            max_font_size: Maximum font size in pixels
            min_line_spacing: Minimum line spacing multiplier
            max_line_spacing: Maximum line spacing multiplier
            min_margin: Minimum margin in pixels
            max_margin: Maximum margin in pixels
        """
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.min_line_spacing = min_line_spacing
        self.max_line_spacing = max_line_spacing
        self.min_margin = min_margin
        self.max_margin = max_margin
        
        # Optimization rules and thresholds
        self.optimization_rules = {
            OptimizationType.FONT_SIZE: {
                'squinting_threshold': 0.3,
                'reading_difficulty_threshold': 0.4,
                'improvement_factor': 1.2
            },
            OptimizationType.LINE_SPACING: {
                'crowding_threshold': 0.4,
                'regression_threshold': 5,
                'improvement_factor': 1.1
            },
            OptimizationType.MARGIN_SIZE: {
                'edge_density_threshold': 0.7,
                'focus_scatter_threshold': 0.5,
                'improvement_factor': 0.9
            },
            OptimizationType.COLOR_CONTRAST: {
                'contrast_threshold': 0.6,
                'accessibility_threshold': 0.8,
                'improvement_factor': 1.5
            }
        }
        
        logger.info("Layout Optimizer initialized")
    
    def optimize_layout(self, gaze_analysis: Dict, 
                       current_layout: Dict[str, any],
                       content_metrics: Optional[Dict] = None) -> LayoutOptimizationResult:
        """
        Optimize layout based on gaze analysis
        
        Args:
            gaze_analysis: Gaze analysis results
            current_layout: Current layout configuration
            content_metrics: Content-specific metrics
            
        Returns:
            LayoutOptimizationResult with optimization recommendations
        """
        try:
            optimizations = []
            
            # Font size optimization
            font_optimization = self._optimize_font_size(gaze_analysis, current_layout)
            if font_optimization:
                optimizations.append(font_optimization)
            
            # Line spacing optimization
            spacing_optimization = self._optimize_line_spacing(gaze_analysis, current_layout)
            if spacing_optimization:
                optimizations.append(spacing_optimization)
            
            # Margin optimization
            margin_optimization = self._optimize_margins(gaze_analysis, current_layout)
            if margin_optimization:
                optimizations.append(margin_optimization)
            
            # Color contrast optimization
            contrast_optimization = self._optimize_color_contrast(gaze_analysis, current_layout)
            if contrast_optimization:
                optimizations.append(contrast_optimization)
            
            # Content positioning optimization
            position_optimization = self._optimize_content_position(gaze_analysis, current_layout)
            if position_optimization:
                optimizations.append(position_optimization)
            
            # Visual hierarchy optimization
            hierarchy_optimization = self._optimize_visual_hierarchy(gaze_analysis, current_layout)
            if hierarchy_optimization:
                optimizations.append(hierarchy_optimization)
            
            # Calculate overall metrics
            overall_confidence = self._calculate_overall_confidence(optimizations)
            expected_improvement = self._calculate_expected_improvement(optimizations)
            
            # Generate implementation priority
            implementation_priority = self._generate_implementation_priority(optimizations)
            
            # Generate CSS changes
            css_changes = self._generate_css_changes(optimizations)
            
            # Generate recommendations
            recommendations = self._generate_optimization_recommendations(optimizations)
            
            return LayoutOptimizationResult(
                optimizations=optimizations,
                overall_confidence=overall_confidence,
                expected_improvement=expected_improvement,
                implementation_priority=implementation_priority,
                css_changes=css_changes,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error optimizing layout: {str(e)}")
            return self._get_default_optimization_result()
    
    def _optimize_font_size(self, gaze_analysis: Dict, 
                           current_layout: Dict[str, any]) -> Optional[LayoutOptimization]:
        """
        Optimize font size based on gaze analysis
        
        Args:
            gaze_analysis: Gaze analysis results
            current_layout: Current layout configuration
            
        Returns:
            Font size optimization or None
        """
        try:
            current_font_size = current_layout.get('font_size', 16)
            
            # Check for squinting patterns (frequent small movements)
            squinting_score = self._detect_squinting_pattern(gaze_analysis)
            
            # Check for reading difficulty (regressions, long fixations)
            reading_difficulty = self._assess_reading_difficulty(gaze_analysis)
            
            # Determine if font size optimization is needed
            needs_optimization = (
                squinting_score > self.optimization_rules[OptimizationType.FONT_SIZE]['squinting_threshold'] or
                reading_difficulty > self.optimization_rules[OptimizationType.FONT_SIZE]['reading_difficulty_threshold']
            )
            
            if not needs_optimization:
                return None
            
            # Calculate recommended font size
            improvement_factor = self.optimization_rules[OptimizationType.FONT_SIZE]['improvement_factor']
            recommended_size = min(
                self.max_font_size,
                int(current_font_size * improvement_factor)
            )
            
            # Calculate confidence and expected improvement
            confidence = max(squinting_score, reading_difficulty)
            expected_improvement = min(0.3, (recommended_size - current_font_size) / current_font_size)
            
            # Determine priority
            priority = OptimizationPriority.HIGH if confidence > 0.7 else OptimizationPriority.MEDIUM
            
            return LayoutOptimization(
                type=OptimizationType.FONT_SIZE,
                priority=priority,
                current_value=current_font_size,
                recommended_value=recommended_size,
                confidence=confidence,
                expected_improvement=expected_improvement,
                description=f"Increase font size from {current_font_size}px to {recommended_size}px to improve readability"
            )
            
        except Exception as e:
            logger.error(f"Error optimizing font size: {str(e)}")
            return None
    
    def _optimize_line_spacing(self, gaze_analysis: Dict, 
                              current_layout: Dict[str, any]) -> Optional[LayoutOptimization]:
        """
        Optimize line spacing based on gaze analysis
        
        Args:
            gaze_analysis: Gaze analysis results
            current_layout: Current layout configuration
            
        Returns:
            Line spacing optimization or None
        """
        try:
            current_spacing = current_layout.get('line_spacing', 1.5)
            
            # Check for crowding issues (clustered fixations)
            crowding_score = self._detect_crowding_issues(gaze_analysis)
            
            # Check for reading regressions
            regression_count = gaze_analysis.get('reading_flow', {}).get('regression_count', 0)
            regression_score = min(1.0, regression_count / 10.0)
            
            # Determine if spacing optimization is needed
            needs_optimization = (
                crowding_score > self.optimization_rules[OptimizationType.LINE_SPACING]['crowding_threshold'] or
                regression_score > self.optimization_rules[OptimizationType.LINE_SPACING]['regression_threshold'] / 10.0
            )
            
            if not needs_optimization:
                return None
            
            # Calculate recommended spacing
            improvement_factor = self.optimization_rules[OptimizationType.LINE_SPACING]['improvement_factor']
            recommended_spacing = min(
                self.max_line_spacing,
                current_spacing * improvement_factor
            )
            
            # Calculate confidence and expected improvement
            confidence = max(crowding_score, regression_score)
            expected_improvement = min(0.2, (recommended_spacing - current_spacing) / current_spacing)
            
            # Determine priority
            priority = OptimizationPriority.MEDIUM if confidence > 0.5 else OptimizationPriority.LOW
            
            return LayoutOptimization(
                type=OptimizationType.LINE_SPACING,
                priority=priority,
                current_value=current_spacing,
                recommended_value=recommended_spacing,
                confidence=confidence,
                expected_improvement=expected_improvement,
                description=f"Increase line spacing from {current_spacing} to {recommended_spacing} to reduce crowding"
            )
            
        except Exception as e:
            logger.error(f"Error optimizing line spacing: {str(e)}")
            return None
    
    def _optimize_margins(self, gaze_analysis: Dict, 
                         current_layout: Dict[str, any]) -> Optional[LayoutOptimization]:
        """
        Optimize margins based on gaze analysis
        
        Args:
            gaze_analysis: Gaze analysis results
            current_layout: Current layout configuration
            
        Returns:
            Margin optimization or None
        """
        try:
            current_margin = current_layout.get('margin_size', 20)
            
            # Check for edge density (attention near edges)
            edge_density = self._calculate_edge_density(gaze_analysis)
            
            # Check for focus scatter (attention spread)
            focus_scatter = self._calculate_focus_scatter(gaze_analysis)
            
            # Determine if margin optimization is needed
            needs_optimization = (
                edge_density > self.optimization_rules[OptimizationType.MARGIN_SIZE]['edge_density_threshold'] or
                focus_scatter > self.optimization_rules[OptimizationType.MARGIN_SIZE]['focus_scatter_threshold']
            )
            
            if not needs_optimization:
                return None
            
            # Calculate recommended margin
            improvement_factor = self.optimization_rules[OptimizationType.MARGIN_SIZE]['improvement_factor']
            recommended_margin = max(
                self.min_margin,
                int(current_margin * improvement_factor)
            )
            
            # Calculate confidence and expected improvement
            confidence = max(edge_density, focus_scatter)
            expected_improvement = min(0.15, abs(recommended_margin - current_margin) / current_margin)
            
            # Determine priority
            priority = OptimizationPriority.LOW if confidence > 0.4 else OptimizationPriority.LOW
            
            return LayoutOptimization(
                type=OptimizationType.MARGIN_SIZE,
                priority=priority,
                current_value=current_margin,
                recommended_value=recommended_margin,
                confidence=confidence,
                expected_improvement=expected_improvement,
                description=f"Adjust margin from {current_margin}px to {recommended_margin}px to improve content focus"
            )
            
        except Exception as e:
            logger.error(f"Error optimizing margins: {str(e)}")
            return None
    
    def _optimize_color_contrast(self, gaze_analysis: Dict, 
                                current_layout: Dict[str, any]) -> Optional[LayoutOptimization]:
        """
        Optimize color contrast based on gaze analysis
        
        Args:
            gaze_analysis: Gaze analysis results
            current_layout: Current layout configuration
            
        Returns:
            Color contrast optimization or None
        """
        try:
            current_contrast = current_layout.get('color_contrast', 0.7)
            
            # Check for contrast issues (low attention in text areas)
            contrast_score = self._assess_contrast_issues(gaze_analysis)
            
            # Check for accessibility compliance
            accessibility_score = self._assess_accessibility_compliance(gaze_analysis)
            
            # Determine if contrast optimization is needed
            needs_optimization = (
                contrast_score < self.optimization_rules[OptimizationType.COLOR_CONTRAST]['contrast_threshold'] or
                accessibility_score < self.optimization_rules[OptimizationType.COLOR_CONTRAST]['accessibility_threshold']
            )
            
            if not needs_optimization:
                return None
            
            # Calculate recommended contrast
            improvement_factor = self.optimization_rules[OptimizationType.COLOR_CONTRAST]['improvement_factor']
            recommended_contrast = min(1.0, current_contrast * improvement_factor)
            
            # Calculate confidence and expected improvement
            confidence = max(1 - contrast_score, 1 - accessibility_score)
            expected_improvement = min(0.4, (recommended_contrast - current_contrast) / current_contrast)
            
            # Determine priority
            priority = OptimizationPriority.HIGH if confidence > 0.6 else OptimizationPriority.MEDIUM
            
            return LayoutOptimization(
                type=OptimizationType.COLOR_CONTRAST,
                priority=priority,
                current_value=current_contrast,
                recommended_value=recommended_contrast,
                confidence=confidence,
                expected_improvement=expected_improvement,
                description=f"Increase color contrast from {current_contrast:.2f} to {recommended_contrast:.2f} for better readability"
            )
            
        except Exception as e:
            logger.error(f"Error optimizing color contrast: {str(e)}")
            return None
    
    def _optimize_content_position(self, gaze_analysis: Dict, 
                                  current_layout: Dict[str, any]) -> Optional[LayoutOptimization]:
        """
        Optimize content positioning based on gaze analysis
        
        Args:
            gaze_analysis: Gaze analysis results
            current_layout: Current layout configuration
            
        Returns:
            Content position optimization or None
        """
        try:
            # Analyze attention distribution
            attention_regions = gaze_analysis.get('areas_of_interest', [])
            
            if not attention_regions:
                return None
            
            # Calculate optimal content positioning
            optimal_position = self._calculate_optimal_content_position(attention_regions)
            current_position = current_layout.get('content_position', {'x': 0, 'y': 0})
            
            # Calculate position improvement
            position_improvement = self._calculate_position_improvement(
                current_position, optimal_position
            )
            
            if position_improvement < 0.1:  # No significant improvement needed
                return None
            
            # Calculate confidence
            confidence = min(1.0, len(attention_regions) / 5.0)
            
            # Determine priority
            priority = OptimizationPriority.MEDIUM if confidence > 0.5 else OptimizationPriority.LOW
            
            return LayoutOptimization(
                type=OptimizationType.CONTENT_POSITION,
                priority=priority,
                current_value=0.0,  # Position is complex, use 0 as placeholder
                recommended_value=position_improvement,
                confidence=confidence,
                expected_improvement=position_improvement,
                description=f"Reposition content to align with user attention patterns"
            )
            
        except Exception as e:
            logger.error(f"Error optimizing content position: {str(e)}")
            return None
    
    def _optimize_visual_hierarchy(self, gaze_analysis: Dict, 
                                  current_layout: Dict[str, any]) -> Optional[LayoutOptimization]:
        """
        Optimize visual hierarchy based on gaze analysis
        
        Args:
            gaze_analysis: Gaze analysis results
            current_layout: Current layout configuration
            
        Returns:
            Visual hierarchy optimization or None
        """
        try:
            # Analyze visual hierarchy effectiveness
            hierarchy_effectiveness = gaze_analysis.get('hierarchy_effectiveness', 'fair')
            
            if hierarchy_effectiveness in ['excellent', 'good']:
                return None
            
            # Calculate hierarchy improvement
            hierarchy_scores = {
                'poor': 0.2,
                'fair': 0.4,
                'good': 0.7,
                'excellent': 0.9
            }
            
            current_score = hierarchy_scores.get(hierarchy_effectiveness, 0.4)
            target_score = 0.7  # Aim for 'good' hierarchy
            
            improvement = target_score - current_score
            
            if improvement < 0.1:
                return None
            
            # Calculate confidence
            confidence = improvement
            
            # Determine priority
            priority = OptimizationPriority.HIGH if confidence > 0.3 else OptimizationPriority.MEDIUM
            
            return LayoutOptimization(
                type=OptimizationType.VISUAL_HIERARCHY,
                priority=priority,
                current_value=current_score,
                recommended_value=target_score,
                confidence=confidence,
                expected_improvement=improvement,
                description="Improve visual hierarchy to better guide user attention"
            )
            
        except Exception as e:
            logger.error(f"Error optimizing visual hierarchy: {str(e)}")
            return None
    
    def _detect_squinting_pattern(self, gaze_analysis: Dict) -> float:
        """Detect squinting pattern from gaze data"""
        # Simplified implementation - in practice, this would analyze micro-movements
        fixation_duration = gaze_analysis.get('average_fixation_duration', 0)
        engagement_score = gaze_analysis.get('engagement_score', 0)
        
        # Higher fixation duration with lower engagement suggests squinting
        squinting_score = max(0, (fixation_duration - 300) / 1000) * (1 - engagement_score)
        
        return min(1.0, squinting_score)
    
    def _assess_reading_difficulty(self, gaze_analysis: Dict) -> float:
        """Assess reading difficulty from gaze data"""
        regression_count = gaze_analysis.get('reading_flow', {}).get('regression_count', 0)
        reading_speed = gaze_analysis.get('reading_flow', {}).get('reading_speed', 0)
        
        # Higher regressions and slower reading suggest difficulty
        difficulty_score = min(1.0, regression_count / 10.0 + (1 - min(1.0, reading_speed / 100.0)))
        
        return difficulty_score
    
    def _detect_crowding_issues(self, gaze_analysis: Dict) -> float:
        """Detect crowding issues from gaze data"""
        # Analyze fixation clustering
        fixation_count = gaze_analysis.get('fixation_count', 0)
        scanpath_length = gaze_analysis.get('scanpath_length', 0)
        
        if scanpath_length > 0:
            # High fixation count relative to scanpath length suggests crowding
            crowding_score = min(1.0, fixation_count / (scanpath_length / 100))
        else:
            crowding_score = 0.0
        
        return crowding_score
    
    def _calculate_edge_density(self, gaze_analysis: Dict) -> float:
        """Calculate attention density near edges"""
        # Simplified implementation
        areas_of_interest = gaze_analysis.get('areas_of_interest', [])
        
        if not areas_of_interest:
            return 0.0
        
        # Count areas near edges (within 50 pixels)
        edge_areas = 0
        for area in areas_of_interest:
            if (area['center_x'] < 50 or area['center_x'] > 750 or
                area['center_y'] < 50 or area['center_y'] > 550):
                edge_areas += 1
        
        edge_density = edge_areas / len(areas_of_interest)
        return edge_density
    
    def _calculate_focus_scatter(self, gaze_analysis: Dict) -> float:
        """Calculate focus scatter (how spread out attention is)"""
        areas_of_interest = gaze_analysis.get('areas_of_interest', [])
        
        if len(areas_of_interest) < 2:
            return 0.0
        
        # Calculate spatial spread of attention areas
        x_coords = [area['center_x'] for area in areas_of_interest]
        y_coords = [area['center_y'] for area in areas_of_interest]
        
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        
        # Normalize scatter score
        scatter_score = min(1.0, (x_std + y_std) / 200.0)
        
        return scatter_score
    
    def _assess_contrast_issues(self, gaze_analysis: Dict) -> float:
        """Assess color contrast issues"""
        # Simplified implementation
        engagement_score = gaze_analysis.get('engagement_score', 0)
        attention_level = gaze_analysis.get('attention_level', 'medium')
        
        # Lower engagement with medium attention suggests contrast issues
        attention_scores = {'very_low': 0.1, 'low': 0.3, 'medium': 0.5, 'high': 0.7, 'very_high': 0.9}
        attention_score = attention_scores.get(attention_level, 0.5)
        
        contrast_issues = max(0, (0.5 - engagement_score) + (0.5 - attention_score))
        
        return min(1.0, contrast_issues)
    
    def _assess_accessibility_compliance(self, gaze_analysis: Dict) -> float:
        """Assess accessibility compliance"""
        # Simplified implementation - would check WCAG guidelines
        engagement_score = gaze_analysis.get('engagement_score', 0)
        
        # Higher engagement suggests better accessibility
        return engagement_score
    
    def _calculate_optimal_content_position(self, attention_regions: List[Dict]) -> Dict[str, float]:
        """Calculate optimal content position based on attention regions"""
        if not attention_regions:
            return {'x': 0, 'y': 0}
        
        # Calculate weighted center of attention
        total_weight = sum(region.get('intensity', 1.0) for region in attention_regions)
        
        if total_weight == 0:
            return {'x': 0, 'y': 0}
        
        weighted_x = sum(region['center_x'] * region.get('intensity', 1.0) for region in attention_regions)
        weighted_y = sum(region['center_y'] * region.get('intensity', 1.0) for region in attention_regions)
        
        optimal_x = weighted_x / total_weight
        optimal_y = weighted_y / total_weight
        
        return {'x': optimal_x, 'y': optimal_y}
    
    def _calculate_position_improvement(self, current_position: Dict[str, float], 
                                      optimal_position: Dict[str, float]) -> float:
        """Calculate position improvement score"""
        distance = np.sqrt(
            (optimal_position['x'] - current_position['x'])**2 +
            (optimal_position['y'] - current_position['y'])**2
        )
        
        # Normalize improvement score
        improvement = min(1.0, distance / 200.0)
        
        return improvement
    
    def _calculate_overall_confidence(self, optimizations: List[LayoutOptimization]) -> float:
        """Calculate overall confidence in optimizations"""
        if not optimizations:
            return 0.0
        
        # Weighted average of individual confidences
        total_weight = sum(opt.confidence for opt in optimizations)
        total_confidence = sum(opt.confidence * opt.confidence for opt in optimizations)
        
        if total_weight == 0:
            return 0.0
        
        return total_confidence / total_weight
    
    def _calculate_expected_improvement(self, optimizations: List[LayoutOptimization]) -> float:
        """Calculate expected overall improvement"""
        if not optimizations:
            return 0.0
        
        # Weighted average of expected improvements
        total_weight = sum(opt.confidence for opt in optimizations)
        total_improvement = sum(opt.expected_improvement * opt.confidence for opt in optimizations)
        
        if total_weight == 0:
            return 0.0
        
        return total_improvement / total_weight
    
    def _generate_implementation_priority(self, optimizations: List[LayoutOptimization]) -> List[OptimizationType]:
        """Generate implementation priority order"""
        # Sort by priority and confidence
        sorted_optimizations = sorted(
            optimizations,
            key=lambda opt: (opt.priority.value, -opt.confidence),
            reverse=True
        )
        
        return [opt.type for opt in sorted_optimizations]
    
    def _generate_css_changes(self, optimizations: List[LayoutOptimization]) -> Dict[str, str]:
        """Generate CSS changes for optimizations"""
        css_changes = {}
        
        for opt in optimizations:
            if opt.type == OptimizationType.FONT_SIZE:
                css_changes['font-size'] = f"{opt.recommended_value}px"
            elif opt.type == OptimizationType.LINE_SPACING:
                css_changes['line-height'] = f"{opt.recommended_value}"
            elif opt.type == OptimizationType.MARGIN_SIZE:
                css_changes['margin'] = f"{opt.recommended_value}px"
            elif opt.type == OptimizationType.COLOR_CONTRAST:
                css_changes['filter'] = f"contrast({opt.recommended_value})"
        
        return css_changes
    
    def _generate_optimization_recommendations(self, optimizations: List[LayoutOptimization]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        for opt in optimizations:
            recommendations.append(opt.description)
        
        # Add general recommendations
        if len(optimizations) > 3:
            recommendations.append("Consider implementing optimizations gradually to measure impact")
        
        if any(opt.priority == OptimizationPriority.CRITICAL for opt in optimizations):
            recommendations.append("Critical optimizations should be implemented immediately")
        
        return recommendations
    
    def _get_default_optimization_result(self) -> LayoutOptimizationResult:
        """Return default optimization result when analysis fails"""
        return LayoutOptimizationResult(
            optimizations=[],
            overall_confidence=0.0,
            expected_improvement=0.0,
            implementation_priority=[],
            css_changes={},
            recommendations=["Unable to analyze layout optimization needs"]
        )
