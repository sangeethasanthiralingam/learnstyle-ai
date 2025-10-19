"""
AI-Powered Content Generation Pipeline
Dynamic content creation system with style-specific transformations
"""

import os
import json
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import textwrap

class ContentType(Enum):
    """Types of content that can be generated"""
    TEXT = "text"
    VISUAL = "visual"
    AUDIO = "audio"
    INTERACTIVE = "interactive"
    QUIZ = "quiz"
    SUMMARY = "summary"

class ContentStyle(Enum):
    """Learning styles for content adaptation"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    MIXED = "mixed"

@dataclass
class ContentRequest:
    """Request for content generation"""
    topic: str
    learning_style: ContentStyle
    difficulty_level: str
    content_type: ContentType
    user_preferences: Dict
    context: Optional[str] = None

@dataclass
class GeneratedContent:
    """Generated content result"""
    content_id: str
    title: str
    content: str
    content_type: ContentType
    style_tags: List[str]
    difficulty_level: str
    metadata: Dict
    created_at: datetime

class ContentGenerator:
    """
    AI-powered content generation system with style-specific adaptations
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.content_templates = self._load_content_templates()
        self.style_adaptations = self._load_style_adaptations()
    
    def generate_content(self, request: ContentRequest) -> GeneratedContent:
        """Generate content based on request parameters"""
        try:
            # Generate base content
            base_content = self._generate_base_content(request)
            
            if not base_content or 'content' not in base_content:
                raise ValueError("Failed to generate base content")
            
            # Apply style-specific adaptations
            adapted_content = self._apply_style_adaptations(base_content, request.learning_style)
            
            # Create content metadata
            metadata = self._create_content_metadata(request, adapted_content)
            
            # Generate content ID
            content_id = f"gen_{int(datetime.now().timestamp())}"
            
            return GeneratedContent(
                content_id=content_id,
                title=adapted_content.get('title', f'Content about {request.topic}'),
                content=adapted_content.get('content', ''),
                content_type=request.content_type,
                style_tags=[request.learning_style.value],
                difficulty_level=request.difficulty_level,
                metadata=metadata,
                created_at=datetime.now()
            )
        except Exception as e:
            # Fallback content generation
            return GeneratedContent(
                content_id=f"gen_{int(datetime.now().timestamp())}",
                title=f"Content about {request.topic}",
                content=f"This is educational content about {request.topic}. Content generation is being improved.",
                content_type=request.content_type,
                style_tags=[request.learning_style.value],
                difficulty_level=request.difficulty_level,
                metadata={'error': str(e), 'fallback': True},
                created_at=datetime.now()
            )
    
    def _generate_base_content(self, request: ContentRequest) -> Dict:
        """Generate base content using AI or templates"""
        
        if request.content_type == ContentType.TEXT:
            return self._generate_text_content(request)
        elif request.content_type == ContentType.VISUAL:
            return self._generate_visual_content(request)
        elif request.content_type == ContentType.QUIZ:
            return self._generate_quiz_content(request)
        elif request.content_type == ContentType.SUMMARY:
            return self._generate_summary_content(request)
        elif request.content_type == ContentType.INTERACTIVE:
            return self._generate_interactive_content(request)
        else:
            return self._generate_text_content(request)
    
    def _generate_text_content(self, request: ContentRequest) -> Dict:
        """Generate text-based content"""
        
        if self.openai_api_key:
            return self._generate_with_openai(request)
        else:
            return self._generate_with_templates(request)
    
    def _generate_interactive_content(self, request: ContentRequest) -> Dict:
        """Generate interactive content"""
        
        if self.openai_api_key:
            return self._generate_with_openai(request)
        else:
            return self._generate_with_templates(request)
    
    def _generate_visual_content(self, request: ContentRequest) -> Dict:
        """Generate visual content"""
        
        if self.openai_api_key:
            return self._generate_with_openai(request)
        else:
            return self._generate_with_templates(request)
    
    def _generate_quiz_content(self, request: ContentRequest) -> Dict:
        """Generate quiz content"""
        
        if self.openai_api_key:
            return self._generate_with_openai(request)
        else:
            return self._generate_with_templates(request)
    
    def _generate_summary_content(self, request: ContentRequest) -> Dict:
        """Generate summary content"""
        
        if self.openai_api_key:
            return self._generate_with_openai(request)
        else:
            return self._generate_with_templates(request)
    
    def _generate_with_openai(self, request: ContentRequest) -> Dict:
        """Generate content using OpenAI API"""
        
        prompt = self._create_content_prompt(request)
        
        try:
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.openai_api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'gpt-3.5-turbo',
                    'messages': [
                        {
                            'role': 'system',
                            'content': 'You are an expert educational content creator specializing in personalized learning materials.'
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    'max_tokens': 1000,
                    'temperature': 0.7
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                return {
                    'title': f"Learn about {request.topic}",
                    'content': content,
                    'type': 'text'
                }
            else:
                return self._generate_with_templates(request)
                
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._generate_with_templates(request)
    
    def _generate_with_templates(self, request: ContentRequest) -> Dict:
        """Generate content using predefined templates"""
        
        template = self.content_templates.get(request.content_type.value, {})
        topic = request.topic
        difficulty = request.difficulty_level
        
        if request.content_type == ContentType.TEXT:
            content = self._create_text_template(topic, difficulty, template)
        elif request.content_type == ContentType.QUIZ:
            content = self._create_quiz_template(topic, difficulty, template)
        elif request.content_type == ContentType.SUMMARY:
            content = self._create_summary_template(topic, difficulty, template)
        elif request.content_type == ContentType.INTERACTIVE:
            content = self._create_interactive_template(topic, difficulty, template)
        elif request.content_type == ContentType.VISUAL:
            content = self._create_visual_template(topic, difficulty, template)
        else:
            content = self._create_text_template(topic, difficulty, template)
        
        return {
            'title': f"Learn about {topic}",
            'content': content,
            'type': request.content_type.value
        }
    
    def _create_content_prompt(self, request: ContentRequest) -> str:
        """Create prompt for AI content generation"""
        
        style_instructions = self.style_adaptations.get(request.learning_style.value, {})
        
        prompt = f"""
        Create educational content about "{request.topic}" with the following specifications:
        
        Learning Style: {request.learning_style.value}
        Difficulty Level: {request.difficulty_level}
        Content Type: {request.content_type.value}
        
        Style-specific instructions:
        {style_instructions.get('instructions', '')}
        
        Please create engaging, educational content that:
        1. Is appropriate for {request.difficulty_level} level
        2. Matches {request.learning_style.value} learning preferences
        3. Is clear, accurate, and engaging
        4. Includes practical examples when relevant
        
        Context: {request.context or 'General learning context'}
        """
        
        return prompt
    
    def _apply_style_adaptations(self, content: Dict, learning_style: ContentStyle) -> Dict:
        """Apply style-specific adaptations to content"""
        
        adaptations = self.style_adaptations.get(learning_style.value, {})
        adapted_content = content.copy()
        
        if learning_style == ContentStyle.VISUAL:
            adapted_content = self._apply_visual_adaptations(adapted_content, adaptations)
        elif learning_style == ContentStyle.AUDITORY:
            adapted_content = self._apply_auditory_adaptations(adapted_content, adaptations)
        elif learning_style == ContentStyle.KINESTHETIC:
            adapted_content = self._apply_kinesthetic_adaptations(adapted_content, adaptations)
        
        return adapted_content
    
    def _apply_visual_adaptations(self, content: Dict, adaptations: Dict) -> Dict:
        """Apply visual learning adaptations"""
        
        # Add visual elements to text content
        if content['type'] == 'text':
            visual_elements = [
                "📊 Key Points:",
                "📈 Important Concepts:",
                "🎯 Learning Objectives:",
                "💡 Visual Summary:"
            ]
            
            # Wrap content with visual markers
            adapted_text = content['content']
            newline = '\n'
            for i, element in enumerate(visual_elements):
                if i < len(adapted_text.split(newline)):
                    adapted_text = adapted_text.replace(
                        adapted_text.split(newline)[i],
                        f"{element}{newline}{adapted_text.split(newline)[i]}"
                    )
            
            content['content'] = adapted_text
        
        # Add visual metadata
        content['visual_elements'] = {
            'has_diagrams': True,
            'color_coded': True,
            'structured_layout': True
        }
        
        return content
    
    def _apply_auditory_adaptations(self, content: Dict, adaptations: Dict) -> Dict:
        """Apply auditory learning adaptations"""
        
        # Add auditory elements to text content
        if content['type'] == 'text':
            auditory_elements = [
                "🎧 Listen to this concept:",
                "💬 Discussion points:",
                "🗣️ Key takeaways to remember:",
                "🎵 Rhythmic patterns to help memory:"
            ]
            
            # Add auditory markers
            adapted_text = content['content']
            newline = '\n'
            for i, element in enumerate(auditory_elements):
                if i < len(adapted_text.split(newline)):
                    adapted_text = adapted_text.replace(
                        adapted_text.split(newline)[i],
                        f"{element}{newline}{adapted_text.split(newline)[i]}"
                    )
            
            content['content'] = adapted_text
        
        # Add auditory metadata
        content['auditory_elements'] = {
            'has_audio_notes': True,
            'discussion_questions': True,
            'verbal_explanations': True
        }
        
        return content
    
    def _apply_kinesthetic_adaptations(self, content: Dict, adaptations: Dict) -> Dict:
        """Apply kinesthetic learning adaptations"""
        
        # Add hands-on elements to text content
        if content['type'] == 'text':
            kinesthetic_elements = [
                "🛠️ Hands-on activity:",
                "🏗️ Build this concept:",
                "🎯 Practice exercise:",
                "🤲 Interactive demonstration:"
            ]
            
            # Add kinesthetic markers
            adapted_text = content['content']
            newline = '\n'
            for i, element in enumerate(kinesthetic_elements):
                if i < len(adapted_text.split(newline)):
                    adapted_text = adapted_text.replace(
                        adapted_text.split(newline)[i],
                        f"{element}{newline}{adapted_text.split(newline)[i]}"
                    )
            
            content['content'] = adapted_text
        
        # Add kinesthetic metadata
        content['kinesthetic_elements'] = {
            'has_activities': True,
            'interactive_exercises': True,
            'hands_on_projects': True
        }
        
        return content
    
    def _create_text_template(self, topic: str, difficulty: str, template: Dict) -> str:
        """Create text content using templates"""
        
        if difficulty == 'beginner':
            return f"""
# Introduction to {topic}

## What is {topic}?
{topic} is a fundamental concept that forms the basis for understanding more advanced topics.

## Key Concepts
- **Basic Definition**: {topic} can be understood as...
- **Why it matters**: Understanding {topic} helps you...
- **Common examples**: You can see {topic} in action when...

## Learning Objectives
By the end of this content, you will be able to:
1. Define {topic} in your own words
2. Identify examples of {topic} in real life
3. Explain why {topic} is important

## Summary
{topic} is an essential concept that provides the foundation for deeper learning. Remember to practice with real examples to solidify your understanding.
"""
        
        elif difficulty == 'intermediate':
            return f"""
# Understanding {topic} in Depth

## Overview
{topic} represents a more complex concept that builds upon basic understanding and requires deeper analysis.

## Core Principles
- **Advanced Definition**: {topic} involves...
- **Key Components**: The main elements of {topic} include...
- **Interconnections**: {topic} relates to other concepts through...

## Practical Applications
- **Real-world examples**: {topic} is used in...
- **Case studies**: Consider how {topic} applies to...
- **Best practices**: When working with {topic}, remember to...

## Analysis Framework
To fully understand {topic}, consider:
1. The underlying principles
2. How it connects to other concepts
3. Practical implementation strategies
4. Common challenges and solutions

## Next Steps
Continue exploring {topic} through hands-on practice and advanced study materials.
"""
        
        else:  # advanced
            return f"""
# Advanced {topic}: Theory and Practice

## Theoretical Foundation
{topic} represents a sophisticated concept that requires deep theoretical understanding and practical expertise.

## Advanced Concepts
- **Complex Definitions**: {topic} encompasses...
- **Theoretical Frameworks**: Multiple approaches to understanding {topic} include...
- **Research Applications**: Current research in {topic} focuses on...

## Critical Analysis
- **Strengths and Limitations**: {topic} offers advantages in... but has limitations in...
- **Alternative Approaches**: Other perspectives on {topic} include...
- **Future Directions**: Emerging trends in {topic} suggest...

## Expert Insights
- **Professional Applications**: In practice, {topic} is used to...
- **Industry Perspectives**: Different industries approach {topic} by...
- **Innovation Opportunities**: {topic} presents opportunities for...

## Synthesis and Integration
Understanding {topic} requires integrating multiple perspectives and applying critical thinking to complex scenarios.
"""
    
    def _create_quiz_template(self, topic: str, difficulty: str, template: Dict) -> str:
        """Create quiz content using templates"""
        
        questions = []
        
        if difficulty == 'beginner':
            questions = [
                f"What is the basic definition of {topic}?",
                f"Which of the following is an example of {topic}?",
                f"Why is understanding {topic} important?",
                f"What are the key characteristics of {topic}?",
                f"How does {topic} relate to everyday life?"
            ]
        elif difficulty == 'intermediate':
            questions = [
                f"Explain the core principles behind {topic}",
                f"How do the different components of {topic} interact?",
                f"What are the practical applications of {topic}?",
                f"Compare and contrast different approaches to {topic}",
                f"What challenges might arise when working with {topic}?"
            ]
        else:  # advanced
            questions = [
                f"Analyze the theoretical foundations of {topic}",
                f"Critically evaluate the strengths and limitations of {topic}",
                f"How might {topic} evolve in the future?",
                f"What are the implications of recent research on {topic}?",
                f"How would you integrate {topic} with other advanced concepts?"
            ]
        
        newline = '\n'
        quiz_content = f"# {topic} Quiz{newline}{newline}"
        
        for i, question in enumerate(questions, 1):
            quiz_content += f"## Question {i}{newline}{question}{newline}{newline}"
            quiz_content += f"**Answer:** [Your answer here]{newline}{newline}"
            quiz_content += f"---{newline}{newline}"
        
        return quiz_content
    
    def _create_summary_template(self, topic: str, difficulty: str, template: Dict) -> str:
        """Create summary content using templates"""
        
        return f"""
# {topic} Summary

## Key Points
- **Main Concept**: {topic} is...
- **Important Details**: The key aspects include...
- **Practical Value**: Understanding {topic} helps with...

## Quick Reference
- **Definition**: {topic} can be defined as...
- **Examples**: Common examples include...
- **Applications**: Used in situations such as...

## Memory Aids
- **Acronym**: [Create a memorable acronym]
- **Visual Cues**: [Key visual elements to remember]
- **Connections**: [How it relates to other concepts]

## Next Steps
- Review the full content on {topic}
- Practice with examples
- Apply to real-world scenarios
"""
    
    def _create_interactive_template(self, topic: str, difficulty: str, template: Dict) -> str:
        """Create interactive content using templates"""
        
        return f"""
# Interactive {topic} Learning

## Hands-On Activities

### Activity 1: Explore {topic}
**Objective**: Understand {topic} through direct interaction
**Steps**:
1. Identify examples of {topic} in your environment
2. Create a simple model or diagram
3. Test your understanding with a practical exercise

### Activity 2: Build Your Understanding
**Objective**: Apply {topic} concepts practically
**Materials**: [List any needed materials]
**Process**:
1. Start with basic concepts
2. Gradually increase complexity
3. Reflect on what you've learned

### Activity 3: Collaborative Learning
**Objective**: Learn {topic} with others
**Format**: Group discussion or project
**Focus**: Share insights and learn from peers

## Interactive Elements
- **Simulations**: [Interactive simulations related to {topic}]
- **Games**: [Learning games that reinforce {topic}]
- **Projects**: [Hands-on projects involving {topic}]

## Reflection Questions
- What did you learn about {topic}?
- How does this connect to other concepts?
- What questions do you still have?
"""
    
    def _create_visual_template(self, topic: str, difficulty: str, template: Dict) -> str:
        """Create visual content using templates"""
        
        return f"""
# Visual Guide to {topic}

## 📊 Key Concepts Diagram
```
[Visual representation of {topic} concepts]
- Main concept at center
- Related concepts branching out
- Connections and relationships shown
```

## 🎯 Visual Learning Objectives
- **Understand**: Visual representation of {topic}
- **Identify**: Key visual elements and patterns
- **Apply**: Create your own visual models

## 📈 Step-by-Step Visual Process
1. **Introduction**: Visual overview of {topic}
2. **Development**: Detailed visual breakdown
3. **Application**: Visual examples and case studies
4. **Synthesis**: Visual summary and connections

## 🖼️ Visual Elements
- **Charts**: Data visualization for {topic}
- **Diagrams**: Process flows and relationships
- **Infographics**: Key information in visual format
- **Maps**: Conceptual mapping of {topic}

## 🎨 Interactive Visual Activities
- **Create**: Design your own visual representation
- **Analyze**: Study provided visual materials
- **Compare**: Visual comparison of different approaches
- **Synthesize**: Combine multiple visual elements

## 📝 Visual Notes Template
```
[Space for visual note-taking]
- Key concepts
- Important diagrams
- Personal insights
- Questions and connections
```
"""
    
    def _create_content_metadata(self, request: ContentRequest, content: Dict) -> Dict:
        """Create metadata for generated content"""
        
        return {
            'generation_method': 'ai_generated',
            'learning_style': request.learning_style.value,
            'difficulty_level': request.difficulty_level,
            'content_length': len(content['content']),
            'has_visual_elements': content.get('visual_elements', {}).get('has_diagrams', False),
            'has_auditory_elements': content.get('auditory_elements', {}).get('has_audio_notes', False),
            'has_kinesthetic_elements': content.get('kinesthetic_elements', {}).get('has_activities', False),
            'user_preferences': request.user_preferences,
            'generation_timestamp': datetime.now().isoformat()
        }
    
    def _load_content_templates(self) -> Dict:
        """Load content generation templates"""
        return {
            'text': {
                'beginner': 'simple_explanation',
                'intermediate': 'detailed_analysis',
                'advanced': 'expert_insights'
            },
            'visual': {
                'beginner': 'basic_diagrams',
                'intermediate': 'detailed_visualizations',
                'advanced': 'complex_visual_models'
            },
            'quiz': {
                'beginner': 'basic_questions',
                'intermediate': 'analytical_questions',
                'advanced': 'critical_thinking_questions'
            },
            'summary': {
                'beginner': 'key_points',
                'intermediate': 'comprehensive_summary',
                'advanced': 'expert_synthesis'
            },
            'interactive': {
                'beginner': 'guided_activities',
                'intermediate': 'exploratory_projects',
                'advanced': 'independent_research'
            }
        }
    
    def _load_style_adaptations(self) -> Dict:
        """Load style-specific adaptation rules"""
        return {
            'visual': {
                'instructions': 'Use diagrams, charts, visual metaphors, and structured layouts. Include color coding and visual hierarchy.',
                'elements': ['diagrams', 'charts', 'infographics', 'visual_summaries']
            },
            'auditory': {
                'instructions': 'Use storytelling, verbal explanations, discussion questions, and rhythmic patterns. Focus on spoken word and conversation.',
                'elements': ['audio_notes', 'discussions', 'verbal_explanations', 'storytelling']
            },
            'kinesthetic': {
                'instructions': 'Include hands-on activities, interactive exercises, real-world applications, and physical demonstrations.',
                'elements': ['activities', 'exercises', 'projects', 'simulations']
            },
            'mixed': {
                'instructions': 'Combine visual, auditory, and kinesthetic elements to create a comprehensive learning experience.',
                'elements': ['multi_modal', 'comprehensive', 'adaptive']
            }
        }

# Example usage
if __name__ == "__main__":
    # Initialize content generator
    generator = ContentGenerator()
    
    # Create content request
    request = ContentRequest(
        topic="Machine Learning",
        learning_style=ContentStyle.VISUAL,
        difficulty_level="intermediate",
        content_type=ContentType.TEXT,
        user_preferences={'prefers_examples': True, 'likes_diagrams': True}
    )
    
    # Generate content
    content = generator.generate_content(request)
    
    print(f"Generated Content:")
    print(f"Title: {content.title}")
    print(f"Type: {content.content_type.value}")
    print(f"Style: {content.style_tags}")
    print(f"Content Preview: {content.content[:200]}...")
