"""
LearnStyle AI - AI Tutor Integration
Style-aware chat system with context-aware explanations
"""

import os
from typing import Dict, List, Optional
from datetime import datetime
import json
import re

class AITutor:
    """
    AI tutor system that adapts responses based on learning styles
    """
    
    def __init__(self):
        self.style_prompts = {
            'visual': {
                'system_prompt': "You are an AI tutor specialized in visual learning. Use visual metaphors, suggest diagrams, focus on spatial concepts, and describe things in visual terms. Encourage the use of mind maps, charts, and visual aids.",
                'response_style': "visual and spatial"
            },
            'auditory': {
                'system_prompt': "You are an AI tutor specialized in auditory learning. Use storytelling, verbal explanations, focus on rhythm and sound patterns. Encourage discussions, verbal repetition, and explain concepts through analogies and narratives.",
                'response_style': "conversational and verbal"
            },
            'kinesthetic': {
                'system_prompt': "You are an AI tutor specialized in kinesthetic learning. Use hands-on examples, physical metaphors, and practical applications. Focus on learning by doing, experimentation, and real-world problem solving.",
                'response_style': "practical and hands-on"
            },
            'general': {
                'system_prompt': "You are a helpful AI tutor who adapts to different learning styles. Provide clear, comprehensive explanations that can work for various learning preferences.",
                'response_style': "adaptable and comprehensive"
            }
        }
        
        # Common educational topics and responses
        self.topic_responses = {
            'programming': {
                'visual': "Let's break down programming concepts using flowcharts and visual diagrams. Think of code as building blocks where each function is a colored block that connects to others.",
                'auditory': "Programming is like learning a new language with its own grammar and vocabulary. Let me walk you through the concepts step by step in a conversational way.",
                'kinesthetic': "The best way to learn programming is by getting your hands dirty with code. Let's start with a practical project you can build and modify."
            },
            'algorithms': {
                'visual': "Algorithms are like visual recipes with flowcharts showing each step. Imagine sorting algorithms as different ways to organize colored blocks.",
                'auditory': "Think of algorithms as step-by-step instructions you'd give to a friend. Let me explain them like a story with clear beginning, middle, and end.",
                'kinesthetic': "Let's implement these algorithms together and see how they work in practice. We'll build and test each one hands-on."
            },
            'data_structures': {
                'visual': "Data structures are like different types of containers - arrays are like shelves, trees are like family trees, and graphs are like road maps connecting cities.",
                'auditory': "Let me explain data structures through analogies and stories. Each structure has its own personality and use case.",
                'kinesthetic': "We'll build these data structures from scratch and see how they behave with real data. Learning by implementing is the best approach."
            }
        }
    
    def generate_response(self, user_message: str, learning_style: str, 
                         conversation_history: Optional[List[Dict]] = None) -> Dict[str, str]:
        """
        Generate a style-aware response to user message
        
        Args:
            user_message: User's input message
            learning_style: User's dominant learning style
            conversation_history: Previous conversation context
            
        Returns:
            Dictionary with response and metadata
        """
        # Determine the topic of the message
        topic = self._identify_topic(user_message)
        
        # Get style-specific system prompt
        style_config = self.style_prompts.get(learning_style, self.style_prompts['general'])
        
        # Generate contextual response
        response = self._generate_contextual_response(
            user_message, 
            learning_style, 
            topic,
            conversation_history
        )
        
        return {
            'response': response,
            'learning_style': learning_style,
            'topic': topic,
            'timestamp': datetime.utcnow().isoformat(),
            'response_type': self._classify_response_type(user_message)
        }
    
    def _identify_topic(self, message: str) -> str:
        """Identify the main topic of the user's message"""
        message_lower = message.lower()
        
        # Programming-related keywords
        programming_keywords = ['code', 'programming', 'python', 'javascript', 'function', 'variable', 'loop', 'class', 'object']
        if any(keyword in message_lower for keyword in programming_keywords):
            return 'programming'
        
        # Algorithm-related keywords
        algorithm_keywords = ['algorithm', 'sort', 'search', 'complexity', 'big o', 'recursive', 'iteration']
        if any(keyword in message_lower for keyword in algorithm_keywords):
            return 'algorithms'
        
        # Data structure keywords
        data_structure_keywords = ['array', 'list', 'tree', 'graph', 'stack', 'queue', 'hash', 'dictionary']
        if any(keyword in message_lower for keyword in data_structure_keywords):
            return 'data_structures'
        
        # Machine learning keywords
        ml_keywords = ['machine learning', 'ai', 'model', 'training', 'dataset', 'neural network']
        if any(keyword in message_lower for keyword in ml_keywords):
            return 'machine_learning'
        
        return 'general'
    
    def _classify_response_type(self, message: str) -> str:
        """Classify the type of response needed"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['what', 'define', 'explain', 'meaning']):
            return 'explanation'
        elif any(word in message_lower for word in ['how', 'steps', 'process']):
            return 'procedure'
        elif any(word in message_lower for word in ['why', 'reason', 'because']):
            return 'reasoning'
        elif any(word in message_lower for word in ['example', 'show me', 'demonstrate']):
            return 'example'
        elif '?' in message:
            return 'question'
        else:
            return 'general'
    
    def _generate_contextual_response(self, user_message: str, learning_style: str, 
                                    topic: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """Generate a contextual response based on style and topic"""
        
        # Get base response template for the topic and style
        base_response = ""
        if topic in self.topic_responses and learning_style in self.topic_responses[topic]:
            base_response = self.topic_responses[topic][learning_style]
        
        # Generate response based on message type
        response_type = self._classify_response_type(user_message)
        
        if response_type == 'explanation':
            response = self._generate_explanation_response(user_message, learning_style, topic, base_response)
        elif response_type == 'procedure':
            response = self._generate_procedure_response(user_message, learning_style, topic)
        elif response_type == 'example':
            response = self._generate_example_response(user_message, learning_style, topic)
        else:
            response = self._generate_general_response(user_message, learning_style, topic, base_response)
        
        # Add style-specific enhancements
        response = self._enhance_response_for_style(response, learning_style)
        
        return response
    
    def _generate_explanation_response(self, message: str, style: str, topic: str, base_response: str) -> str:
        """Generate explanation-type responses"""
        style_starters = {
            'visual': "Let me paint a picture of this concept for you. ",
            'auditory': "Let me explain this step by step in simple terms. ",
            'kinesthetic': "Let's dive right into this concept with a practical approach. "
        }
        
        starter = style_starters.get(style, "Let me explain this concept. ")
        
        if topic == 'programming':
            return f"{starter}Programming is about giving instructions to computers. {base_response} What specific aspect would you like me to focus on?"
        elif topic == 'algorithms':
            return f"{starter}Algorithms are step-by-step procedures for solving problems. {base_response} Which algorithm are you most curious about?"
        else:
            return f"{starter}{base_response} What would you like to explore further?"
    
    def _generate_procedure_response(self, message: str, style: str, topic: str) -> str:
        """Generate procedure/how-to responses"""
        if style == 'visual':
            return "I'll break this down into visual steps for you. Think of it as a flowchart where each step leads to the next. Would you like me to describe each step in detail?"
        elif style == 'auditory':
            return "Let me walk you through this process step by step, like I'm guiding you through it in person. I'll explain each stage clearly as we go."
        elif style == 'kinesthetic':
            return "The best way to learn this is by doing it. Let's start with a hands-on example where you can practice each step as we go through it."
        else:
            return "I'll guide you through this process step by step. Would you prefer a detailed explanation or a quick overview first?"
    
    def _generate_example_response(self, message: str, style: str, topic: str) -> str:
        """Generate example-focused responses"""
        if style == 'visual':
            return "Great question! Let me show you with a visual example. Imagine we have a diagram that illustrates this concept clearly..."
        elif style == 'auditory':
            return "Perfect! Let me tell you a story that demonstrates this concept in action. This example will help you understand how it works..."
        elif style == 'kinesthetic':
            return "Excellent! Let's work through a real example together where you can see the concept in action and try it yourself..."
        else:
            return "I'd be happy to provide an example. Here's a practical demonstration of this concept..."
    
    def _generate_general_response(self, message: str, style: str, topic: str, base_response: str) -> str:
        """Generate general responses"""
        style_approaches = {
            'visual': "I'll help you visualize this concept clearly. ",
            'auditory': "Let me discuss this with you in a conversational way. ",
            'kinesthetic': "Let's explore this through practical examples. "
        }
        
        approach = style_approaches.get(style, "I'm here to help you understand this. ")
        return f"{approach}{base_response} What specific aspect interests you most?"
    
    def _enhance_response_for_style(self, response: str, learning_style: str) -> str:
        """Add style-specific enhancements to responses"""
        enhancements = {
            'visual': [
                " 🎨 Consider creating a mind map or diagram to visualize this.",
                " 📊 A flowchart might help you see the connections.",
                " 🖼️ Try to picture this concept as a visual representation."
            ],
            'auditory': [
                " 🎵 Try explaining this concept out loud to reinforce learning.",
                " 👥 Discussing this with others can deepen understanding.",
                " 🔊 Consider using voice notes to record key points."
            ],
            'kinesthetic': [
                " ✋ Try implementing this in practice to solidify learning.",
                " 🛠️ Build a small project to apply these concepts.",
                " 🎯 Practice with real examples to master this skill."
            ]
        }
        
        if learning_style in enhancements:
            import random
            enhancement = random.choice(enhancements[learning_style])
            response += enhancement
        
        return response
    
    def get_study_recommendations(self, learning_style: str, topic: str) -> List[str]:
        """Get style-specific study recommendations"""
        recommendations = {
            'visual': [
                "Create mind maps and concept diagrams",
                "Use color coding to organize information",
                "Watch educational videos and visual tutorials",
                "Draw flowcharts to understand processes",
                "Use sticky notes and visual organizers"
            ],
            'auditory': [
                "Listen to educational podcasts",
                "Participate in study groups and discussions",
                "Read material aloud or use text-to-speech",
                "Create audio recordings of key concepts",
                "Teach concepts to others verbally"
            ],
            'kinesthetic': [
                "Practice with hands-on exercises",
                "Build projects to apply concepts",
                "Take frequent breaks and move around",
                "Use physical objects to represent ideas",
                "Learn through trial and error experimentation"
            ]
        }
        
        return recommendations.get(learning_style, [
            "Use multiple learning approaches",
            "Practice regularly with varied exercises",
            "Seek feedback and adjust study methods",
            "Break complex topics into smaller parts"
        ])
    
    def analyze_conversation_patterns(self, conversation_history: List[Dict]) -> Dict:
        """Analyze conversation patterns to improve responses"""
        if not conversation_history:
            return {}
        
        topics = [msg.get('topic', 'general') for msg in conversation_history]
        response_types = [msg.get('response_type', 'general') for msg in conversation_history]
        
        # Count frequency of topics and response types
        topic_counts = {}
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        response_type_counts = {}
        for resp_type in response_types:
            response_type_counts[resp_type] = response_type_counts.get(resp_type, 0) + 1
        
        return {
            'most_discussed_topics': sorted(topic_counts.items(), key=lambda x: x[1], reverse=True),
            'common_question_types': sorted(response_type_counts.items(), key=lambda x: x[1], reverse=True),
            'conversation_length': len(conversation_history)
        }

# Integration with external AI services (for production use)
class OpenAITutorIntegration:
    """
    Integration with OpenAI API for production-grade AI responses
    This would be used in production instead of the rule-based system above
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            import openai
            openai.api_key = api_key
            self.openai = openai
            self.available = True
        except ImportError:
            print("OpenAI package not installed. Install with: pip install openai")
            self.available = False
    
    def generate_response(self, user_message: str, learning_style: str, context: str = "") -> str:
        """Generate response using OpenAI API with style-specific prompts"""
        style_prompts = {
            'visual': "Respond as a visual learning tutor. Use spatial metaphors, suggest diagrams, and focus on visual concepts. ",
            'auditory': "Respond as an auditory learning tutor. Use conversational tone, storytelling, and verbal explanations. ",
            'kinesthetic': "Respond as a kinesthetic learning tutor. Focus on hands-on examples and practical applications. "
        }
        
        system_prompt = style_prompts.get(learning_style, "Respond as a helpful educational tutor. ")
        
        if not self.available:
            return f"AI Tutor ({learning_style}): I understand you're asking about '{user_message}'. Let me help you with that using {learning_style} learning techniques. [OpenAI not available]"
        
        try:
            response = self.openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt + " " + context},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return f"AI Tutor ({learning_style}): I understand you're asking about '{user_message}'. Let me help you with that using {learning_style} learning techniques. [API Error: {str(e)}]"

if __name__ == "__main__":
    # Example usage
    tutor = AITutor()
    
    # Test different learning styles
    test_message = "How do I learn programming?"
    
    for style in ['visual', 'auditory', 'kinesthetic']:
        response = tutor.generate_response(test_message, style)
        print(f"\\n{style.upper()} Response:")
        print(response['response'])
        print(f"Topic: {response['topic']}, Type: {response['response_type']}")
    
    # Test study recommendations
    print("\\nStudy recommendations for visual learners:")
    recommendations = tutor.get_study_recommendations('visual', 'programming')
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")