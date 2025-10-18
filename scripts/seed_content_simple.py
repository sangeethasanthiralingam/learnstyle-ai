"""
Simple Content Seeding Script for LearnStyle AI
Populates the database with sample educational content
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app and database from the main app.py
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the main app.py file (not the app folder)
import importlib.util
spec = importlib.util.spec_from_file_location("main_app", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"))
main_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_app)

from app.models import ContentLibrary

def seed_content():
    """Seed the database with sample educational content"""
    
    sample_content = [
        # Visual Learning Content
        {
            'title': 'Machine Learning Fundamentals - Visual Guide',
            'description': 'Comprehensive visual introduction to machine learning with interactive diagrams, flowcharts, and infographics.',
            'content_type': 'video',
            'style_tags': 'visual',
            'difficulty_level': 'beginner',
            'url_path': '/content/videos/ml-fundamentals-visual'
        },
        {
            'title': 'Data Visualization with Python',
            'description': 'Learn to create stunning visualizations using matplotlib, seaborn, and plotly with step-by-step visual tutorials.',
            'content_type': 'interactive',
            'style_tags': 'visual',
            'difficulty_level': 'intermediate',
            'url_path': '/content/interactive/data-viz-python'
        },
        {
            'title': 'Neural Network Architecture Diagrams',
            'description': 'Interactive diagrams showing different neural network architectures, activation functions, and data flow.',
            'content_type': 'interactive',
            'style_tags': 'visual',
            'difficulty_level': 'intermediate',
            'url_path': '/content/interactive/neural-network-diagrams'
        },
        {
            'title': 'Statistics Infographic Collection',
            'description': 'Beautiful infographics explaining statistical concepts, probability distributions, and hypothesis testing.',
            'content_type': 'text',
            'style_tags': 'visual',
            'difficulty_level': 'beginner',
            'url_path': '/content/infographics/statistics-collection'
        },
        
        # Auditory Learning Content
        {
            'title': 'Data Science Podcast Series',
            'description': 'Expert discussions on data science trends, techniques, and real-world applications with industry leaders.',
            'content_type': 'audio',
            'style_tags': 'auditory',
            'difficulty_level': 'intermediate',
            'url_path': '/content/audio/data-science-podcast'
        },
        {
            'title': 'Machine Learning Audio Lectures',
            'description': 'Comprehensive audio lectures covering ML algorithms, mathematics, and practical applications.',
            'content_type': 'audio',
            'style_tags': 'auditory',
            'difficulty_level': 'advanced',
            'url_path': '/content/audio/ml-lectures'
        },
        {
            'title': 'Python Programming Audio Guide',
            'description': 'Step-by-step audio instructions for learning Python programming with clear verbal explanations.',
            'content_type': 'audio',
            'style_tags': 'auditory',
            'difficulty_level': 'beginner',
            'url_path': '/content/audio/python-audio-guide'
        },
        {
            'title': 'AI Ethics Discussion Panel',
            'description': 'Roundtable discussion on AI ethics, bias, and responsible AI development with industry experts.',
            'content_type': 'audio',
            'style_tags': 'auditory',
            'difficulty_level': 'intermediate',
            'url_path': '/content/audio/ai-ethics-discussion'
        },
        
        # Kinesthetic Learning Content
        {
            'title': 'Hands-on Python Workshop',
            'description': 'Interactive coding exercises and real projects to master Python programming through practice.',
            'content_type': 'interactive',
            'style_tags': 'kinesthetic',
            'difficulty_level': 'beginner',
            'url_path': '/content/interactive/python-workshop'
        },
        {
            'title': 'Machine Learning Lab Exercises',
            'description': 'Practical ML experiments using Jupyter notebooks with real datasets and hands-on model building.',
            'content_type': 'interactive',
            'style_tags': 'kinesthetic',
            'difficulty_level': 'intermediate',
            'url_path': '/content/interactive/ml-lab-exercises'
        },
        {
            'title': 'Data Analysis Project Challenge',
            'description': 'Complete end-to-end data analysis project with real-world dataset and guided implementation.',
            'content_type': 'interactive',
            'style_tags': 'kinesthetic',
            'difficulty_level': 'advanced',
            'url_path': '/content/interactive/data-analysis-project'
        },
        {
            'title': 'AI Model Building Workshop',
            'description': 'Build and deploy your own AI models from scratch with hands-on coding and experimentation.',
            'content_type': 'interactive',
            'style_tags': 'kinesthetic',
            'difficulty_level': 'advanced',
            'url_path': '/content/interactive/ai-model-workshop'
        },
        
        # Multi-style Content
        {
            'title': 'Complete Data Science Bootcamp',
            'description': 'Comprehensive course combining visual tutorials, audio lectures, and hands-on projects.',
            'content_type': 'video',
            'style_tags': 'visual,auditory,kinesthetic',
            'difficulty_level': 'intermediate',
            'url_path': '/content/videos/data-science-bootcamp'
        },
        {
            'title': 'Interactive Learning Dashboard',
            'description': 'Multi-modal learning platform with visual charts, audio explanations, and interactive exercises.',
            'content_type': 'interactive',
            'style_tags': 'visual,auditory,kinesthetic',
            'difficulty_level': 'beginner',
            'url_path': '/content/interactive/learning-dashboard'
        },
        {
            'title': 'AI Fundamentals Masterclass',
            'description': 'Comprehensive AI course with visual diagrams, audio discussions, and practical coding exercises.',
            'content_type': 'video',
            'style_tags': 'visual,auditory,kinesthetic',
            'difficulty_level': 'intermediate',
            'url_path': '/content/videos/ai-fundamentals-masterclass'
        }
    ]
    
    print("Seeding content library...")
    
    with main_app.app.app_context():
        # Clear existing content
        ContentLibrary.query.delete()
        
        # Add sample content
        for content_data in sample_content:
            content = ContentLibrary(**content_data)
            main_app.db.session.add(content)
        
        main_app.db.session.commit()
        print(f"Successfully seeded {len(sample_content)} content items")

if __name__ == '__main__':
    seed_content()
