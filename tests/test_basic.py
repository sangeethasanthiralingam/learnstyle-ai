"""
Basic tests for LearnStyle AI application
"""

import pytest
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the main app.py file
import importlib.util
spec = importlib.util.spec_from_file_location("main_app", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"))
main_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_app)

from app.models import User, LearningProfile, ContentLibrary

@pytest.fixture
def client():
    """Create a test client"""
    main_app.app.config['TESTING'] = True
    main_app.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with main_app.app.test_client() as client:
        with main_app.app.app_context():
            main_app.db.create_all()
            yield client

def test_home_page(client):
    """Test that the home page loads correctly"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'LearnStyle AI' in response.data

def test_login_page(client):
    """Test that the login page loads correctly"""
    response = client.get('/login')
    assert response.status_code == 200
    assert b'Welcome Back' in response.data

def test_register_page(client):
    """Test that the register page loads correctly"""
    response = client.get('/register')
    assert response.status_code == 200
    assert b'Join LearnStyle AI' in response.data

def test_quiz_page_requires_login(client):
    """Test that quiz page requires authentication"""
    response = client.get('/quiz')
    assert response.status_code == 302  # Redirect to login

def test_dashboard_requires_login(client):
    """Test that dashboard requires authentication"""
    response = client.get('/dashboard')
    assert response.status_code == 302  # Redirect to login

def test_chat_requires_login(client):
    """Test that chat requires authentication"""
    response = client.get('/chat')
    assert response.status_code == 302  # Redirect to login

def test_user_registration(client):
    """Test user registration"""
    response = client.post('/register', data={
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'TestPass123!',
        'confirm_password': 'TestPass123!'
    })
    assert response.status_code == 302  # Redirect after successful registration
    
    # Check if user was created
    with main_app.app.app_context():
        user = User.query.filter_by(username='testuser').first()
        assert user is not None
        assert user.email == 'test@example.com'

def test_user_login(client):
    """Test user login"""
    # First create a user with unique credentials
    with main_app.app.app_context():
        user = User(username='testuser2', email='test2@example.com')
        user.set_password('TestPass123!')
        main_app.db.session.add(user)
        main_app.db.session.commit()
    
    # Test login
    response = client.post('/login', data={
        'username': 'testuser2',
        'password': 'TestPass123!'
    })
    assert response.status_code == 302  # Redirect after successful login

def test_content_library_exists(client):
    """Test that content library has seeded content"""
    with main_app.app.app_context():
        content_count = ContentLibrary.query.count()
        assert content_count > 0
        
        # Check for different content types
        visual_content = ContentLibrary.query.filter(
            ContentLibrary.style_tags.contains('visual')
        ).count()
        assert visual_content > 0
        
        auditory_content = ContentLibrary.query.filter(
            ContentLibrary.style_tags.contains('auditory')
        ).count()
        assert auditory_content > 0
        
        kinesthetic_content = ContentLibrary.query.filter(
            ContentLibrary.style_tags.contains('kinesthetic')
        ).count()
        assert kinesthetic_content > 0

def test_404_error_page(client):
    """Test 404 error page"""
    response = client.get('/nonexistent-page')
    assert response.status_code == 404
    assert b'Page Not Found' in response.data

if __name__ == '__main__':
    pytest.main([__file__])
