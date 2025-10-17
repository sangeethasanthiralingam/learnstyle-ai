# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.template .env
# Edit .env with your configuration, especially SECRET_KEY
```

### Database Operations
```bash
# Initialize database (creates all tables)
python -c "from app import create_app, db; app = create_app(); app.app_context().push(); db.create_all()"

# Seed sample content (run after database initialization)
python -c "from app.utils.content_manager import seed_sample_content; seed_sample_content()"
```

### Running the Application
```bash
# Run development server
python app.py
# Access at http://localhost:5000

# Production deployment with Gunicorn (install first: pip install gunicorn)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Machine Learning Model Management
```bash
# Train ML model manually (auto-trains on first use if models don't exist)
python ml_models/learning_style_predictor.py
```

### Testing
```bash
# Install test dependencies
pip install pytest pytest-flask

# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_ml_models.py
```

### Code Quality
```bash
# Format code
black app/ ml_models/ tests/

# Lint code
flake8 app/ ml_models/ tests/
```

## Architecture Overview

### Core System Design
LearnStyle AI is a Flask-based adaptive learning system that uses machine learning to detect individual learning styles and delivers personalized educational content. The system follows a modular architecture with clear separation of concerns.

**Key Components:**
- **Machine Learning Engine**: Random Forest classifier predicting learning styles from 15-question assessments
- **Content Recommendation System**: Personalized content delivery based on user learning profiles
- **AI Tutor**: Style-aware conversational interface that adapts responses to learning preferences
- **Progress Tracking**: Comprehensive analytics on user engagement and learning patterns

### Technology Stack
- **Backend**: Flask 2.3.3 with SQLAlchemy for database ORM
- **ML Framework**: scikit-learn with Random Forest and Decision Tree models
- **Database**: SQLite (development), PostgreSQL (production)
- **Frontend**: Bootstrap 5, vanilla JavaScript
- **Authentication**: Flask-Login with Werkzeug password hashing

### Directory Structure Deep Dive

**Application Structure (`app/`):**
- `models/`: SQLAlchemy database models (User, LearningProfile, QuizResponse, ContentLibrary, UserProgress, ChatHistory)
- `routes/`: RESTful API endpoints with authentication and ML integration
- `utils/`: Business logic modules:
  - `ai_tutor.py`: Style-aware AI responses with rule-based conversation system
  - `content_manager.py`: Sophisticated content recommendation algorithm

**Machine Learning (`ml_models/`):**
- `learning_style_predictor.py`: Core ML engine with synthetic data generation and model persistence
- `saved_models/`: Persisted trained models (Random Forest, Decision Tree)
- `training/`: Training data and scripts

### Data Flow Architecture

1. **User Assessment**: 15-question quiz → ML prediction → Learning profile creation
2. **Content Personalization**: Learning profile + engagement history → Content recommendation algorithm → Personalized content feed
3. **AI Tutoring**: User message + learning style context → Style-aware response generation
4. **Continuous Learning**: User interactions → Progress tracking → Profile adaptation

### Machine Learning Implementation

**Learning Style Prediction:**
- **Model**: Random Forest Classifier (primary), Decision Tree Classifier (comparison)
- **Input**: 15 quiz responses (1-3 scale) mapped to VARK learning model
- **Output**: Probability distribution across Visual, Auditory, Kinesthetic styles
- **Training**: Synthetic dataset with realistic learning patterns (1000+ samples)
- **Accuracy**: 85%+ with cross-validation

**Content Recommendation Algorithm:**
1. Primary style matching (50% weight)
2. Engagement pattern analysis (25% weight)
3. Secondary style blending (15% weight)
4. Difficulty adaptation (10% weight)
5. Content diversity injection

### Database Schema Key Relationships

- **User** → **LearningProfile** (1:1): Core user learning characteristics
- **User** → **QuizResponse** (1:many): Historical assessment data
- **User** → **UserProgress** (1:many): Content engagement tracking
- **ContentLibrary** → **UserProgress** (1:many): Content consumption analytics
- **User** → **ChatHistory** (1:many): AI tutor conversation logs

### API Endpoints Structure

**Authentication**: `/register`, `/login`, `/logout`
**Assessment**: `/api/quiz` (POST), `/api/predict` (POST)
**Content**: `/api/content` (GET), `/api/progress` (POST)
**AI Tutor**: `/api/chat` (POST)
**Profile**: `/api/profile` (GET)

### Style-Aware AI Tutor System

The AI tutor uses rule-based responses with style-specific adaptations:
- **Visual**: Emphasizes diagrams, spatial metaphors, color-coding suggestions
- **Auditory**: Uses conversational tone, storytelling, verbal explanations
- **Kinesthetic**: Focuses on hands-on examples, practical applications, experimentation

## Important Configuration

### Environment Variables
Key variables in `.env`:
- `SECRET_KEY`: Flask secret key (change in production)
- `DATABASE_URL`: Database connection string
- `FLASK_ENV`: development/production
- `OPENAI_API_KEY`: Optional for production AI integration

### Model Configuration
ML model parameters in `ml_models/learning_style_predictor.py`:
- `n_estimators=100` for Random Forest
- `n_samples=1000` for synthetic training data generation
- Models auto-save to `ml_models/saved_models/`

### Content Management
Content types: `video`, `audio`, `interactive`, `text`
Learning styles: `visual`, `auditory`, `kinesthetic`
Difficulty levels: `beginner`, `intermediate`, `advanced`

## Development Patterns

### Database Operations
Always use application context for database operations:
```python
with app.app_context():
    db.create_all()
```

### ML Model Integration
Models lazy-load and auto-train on first use. Check model existence before predictions:
```python
try:
    ml_predictor.load_models()
except:
    X, y = ml_predictor.generate_synthetic_dataset(1000)
    ml_predictor.train_models(X, y)
    ml_predictor.save_models()
```

### Content Recommendation
Use the `ContentManager` class for all content operations. It handles personalization logic, engagement analysis, and content filtering automatically.

### Error Handling
The application uses Flask error handlers for 404/500 errors with custom templates in `templates/errors/`.

## Testing Strategy

Tests should be organized in `tests/` with separate modules for:
- Unit tests: Individual component testing
- Integration tests: API endpoint testing
- ML model tests: Algorithm accuracy and performance testing

Use `pytest-flask` for Flask-specific testing utilities and database fixtures.

## Production Considerations

### Deployment
- Use Gunicorn for production WSGI server
- Switch to PostgreSQL for production database
- Set `FLASK_ENV=production` and secure `SECRET_KEY`
- Implement proper logging and monitoring

### Scaling
- ML model retraining should be scheduled (monthly/weekly)
- Content recommendation caching for high-traffic scenarios
- Database indexing on frequently queried fields (user_id, content_type, style_tags)

### Security
- CSRF protection enabled by default
- Password hashing with Werkzeug
- Session management with Flask-Login
- Input validation on all API endpoints

## Integration Notes

### OpenAI Integration
Production AI tutor can integrate with OpenAI API by implementing the `OpenAITutorIntegration` class in `app/utils/ai_tutor.py`. Set `OPENAI_API_KEY` in environment variables.

### Analytics
The system tracks comprehensive user engagement metrics. Implement dashboard visualization using progress and chat history data.

### Content Creation
Use the `ContentManager.add_content()` method to programmatically add educational content. The system supports multiple content types and learning style tags.