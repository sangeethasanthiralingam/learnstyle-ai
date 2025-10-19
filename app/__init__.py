from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'  # redirect to login if not logged in

def create_app():
    app = Flask(
        __name__,
        template_folder='../templates',
        static_folder='../static'
    )

    # Config
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change')
    
    # Use MySQL if DATABASE_URL is set, otherwise fallback to SQLite
    database_url = os.environ.get('DATABASE_URL')
    if database_url:
        app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    else:
        # Fallback to SQLite for development
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///learnstyle.db'
    
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)

    # Import and register blueprints
    from app.routes import auth_bp, api_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)

    return app
