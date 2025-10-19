#!/usr/bin/env python3
"""
OpenAI Setup Script for LearnStyle AI
This script helps you set up your OpenAI API key
"""

import os
import sys

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = '.env'
    
    if os.path.exists(env_file):
        print(f"✅ {env_file} already exists")
        return True
    
    # Create .env file from template
    env_content = """# LearnStyle AI Environment Configuration

# Flask Configuration
SECRET_KEY=your-secret-key-here-change-in-production
FLASK_ENV=development
FLASK_DEBUG=True

# Database Configuration
MYSQL_USER=root
MYSQL_PASSWORD=your-mysql-password
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=learnstyle_ai

# AI Integration
OPENAI_API_KEY=your-openai-api-key-here

# Optional: External Services
# REDIS_URL=redis://localhost:6379/0
# CELERY_BROKER_URL=redis://localhost:6379/0

# Email Configuration (Optional)
# MAIL_SERVER=smtp.gmail.com
# MAIL_PORT=587
# MAIL_USE_TLS=True
# MAIL_USERNAME=your-email@gmail.com
# MAIL_PASSWORD=your-app-password

# File Upload Configuration
# MAX_CONTENT_LENGTH=16777216  # 16MB
# UPLOAD_FOLDER=uploads
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Created {env_file} file")
        return True
    except Exception as e:
        print(f"❌ Error creating {env_file}: {e}")
        return False

def check_openai_installation():
    """Check if OpenAI package is installed"""
    try:
        import openai
        print("✅ OpenAI package is installed")
        return True
    except ImportError:
        print("❌ OpenAI package not found")
        print("📦 Install it with: pip install openai")
        return False

def check_api_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    if api_key == 'your-openai-api-key-here':
        print("⚠️  OPENAI_API_KEY is still set to placeholder value")
        return False
    
    if api_key.startswith('sk-'):
        print("✅ OpenAI API key appears to be valid")
        return True
    else:
        print("⚠️  OpenAI API key doesn't start with 'sk-' - please verify")
        return False

def main():
    """Main setup function"""
    print("🚀 LearnStyle AI - OpenAI Setup")
    print("=" * 40)
    
    # Step 1: Create .env file
    print("\n1. Creating .env file...")
    create_env_file()
    
    # Step 2: Check OpenAI installation
    print("\n2. Checking OpenAI package...")
    openai_installed = check_openai_installation()
    
    # Step 3: Check API key
    print("\n3. Checking API key...")
    api_key_valid = check_api_key()
    
    print("\n" + "=" * 40)
    print("📋 Setup Summary:")
    print(f"   .env file: {'✅' if os.path.exists('.env') else '❌'}")
    print(f"   OpenAI package: {'✅' if openai_installed else '❌'}")
    print(f"   API key: {'✅' if api_key_valid else '❌'}")
    
    if not openai_installed:
        print("\n🔧 Next steps:")
        print("   1. Install OpenAI: pip install openai")
        print("   2. Get API key from: https://platform.openai.com/api-keys")
        print("   3. Add API key to .env file")
    elif not api_key_valid:
        print("\n🔧 Next steps:")
        print("   1. Get API key from: https://platform.openai.com/api-keys")
        print("   2. Add API key to .env file: OPENAI_API_KEY=sk-your-key-here")
    else:
        print("\n🎉 Setup complete! Your OpenAI integration is ready.")
        print("   You can now run: python app.py")

if __name__ == "__main__":
    main()
