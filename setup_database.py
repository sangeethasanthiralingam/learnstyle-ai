#!/usr/bin/env python3
"""
Database Setup Script for LearnStyle AI
This script creates the MySQL database and initializes all tables.
"""

import os
import sys
import pymysql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_database():
    """Create the MySQL database if it doesn't exist"""
    
    # Get database configuration
    mysql_user = os.environ.get('MYSQL_USER', 'root')
    mysql_password = os.environ.get('MYSQL_PASSWORD', '')
    mysql_host = os.environ.get('MYSQL_HOST', 'localhost')
    mysql_port = int(os.environ.get('MYSQL_PORT', '3306'))
    mysql_database = os.environ.get('MYSQL_DATABASE', 'learnstyle_ai')
    
    print(f"Connecting to MySQL server at {mysql_host}:{mysql_port}")
    print(f"Creating database: {mysql_database}")
    
    try:
        # Connect to MySQL server (without specifying database)
        connection = pymysql.connect(
            host=mysql_host,
            port=mysql_port,
            user=mysql_user,
            password=mysql_password,
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            # Create database if it doesn't exist
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{mysql_database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print(f"✅ Database '{mysql_database}' created successfully")
            
            # Use the database
            cursor.execute(f"USE `{mysql_database}`")
            
            # Create a test table to verify connection
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS `test_connection` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Drop test table
            cursor.execute("DROP TABLE IF EXISTS `test_connection`")
            
        connection.close()
        print("✅ Database setup completed successfully!")
        return True
        
    except pymysql.Error as e:
        print(f"❌ Error creating database: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def create_tables():
    """Create all application tables using Flask-SQLAlchemy"""
    
    print("\nCreating application tables...")
    
    try:
        # Import Flask app and models from the correct path
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Import from the main app.py file (rename to avoid conflict with app directory)
        import importlib.util
        spec = importlib.util.spec_from_file_location("main_app", "app.py")
        main_app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_app)
        from app.models import db
        
        with main_app.app.app_context():
            # Create all tables
            db.create_all()
            print("✅ All tables created successfully!")
            
            # Verify tables were created
            from sqlalchemy import text
            result = db.session.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result]
            
            print(f"\n📋 Created tables:")
            for table in tables:
                print(f"  - {table}")
                
            return True
            
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_admin_user():
    """Create a default admin user"""
    
    print("\nCreating default admin user...")
    
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Import from the main app.py file (rename to avoid conflict with app directory)
        import importlib.util
        spec = importlib.util.spec_from_file_location("main_app", "app.py")
        main_app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_app)
        from app.models import db, User
        
        with main_app.app.app_context():
            # Check if admin user already exists
            admin_user = User.query.filter_by(username='admin').first()
            
            if admin_user:
                print("✅ Admin user already exists")
                return True
            
            # Create admin user
            admin_user = User(
                username='admin',
                email='admin@learnstyle.ai',
                role='admin',
                is_active=True
            )
            admin_user.set_password('admin123')  # Change this in production!
            
            db.session.add(admin_user)
            db.session.commit()
            
            print("✅ Admin user created successfully!")
            print("   Username: admin")
            print("   Password: admin123")
            print("   ⚠️  Please change the password after first login!")
            
            return True
            
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main setup function"""
    
    print("🚀 LearnStyle AI Database Setup")
    print("=" * 40)
    
    # Step 1: Create database
    if not create_database():
        print("\n❌ Database setup failed!")
        sys.exit(1)
    
    # Step 2: Create tables
    if not create_tables():
        print("\n❌ Table creation failed!")
        sys.exit(1)
    
    # Step 3: Create admin user
    if not create_admin_user():
        print("\n❌ Admin user creation failed!")
        sys.exit(1)
    
    print("\n🎉 Database setup completed successfully!")
    print("\nNext steps:")
    print("1. Update your .env file with the correct database credentials")
    print("2. Run: python app.py")
    print("3. Visit: http://localhost:5000")
    print("4. Login with admin/admin123 and change the password")

if __name__ == "__main__":
    main()
