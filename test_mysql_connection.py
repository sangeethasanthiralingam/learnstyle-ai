#!/usr/bin/env python3
"""
Test MySQL connection and database setup
"""

import os
import pymysql
from dotenv import load_dotenv

def test_mysql_connection():
    try:
        print("🔍 Testing MySQL connection...")
        
        # Load environment variables
        load_dotenv()
        
        # Get MySQL credentials from environment
        host = os.getenv('MYSQL_HOST', 'localhost')
        user = os.getenv('MYSQL_USER', 'root')
        password = os.getenv('MYSQL_PASSWORD', '')
        database = os.getenv('MYSQL_DATABASE', 'learnstyle_ai')
        port = int(os.getenv('MYSQL_PORT', 3306))
        
        print(f"📊 Connecting to: {user}@{host}:{port}/{database}")
        
        # Test connection
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
            charset='utf8mb4'
        )
        
        cursor = connection.cursor()
        
        # Test basic queries
        print("1️⃣ Testing basic queries...")
        
        # Check if tables exist
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"   📋 Found {len(tables)} tables:")
        for table in tables:
            print(f"      - {table[0]}")
        
        # Check users table
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"   👥 Users: {user_count}")
        
        # Check learning site activities
        cursor.execute("SELECT COUNT(*) FROM learning_site_activity")
        activity_count = cursor.fetchone()[0]
        print(f"   📊 Learning activities: {activity_count}")
        
        # Test Flask app connection
        print("2️⃣ Testing Flask app connection...")
        try:
            from app import create_app, db
            from app.models import User
            
            app = create_app()
            with app.app_context():
                users = User.query.all()
                print(f"   ✅ Flask can connect to MySQL: {len(users)} users found")
                
                # Test specific user
                user2 = User.query.get(2)
                if user2:
                    print(f"   ✅ User 2 found: {user2.username}")
                    print(f"   🔑 Password check: {user2.check_password('password123')}")
                else:
                    print("   ❌ User 2 not found")
                    
        except Exception as e:
            print(f"   ❌ Flask connection error: {e}")
        
        connection.close()
        
        print("\n🎉 MySQL connection test completed successfully!")
        print("✅ Database is ready for use")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Make sure MySQL server is running")
        print("2. Check your MySQL credentials in .env file")
        print("3. Run setup_mysql_with_password.py to configure MySQL")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mysql_connection()
