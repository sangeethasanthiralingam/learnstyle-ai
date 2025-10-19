#!/usr/bin/env python3
"""
Setup MySQL database for LearnStyle AI
"""

import pymysql
import os
from datetime import datetime

def setup_mysql():
    try:
        print("🔧 Setting up MySQL database...")
        
        # MySQL connection parameters
        host = 'localhost'
        user = 'root'
        password = 'root@1234'  # Default empty password for local MySQL
        database = 'learnstyle_ai'
        
        # Connect to MySQL server (without database)
        print("1️⃣ Connecting to MySQL server...")
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            charset='utf8mb4'
        )
        
        cursor = connection.cursor()
        
        # Create database
        print("2️⃣ Creating database...")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        print(f"✅ Database '{database}' created successfully")
        
        # Use the database
        cursor.execute(f"USE {database}")
        
        # Create tables using the existing SQL file
        print("3️⃣ Creating tables...")
        
        # Read the SQL file
        with open('create_tables.sql', 'r') as f:
            sql_content = f.read()
        
        # Split into individual statements
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
        
        for statement in statements:
            if statement and not statement.startswith('--'):
                try:
                    cursor.execute(statement)
                    print(f"   ✅ Executed: {statement[:50]}...")
                except pymysql.Error as e:
                    if "already exists" not in str(e):
                        print(f"   ⚠️ Warning: {e}")
        
        connection.commit()
        print("✅ Tables created successfully")
        
        # Create .env file with MySQL configuration
        print("4️⃣ Creating .env file...")
        env_content = f"""# LearnStyle AI Environment Configuration
SECRET_KEY=your-secret-key-change-this-in-production
DATABASE_URL=mysql+pymysql://{user}:{password}@{host}:3306/{database}

# MySQL Configuration
MYSQL_HOST={host}
MYSQL_PORT=3306
MYSQL_USER={user}
MYSQL_PASSWORD={password}
MYSQL_DATABASE={database}

# OpenAI Configuration (add your key here)
OPENAI_API_KEY=your-openai-api-key-here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ .env file created")
        
        connection.close()
        
        print("\n🎉 MySQL setup completed successfully!")
        print(f"📊 Database: {database}")
        print(f"🔗 Connection: mysql+pymysql://{user}@{host}:3306/{database}")
        print("\n📝 Next steps:")
        print("1. Update your .env file with your MySQL password if needed")
        print("2. Run: python migrate_data.py (to migrate data from SQLite)")
        print("3. Run: python app.py (to start the application)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    setup_mysql()
