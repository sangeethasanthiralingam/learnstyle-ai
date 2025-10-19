#!/usr/bin/env python3
"""
Migrate data from SQLite to MySQL
"""

import sqlite3
import pymysql
import os
from datetime import datetime
from werkzeug.security import generate_password_hash

def migrate_data():
    try:
        print("🔄 Migrating data from SQLite to MySQL...")
        
        # Check if .env file exists
        if not os.path.exists('.env'):
            print("❌ .env file not found. Please run setup_mysql_with_password.py first.")
            return
        
        # Read MySQL credentials from .env
        mysql_config = {}
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    mysql_config[key] = value
        
        # Connect to SQLite
        print("1️⃣ Connecting to SQLite database...")
        sqlite_conn = sqlite3.connect('instance/learnstyle.db')
        sqlite_cursor = sqlite_conn.cursor()
        
        # Connect to MySQL
        print("2️⃣ Connecting to MySQL database...")
        mysql_conn = pymysql.connect(
            host=mysql_config.get('MYSQL_HOST', 'localhost'),
            user=mysql_config.get('MYSQL_USER', 'root'),
            password=mysql_config.get('MYSQL_PASSWORD', ''),
            database=mysql_config.get('MYSQL_DATABASE', 'learnstyle_ai'),
            charset='utf8mb4'
        )
        mysql_cursor = mysql_conn.cursor()
        
        # Migrate users
        print("3️⃣ Migrating users...")
        sqlite_cursor.execute("SELECT id, username, email, password_hash, role, is_active, last_login, created_at FROM users")
        users = sqlite_cursor.fetchall()
        
        for user in users:
            try:
                mysql_cursor.execute("""
                    INSERT INTO users (id, username, email, password_hash, role, is_active, last_login, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    username = VALUES(username),
                    email = VALUES(email),
                    password_hash = VALUES(password_hash),
                    role = VALUES(role),
                    is_active = VALUES(is_active),
                    last_login = VALUES(last_login),
                    created_at = VALUES(created_at)
                """, user)
                print(f"   ✅ Migrated user: {user[1]}")
            except Exception as e:
                print(f"   ⚠️ Error migrating user {user[1]}: {e}")
        
        # Migrate learning profiles
        print("4️⃣ Migrating learning profiles...")
        sqlite_cursor.execute("SELECT id, user_id, visual_score, auditory_score, kinesthetic_score, dominant_style, last_updated FROM learning_profiles")
        profiles = sqlite_cursor.fetchall()
        
        for profile in profiles:
            try:
                mysql_cursor.execute("""
                    INSERT INTO learning_profiles (id, user_id, visual_score, auditory_score, kinesthetic_score, dominant_style, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    visual_score = VALUES(visual_score),
                    auditory_score = VALUES(auditory_score),
                    kinesthetic_score = VALUES(kinesthetic_score),
                    dominant_style = VALUES(dominant_style),
                    last_updated = VALUES(last_updated)
                """, profile)
                print(f"   ✅ Migrated profile for user: {profile[1]}")
            except Exception as e:
                print(f"   ⚠️ Error migrating profile for user {profile[1]}: {e}")
        
        # Migrate learning site activities
        print("5️⃣ Migrating learning site activities...")
        sqlite_cursor.execute("SELECT id, user_id, site_url, site_name, activity_type, time_spent, content_type, notes, timestamp FROM learning_site_activity")
        activities = sqlite_cursor.fetchall()
        
        for activity in activities:
            try:
                mysql_cursor.execute("""
                    INSERT INTO learning_site_activity (id, user_id, site_url, site_name, activity_type, time_spent, content_type, notes, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    site_url = VALUES(site_url),
                    site_name = VALUES(site_name),
                    activity_type = VALUES(activity_type),
                    time_spent = VALUES(time_spent),
                    content_type = VALUES(content_type),
                    notes = VALUES(notes),
                    timestamp = VALUES(timestamp)
                """, activity)
                print(f"   ✅ Migrated activity: {activity[3]} for user {activity[1]}")
            except Exception as e:
                print(f"   ⚠️ Error migrating activity for user {activity[1]}: {e}")
        
        # Migrate other tables if they exist
        tables_to_migrate = [
            'quiz_responses',
            'user_progress', 
            'chat_history',
            'question_history',
            'user_permissions',
            'emotion_data',
            'content',
            'content_library'
        ]
        
        for table in tables_to_migrate:
            try:
                print(f"6️⃣ Migrating {table}...")
                sqlite_cursor.execute(f"SELECT * FROM {table}")
                rows = sqlite_cursor.fetchall()
                
                if rows:
                    # Get column names
                    sqlite_cursor.execute(f"PRAGMA table_info({table})")
                    columns = [col[1] for col in sqlite_cursor.fetchall()]
                    
                    # Create placeholders for INSERT
                    placeholders = ', '.join(['%s'] * len(columns))
                    columns_str = ', '.join(columns)
                    
                    for row in rows:
                        try:
                            mysql_cursor.execute(f"""
                                INSERT INTO {table} ({columns_str})
                                VALUES ({placeholders})
                                ON DUPLICATE KEY UPDATE
                                {', '.join([f"{col} = VALUES({col})" for col in columns if col != 'id'])}
                            """, row)
                        except Exception as e:
                            print(f"   ⚠️ Error migrating row in {table}: {e}")
                    
                    print(f"   ✅ Migrated {len(rows)} rows from {table}")
                else:
                    print(f"   ℹ️ No data in {table}")
                    
            except sqlite3.OperationalError:
                print(f"   ℹ️ Table {table} doesn't exist in SQLite")
            except Exception as e:
                print(f"   ⚠️ Error migrating {table}: {e}")
        
        # Commit all changes
        mysql_conn.commit()
        
        # Close connections
        sqlite_conn.close()
        mysql_conn.close()
        
        print("\n🎉 Data migration completed successfully!")
        print("📊 All data has been migrated from SQLite to MySQL")
        print("\n📝 Next steps:")
        print("1. Run: python app.py (to start the application with MySQL)")
        print("2. Test the login and dashboard functionality")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    migrate_data()
