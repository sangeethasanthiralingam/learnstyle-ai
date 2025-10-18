# MySQL Database Setup for LearnStyle AI

This guide will help you set up MySQL database for the LearnStyle AI application.

## Prerequisites

- Python 3.8 or higher
- MySQL Server 8.0 or higher
- pip (Python package manager)

## Step 1: Install MySQL Server

### Windows
1. Download MySQL Installer from [mysql.com](https://dev.mysql.com/downloads/installer/)
2. Run the installer and follow the setup wizard
3. Choose "MySQL Server" and "MySQL Workbench" (optional)
4. Set a root password during installation
5. Note down the root password for later use

### macOS
```bash
# Using Homebrew
brew install mysql
brew services start mysql

# Set root password
mysql_secure_installation
```

### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install MySQL server
sudo apt install mysql-server

# Secure installation
sudo mysql_secure_installation

# Start MySQL service
sudo systemctl start mysql
sudo systemctl enable mysql
```

### CentOS/RHEL
```bash
# Install MySQL server
sudo yum install mysql-server

# Start MySQL service
sudo systemctl start mysqld
sudo systemctl enable mysqld

# Get temporary password
sudo grep 'temporary password' /var/log/mysqld.log

# Secure installation
sudo mysql_secure_installation
```

## Step 2: Install Python Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Or install MySQL-specific packages
pip install PyMySQL cryptography
```

## Step 3: Configure Environment Variables

1. Copy the example environment file:
```bash
cp env.example .env
```

2. Edit the `.env` file with your MySQL credentials:
```env
# Database Configuration
MYSQL_USER=root
MYSQL_PASSWORD=your-mysql-password
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=learnstyle_ai

# Flask Configuration
SECRET_KEY=your-secret-key-here-change-in-production
```

## Step 4: Create Database and Tables

Run the database setup script:

```bash
python setup_database.py
```

This script will:
- Create the `learnstyle_ai` database
- Create all necessary tables
- Create a default admin user (admin/admin123)

## Step 5: Verify Installation

1. Start the application:
```bash
python app.py
```

2. Visit http://localhost:5000
3. Login with admin/admin123
4. Change the admin password immediately

## Database Schema

The application creates the following tables:

- `users` - User accounts and authentication
- `learning_profiles` - Learning style preferences
- `quiz_responses` - Learning style assessment responses
- `content_library` - Educational content
- `user_progress` - Learning progress tracking
- `chat_history` - AI tutor conversations

## Troubleshooting

### Connection Issues

1. **Access Denied Error**:
   - Verify MySQL credentials in `.env` file
   - Ensure MySQL server is running
   - Check if user has proper permissions

2. **Database Not Found**:
   - Run the setup script: `python setup_database.py`
   - Manually create database: `CREATE DATABASE learnstyle_ai;`

3. **Port Already in Use**:
   - Check if MySQL is running on port 3306
   - Change port in `.env` file if needed

### Performance Optimization

1. **Connection Pooling**:
   The application is configured with connection pooling:
   ```python
   app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
       'pool_pre_ping': True,
       'pool_recycle': 300,
       'pool_timeout': 20,
       'max_overflow': 0
   }
   ```

2. **MySQL Configuration**:
   Add these settings to your MySQL configuration file (`my.cnf` or `my.ini`):
   ```ini
   [mysqld]
   innodb_buffer_pool_size = 256M
   max_connections = 100
   query_cache_size = 32M
   ```

## Security Considerations

1. **Change Default Passwords**:
   - Change admin password after first login
   - Use strong passwords for MySQL root user

2. **Database Permissions**:
   - Create a dedicated MySQL user for the application
   - Grant only necessary permissions

3. **Environment Variables**:
   - Never commit `.env` file to version control
   - Use strong secret keys in production

## Production Deployment

For production deployment:

1. **Use Environment Variables**:
   ```bash
   export MYSQL_USER=learnstyle_user
   export MYSQL_PASSWORD=strong_password
   export MYSQL_HOST=your-mysql-server
   export MYSQL_DATABASE=learnstyle_ai
   ```

2. **SSL Configuration**:
   ```python
   app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
       'pool_pre_ping': True,
       'pool_recycle': 300,
       'pool_timeout': 20,
       'max_overflow': 0,
       'connect_args': {
           'ssl_disabled': False,
           'ssl_ca': '/path/to/ca-cert.pem'
       }
   }
   ```

3. **Backup Strategy**:
   ```bash
   # Create database backup
   mysqldump -u root -p learnstyle_ai > backup.sql
   
   # Restore from backup
   mysql -u root -p learnstyle_ai < backup.sql
   ```

## Support

If you encounter issues:

1. Check MySQL error logs
2. Verify database connection with MySQL Workbench
3. Test connection with Python:
   ```python
   import pymysql
   connection = pymysql.connect(
       host='localhost',
       user='root',
       password='your_password',
       database='learnstyle_ai'
   )
   print("Connection successful!")
   connection.close()
   ```

For additional help, check the application logs or contact support.
