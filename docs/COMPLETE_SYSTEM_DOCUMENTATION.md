# LearnStyle AI - Complete System Documentation

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Learning Tracking System](#learning-tracking-system)
3. [Admin Content Management](#admin-content-management)
4. [User Permissions & Privacy](#user-permissions--privacy)
5. [API Documentation](#api-documentation)
6. [Database Schema](#database-schema)
7. [Browser Extension](#browser-extension)
8. [Deployment Guide](#deployment-guide)
9. [User Guide](#user-guide)
10. [Admin Guide](#admin-guide)

## 🎯 System Overview

LearnStyle AI is a comprehensive personalized learning platform that tracks user learning activities across multiple platforms and provides AI-powered recommendations based on individual learning styles.

### Core Features
- **Learning Style Assessment**: AI-powered quiz to determine user's learning style
- **Cross-Platform Tracking**: Monitors learning activities on YouTube, GitHub, Coursera, etc.
- **Personalized Content**: AI-generated content based on learning style
- **Real-time Analytics**: Live dashboard with learning insights
- **Privacy-First Design**: Complete user control over data collection
- **Admin Content Management**: Full CRUD operations for learning content

## 🔍 Learning Tracking System

### Supported Learning Platforms

#### Educational Platforms
- **Coursera** - University-level courses
- **Khan Academy** - Free educational content
- **edX** - Online courses from top universities
- **Udemy** - Skill-based courses
- **Udacity** - Technology courses
- **Skillshare** - Creative and business skills
- **Pluralsight** - Technology and creative skills

#### Coding Platforms
- **GitHub** - Code repositories and collaboration
- **Stack Overflow** - Programming Q&A
- **LeetCode** - Coding challenges and interviews
- **HackerRank** - Programming contests
- **CodeWars** - Coding challenges
- **FreeCodeCamp** - Free coding curriculum
- **Codecademy** - Interactive coding lessons

#### Video Learning
- **YouTube** - Educational videos and tutorials
- **Vimeo** - Professional educational content
- **TED** - Inspirational talks and lectures

#### Research Platforms
- **Google Scholar** - Academic papers and research
- **ResearchGate** - Scientific collaboration
- **Academia.edu** - Academic social network
- **arXiv** - Preprint repository
- **IEEE Xplore** - Engineering and technology papers
- **ACM Digital Library** - Computer science papers

### Activity Types Tracked
- **Visit** - Basic site visits
- **Study** - Active learning sessions
- **Video** - Video content consumption
- **Quiz** - Assessment and testing
- **Assignment** - Homework and projects
- **Discussion** - Forum participation
- **Lecture** - Educational content consumption
- **Code Review** - Code analysis and review
- **Research** - Academic paper reading

## 👨‍💼 Admin Content Management

### Content Management Features

#### 1. Content Creation
- **Rich Text Editor**: WYSIWYG editor for content creation
- **Media Upload**: Support for images, videos, and documents
- **Content Templates**: Pre-built templates for different content types
- **Bulk Import**: CSV/Excel import for large content sets
- **Version Control**: Track content changes and revisions

#### 2. Content Organization
- **Categories**: Organize content by subject, difficulty, or type
- **Tags**: Flexible tagging system for content discovery
- **Learning Paths**: Create structured learning sequences
- **Prerequisites**: Set content dependencies
- **Difficulty Levels**: Beginner, Intermediate, Advanced

#### 3. Content Analytics
- **Engagement Metrics**: Views, completion rates, time spent
- **User Feedback**: Ratings, comments, and reviews
- **Performance Analytics**: Learning outcomes and effectiveness
- **A/B Testing**: Test different content versions
- **Heat Maps**: Visual content interaction analysis

#### 4. Content Moderation
- **Content Review**: Moderation queue for new content
- **Quality Control**: Automated and manual quality checks
- **Content Flagging**: User-reported content issues
- **Approval Workflow**: Multi-level content approval process
- **Content Scheduling**: Publish content at specific times

### Admin Dashboard Features

#### Content Management Interface
```html
<!-- Content Management Dashboard -->
<div class="admin-content-dashboard">
    <div class="content-stats">
        <div class="stat-card">
            <h3>Total Content</h3>
            <span class="stat-number">{{ total_content }}</span>
        </div>
        <div class="stat-card">
            <h3>Published</h3>
            <span class="stat-number">{{ published_content }}</span>
        </div>
        <div class="stat-card">
            <h3>Draft</h3>
            <span class="stat-number">{{ draft_content }}</span>
        </div>
        <div class="stat-card">
            <h3>Pending Review</h3>
            <span class="stat-number">{{ pending_content }}</span>
        </div>
    </div>
    
    <div class="content-actions">
        <button class="btn btn-primary" onclick="createContent()">
            <i class="fas fa-plus"></i> Create New Content
        </button>
        <button class="btn btn-secondary" onclick="importContent()">
            <i class="fas fa-upload"></i> Import Content
        </button>
        <button class="btn btn-info" onclick="exportContent()">
            <i class="fas fa-download"></i> Export Content
        </button>
    </div>
    
    <div class="content-list">
        <!-- Content table with CRUD operations -->
    </div>
</div>
```

#### Content Creation Form
```html
<!-- Content Creation Form -->
<form id="content-form" class="content-creation-form">
    <div class="form-group">
        <label for="content-title">Title</label>
        <input type="text" id="content-title" name="title" required>
    </div>
    
    <div class="form-group">
        <label for="content-type">Content Type</label>
        <select id="content-type" name="content_type" required>
            <option value="article">Article</option>
            <option value="video">Video</option>
            <option value="quiz">Quiz</option>
            <option value="interactive">Interactive</option>
            <option value="assignment">Assignment</option>
        </select>
    </div>
    
    <div class="form-group">
        <label for="learning-style">Target Learning Style</label>
        <select id="learning-style" name="learning_style">
            <option value="visual">Visual</option>
            <option value="auditory">Auditory</option>
            <option value="kinesthetic">Kinesthetic</option>
            <option value="reading">Reading/Writing</option>
            <option value="multimodal">Multimodal</option>
        </select>
    </div>
    
    <div class="form-group">
        <label for="difficulty-level">Difficulty Level</label>
        <select id="difficulty-level" name="difficulty_level">
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
        </select>
    </div>
    
    <div class="form-group">
        <label for="content-body">Content Body</label>
        <textarea id="content-body" name="content" rows="10" required></textarea>
    </div>
    
    <div class="form-group">
        <label for="content-tags">Tags</label>
        <input type="text" id="content-tags" name="tags" placeholder="Enter tags separated by commas">
    </div>
    
    <div class="form-actions">
        <button type="submit" class="btn btn-primary">Create Content</button>
        <button type="button" class="btn btn-secondary" onclick="saveDraft()">Save as Draft</button>
        <button type="button" class="btn btn-outline" onclick="previewContent()">Preview</button>
    </div>
</form>
```

## 🔒 User Permissions & Privacy

### Permission Types

#### 1. Device Permissions
- **Camera Access**: For emotion detection and engagement analysis
- **Microphone Access**: For voice emotion analysis
- **Location Access**: For contextual learning recommendations

#### 2. Data Collection Permissions
- **Biometric Data**: Heart rate variability, stress monitoring
- **Educational Sites**: Coursera, Khan Academy, edX, Udemy
- **Coding Platforms**: GitHub, Stack Overflow, LeetCode
- **Video Learning**: YouTube, Vimeo, educational channels
- **Research Sites**: Google Scholar, ResearchGate, arXiv

#### 3. Privacy Controls
- **Data Retention**: Choose how long data is stored
- **Data Sharing**: Control sharing with third parties
- **Analytics**: Opt-in/out of usage analytics
- **Marketing**: Control marketing communications

### Permission Management Interface

```html
<!-- Permission Management -->
<div class="permission-management">
    <div class="permission-category">
        <h3>Device Access</h3>
        <div class="permission-item">
            <div class="permission-info">
                <h4>Camera Access</h4>
                <p>Used for emotion detection and engagement analysis</p>
            </div>
            <div class="permission-control">
                <label class="toggle-switch">
                    <input type="checkbox" id="camera-permission">
                    <span class="slider"></span>
                </label>
            </div>
        </div>
    </div>
    
    <div class="permission-category">
        <h3>Learning Site Tracking</h3>
        <div class="permission-item">
            <div class="permission-info">
                <h4>Educational Platforms</h4>
                <p>Track learning activities on Coursera, Khan Academy, etc.</p>
            </div>
            <div class="permission-control">
                <label class="toggle-switch">
                    <input type="checkbox" id="edu-tracking">
                    <span class="slider"></span>
                </label>
            </div>
        </div>
    </div>
</div>
```

## 📡 API Documentation

### Authentication Endpoints
```
POST /api/auth/login - User login
POST /api/auth/register - User registration
POST /api/auth/logout - User logout
GET /api/auth/profile - Get user profile
PUT /api/auth/profile - Update user profile
```

### Learning Tracking Endpoints
```
GET /api/learning-sites - Get learning activities
POST /api/learning-sites - Track learning activity
GET /api/learning-sites/stats - Get learning statistics
DELETE /api/learning-sites/{id} - Delete activity record
```

### Content Management Endpoints
```
GET /api/content - Get content list
POST /api/content - Create new content
GET /api/content/{id} - Get specific content
PUT /api/content/{id} - Update content
DELETE /api/content/{id} - Delete content
POST /api/content/{id}/publish - Publish content
POST /api/content/{id}/unpublish - Unpublish content
```

### Permission Endpoints
```
GET /api/permissions - Get user permissions
POST /api/permissions - Update user permissions
GET /api/permissions/status - Get permission status
```

### Analytics Endpoints
```
GET /api/analytics/learning - Learning analytics
GET /api/analytics/content - Content performance
GET /api/analytics/users - User engagement
GET /api/analytics/system - System health
```

## 🗄️ Database Schema

### Core Tables

#### Users Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(128),
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### Learning Profiles Table
```sql
CREATE TABLE learning_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    dominant_style VARCHAR(20),
    visual_score FLOAT DEFAULT 0.0,
    auditory_score FLOAT DEFAULT 0.0,
    kinesthetic_score FLOAT DEFAULT 0.0,
    reading_score FLOAT DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
```

#### Content Table
```sql
CREATE TABLE content (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title VARCHAR(200) NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    learning_style VARCHAR(20),
    difficulty_level VARCHAR(20),
    tags TEXT,
    author_id INTEGER,
    status VARCHAR(20) DEFAULT 'draft',
    views INTEGER DEFAULT 0,
    rating FLOAT DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (author_id) REFERENCES users (id)
);
```

#### Learning Site Activity Table
```sql
CREATE TABLE learning_site_activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    site_url VARCHAR(500) NOT NULL,
    site_name VARCHAR(200),
    activity_type VARCHAR(50) DEFAULT 'visit',
    time_spent INTEGER DEFAULT 0,
    content_type VARCHAR(100) DEFAULT 'general',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
```

#### User Permissions Table
```sql
CREATE TABLE user_permissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER UNIQUE NOT NULL,
    camera_access BOOLEAN DEFAULT FALSE,
    microphone_access BOOLEAN DEFAULT FALSE,
    location_access BOOLEAN DEFAULT FALSE,
    biometric_data BOOLEAN DEFAULT FALSE,
    edu_sites_tracking BOOLEAN DEFAULT FALSE,
    research_tracking BOOLEAN DEFAULT FALSE,
    coding_tracking BOOLEAN DEFAULT FALSE,
    video_tracking BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
```

## 🌐 Browser Extension

### Extension Structure
```
browser-extension/
├── manifest.json          # Extension configuration
├── content.js            # Content script for site injection
├── background.js         # Background service worker
├── popup.html           # Extension popup interface
├── popup.js             # Popup functionality
├── icons/               # Extension icons
│   ├── icon16.png
│   ├── icon32.png
│   ├── icon48.png
│   └── icon128.png
└── README.md            # Extension documentation
```

### Extension Features
- **Automatic Site Detection**: Recognizes learning platforms
- **Permission Management**: Respects user privacy settings
- **Real-time Tracking**: Monitors learning activities
- **Data Synchronization**: Sends data to LearnStyle AI
- **Visual Indicators**: Shows tracking status

## 🚀 Deployment Guide

### Prerequisites
- Python 3.8+
- Node.js 14+
- MySQL 8.0+
- Redis (optional, for caching)

### Installation Steps

1. **Clone Repository**
```bash
git clone https://github.com/your-org/learnstyle-ai.git
cd learnstyle-ai
```

2. **Install Python Dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Node.js Dependencies**
```bash
cd browser-extension
npm install
```

4. **Database Setup**
```bash
python setup_database.py
```

5. **Environment Configuration**
```bash
cp env.example .env
# Edit .env with your configuration
```

6. **Run Application**
```bash
python app.py
```

### Production Deployment

1. **Use Production WSGI Server**
```bash
gunicorn --bind 0.0.0.0:8000 app:app
```

2. **Configure Reverse Proxy (Nginx)**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

3. **SSL Configuration**
```bash
certbot --nginx -d your-domain.com
```

## 📖 User Guide

### Getting Started

1. **Create Account**
   - Visit the registration page
   - Fill in your details
   - Verify your email

2. **Take Learning Style Assessment**
   - Complete the quiz
   - Get your learning style results
   - View personalized recommendations

3. **Set Privacy Preferences**
   - Go to Privacy & Permissions
   - Choose what data to share
   - Enable/disable tracking features

4. **Install Browser Extension**
   - Download the extension
   - Install in your browser
   - Log in to your account

5. **Start Learning**
   - Visit learning sites
   - Extension tracks your activities
   - View insights in dashboard

### Using the Dashboard

1. **Learning Progress**
   - View completed content
   - Track learning hours
   - See achievement badges

2. **Learning Sites Activity**
   - Recent learning sites
   - Time spent on each platform
   - Learning patterns

3. **Personalized Content**
   - AI-generated recommendations
   - Content matching your style
   - Difficulty-appropriate materials

4. **AI Tutor Chat**
   - Ask questions
   - Get personalized answers
   - Learn at your own pace

## 👨‍💼 Admin Guide

### Content Management

1. **Creating Content**
   - Go to Admin Dashboard
   - Click "Create New Content"
   - Fill in content details
   - Choose target learning style
   - Set difficulty level
   - Publish or save as draft

2. **Managing Content**
   - View all content
   - Edit existing content
   - Delete unwanted content
   - Moderate user-generated content

3. **Content Analytics**
   - View engagement metrics
   - Analyze user feedback
   - Track content performance
   - Optimize content strategy

### User Management

1. **User Overview**
   - View all users
   - Check user activity
   - Monitor learning progress
   - Manage user roles

2. **Learning Analytics**
   - Overall learning trends
   - Popular content
   - User engagement
   - Learning effectiveness

3. **System Monitoring**
   - System health status
   - Performance metrics
   - Error tracking
   - Resource usage

### Privacy & Compliance

1. **Data Management**
   - User data overview
   - Privacy settings
   - Data retention policies
   - GDPR compliance

2. **Permission Management**
   - User permission overview
   - Privacy controls
   - Data collection settings
   - Consent management

## 🔧 Technical Specifications

### System Requirements
- **Backend**: Python 3.8+, Flask, SQLAlchemy
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Database**: MySQL 8.0+
- **Browser Extension**: Manifest V3
- **AI/ML**: scikit-learn, pandas, numpy

### Performance Metrics
- **Response Time**: < 200ms for API calls
- **Uptime**: 99.9% availability
- **Concurrent Users**: 1000+ simultaneous users
- **Data Processing**: Real-time learning analytics

### Security Features
- **Authentication**: JWT-based authentication
- **Authorization**: Role-based access control
- **Data Encryption**: AES-256 encryption
- **Privacy**: GDPR/CCPA compliant
- **HTTPS**: SSL/TLS encryption

## 📞 Support & Maintenance

### Support Channels
- **Email**: support@learnstyle-ai.com
- **Documentation**: https://docs.learnstyle-ai.com
- **Community Forum**: https://community.learnstyle-ai.com
- **GitHub Issues**: https://github.com/your-org/learnstyle-ai/issues

### Maintenance Schedule
- **Daily**: System health checks
- **Weekly**: Performance optimization
- **Monthly**: Security updates
- **Quarterly**: Feature updates

---

This documentation provides a comprehensive overview of the LearnStyle AI system, including learning tracking, admin content management, user permissions, and complete technical specifications.
