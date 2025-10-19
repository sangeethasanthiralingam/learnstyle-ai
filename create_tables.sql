-- Users table
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    is_active TINYINT(1) DEFAULT 1,
    last_login DATETIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Learning profiles table
CREATE TABLE learning_profiles (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    visual_score FLOAT DEFAULT 0.0,
    auditory_score FLOAT DEFAULT 0.0,
    kinesthetic_score FLOAT DEFAULT 0.0,
    dominant_style VARCHAR(20),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Quiz responses table
CREATE TABLE quiz_responses (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    question_1 INT,
    question_2 INT,
    question_3 INT,
    question_4 INT,
    question_5 INT,
    question_6 INT,
    question_7 INT,
    question_8 INT,
    question_9 INT,
    question_10 INT,
    question_11 INT,
    question_12 INT,
    question_13 INT,
    question_14 INT,
    question_15 INT,
    submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Content library table
CREATE TABLE content_library (
    id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    content_type VARCHAR(20) NOT NULL,
    style_tags VARCHAR(100),
    difficulty_level VARCHAR(20),
    url_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User progress table
CREATE TABLE user_progress (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    content_id INT NOT NULL,
    completion_status VARCHAR(20) DEFAULT 'started',
    time_spent INT DEFAULT 0,
    score FLOAT,
    engagement_rating INT,
    completed_at DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (content_id) REFERENCES content_library(id)
);

-- Chat history table
CREATE TABLE chat_history (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    user_message TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    learning_style_context VARCHAR(20),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Question history table
CREATE TABLE question_history (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    learning_style VARCHAR(20),
    topic_category VARCHAR(50),
    confidence_score FLOAT,
    user_rating INT,
    is_saved TINYINT(1) DEFAULT 0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- User permissions table
CREATE TABLE user_permissions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL UNIQUE,
    camera_access TINYINT(1) DEFAULT 0,
    microphone_access TINYINT(1) DEFAULT 0,
    location_access TINYINT(1) DEFAULT 0,
    biometric_data TINYINT(1) DEFAULT 0,
    edu_sites_tracking TINYINT(1) DEFAULT 0,
    research_tracking TINYINT(1) DEFAULT 0,
    coding_tracking TINYINT(1) DEFAULT 0,
    video_tracking TINYINT(1) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Learning site activity table
CREATE TABLE learning_site_activity (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    site_url VARCHAR(500) NOT NULL,
    site_name VARCHAR(200),
    activity_type VARCHAR(50) DEFAULT 'visit',
    time_spent INT DEFAULT 0,
    content_type VARCHAR(100) DEFAULT 'general',
    notes TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Emotion data table
CREATE TABLE emotion_data (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    emotion_type VARCHAR(50) NOT NULL,
    confidence_score FLOAT,
    facial_emotion VARCHAR(50),
    voice_emotion VARCHAR(50),
    engagement_level FLOAT,
    context VARCHAR(100) DEFAULT 'learning',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Content table
CREATE TABLE content (
    id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(200) NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    learning_style VARCHAR(20),
    difficulty_level VARCHAR(20) DEFAULT 'beginner',
    tags TEXT,
    author_id INT NOT NULL,
    status VARCHAR(20) DEFAULT 'draft',
    views INT DEFAULT 0,
    rating FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (author_id) REFERENCES users(id)
);

-- Indexes
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_learning_profiles_user_id ON learning_profiles(user_id);
CREATE INDEX idx_quiz_responses_user_id ON quiz_responses(user_id);
CREATE INDEX idx_user_progress_user_id ON user_progress(user_id);
CREATE INDEX idx_user_progress_content_id ON user_progress(content_id);
CREATE INDEX idx_chat_history_user_id ON chat_history(user_id);
CREATE INDEX idx_question_history_user_id ON question_history(user_id);
CREATE INDEX idx_user_permissions_user_id ON user_permissions(user_id);
CREATE INDEX idx_learning_site_activity_user_id ON learning_site_activity(user_id);
CREATE INDEX idx_emotion_data_user_id ON emotion_data(user_id);
CREATE INDEX idx_content_author_id ON content(author_id);
CREATE INDEX idx_content_status ON content(status);