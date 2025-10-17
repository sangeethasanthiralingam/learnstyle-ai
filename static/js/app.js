/**
 * LearnStyle AI - Main JavaScript Application
 * Handles quiz functionality, dashboard interactions, and API calls
 */

class LearnStyleApp {
    constructor() {
        this.currentQuestionIndex = 0;
        this.quizAnswers = [];
        this.totalQuestions = 15;
        
        this.initializeApp();
    }
    
    initializeApp() {
        // Initialize components based on current page
        const path = window.location.pathname;
        
        if (path.includes('quiz')) {
            this.initializeQuiz();
        } else if (path.includes('dashboard')) {
            this.initializeDashboard();
        } else if (path.includes('chat')) {
            this.initializeChat();
        }
        
        // Initialize common components
        this.initializeCommon();
    }
    
    initializeCommon() {
        // Auto-dismiss alerts after 5 seconds
        setTimeout(() => {
            const alerts = document.querySelectorAll('.alert');
            alerts.forEach(alert => {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            });
        }, 5000);
        
        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }
    
    initializeQuiz() {
        console.log('Initializing quiz...');
        this.quizQuestions = [
            {
                question: "When learning something new, I prefer to:",
                options: [
                    "Read about it or see diagrams and charts",
                    "Listen to explanations or discussions",
                    "Jump in and try it hands-on"
                ]
            },
            {
                question: "I remember information best when:",
                options: [
                    "I can visualize it or see it written down",
                    "I hear it explained or discuss it with others",
                    "I practice or apply it myself"
                ]
            },
            {
                question: "When following directions, I prefer:",
                options: [
                    "Written instructions with diagrams",
                    "Verbal explanations",
                    "To figure it out by doing it"
                ]
            },
            {
                question: "In a classroom, I learn best when:",
                options: [
                    "The teacher uses visual aids like slides or diagrams",
                    "The teacher lectures and we have discussions",
                    "We do hands-on activities and experiments"
                ]
            },
            {
                question: "When studying, I prefer to:",
                options: [
                    "Make notes, highlight text, and create mind maps",
                    "Read aloud or explain concepts to others",
                    "Use flashcards and practice exercises"
                ]
            },
            {
                question: "I understand concepts better when:",
                options: [
                    "I can see examples and visual representations",
                    "Someone explains them to me verbally",
                    "I can work through examples myself"
                ]
            },
            {
                question: "When problem-solving, I tend to:",
                options: [
                    "Draw diagrams or make lists",
                    "Talk through the problem out loud",
                    "Try different approaches until something works"
                ]
            },
            {
                question: "I prefer books and materials that:",
                options: [
                    "Have lots of charts, graphs, and visual elements",
                    "I can read aloud or that have audio versions",
                    "Include interactive exercises and activities"
                ]
            },
            {
                question: "When learning a new skill, I:",
                options: [
                    "Watch demonstrations and study examples",
                    "Listen to instructions and ask questions",
                    "Jump in and learn by trial and error"
                ]
            },
            {
                question: "In meetings or presentations, I:",
                options: [
                    "Take detailed notes and sketch ideas",
                    "Focus on listening and asking questions",
                    "Prefer interactive workshops over lectures"
                ]
            },
            {
                question: "When I need to remember something important, I:",
                options: [
                    "Write it down or create visual reminders",
                    "Repeat it to myself or tell someone else",
                    "Associate it with an action or experience"
                ]
            },
            {
                question: "I work best in environments that are:",
                options: [
                    "Visually organized with charts and references",
                    "Quiet enough to focus on audio content",
                    "Flexible where I can move around and be active"
                ]
            },
            {
                question: "When explaining something to others, I:",
                options: [
                    "Draw pictures or use visual aids",
                    "Explain verbally with examples",
                    "Show them how to do it step by step"
                ]
            },
            {
                question: "I prefer learning materials that:",
                options: [
                    "Are well-organized visually with headings and bullet points",
                    "I can listen to like podcasts or audio books",
                    "Include quizzes, exercises, and interactive elements"
                ]
            },
            {
                question: "When I'm confused about something, I:",
                options: [
                    "Look for diagrams, examples, or visual explanations",
                    "Ask someone to explain it to me",
                    "Try to work through it by experimenting"
                ]
            }
        ];
        
        this.renderQuiz();
        this.attachQuizEventListeners();
    }
    
    renderQuiz() {
        const quizContainer = document.getElementById('quiz-container');
        if (!quizContainer) return;
        
        const question = this.quizQuestions[this.currentQuestionIndex];
        const progress = ((this.currentQuestionIndex + 1) / this.totalQuestions) * 100;
        
        quizContainer.innerHTML = `
            <div class="quiz-card p-4">
                <div class="quiz-progress mb-4">
                    <div class="quiz-progress-bar" style="width: ${progress}%"></div>
                </div>
                
                <div class="text-center mb-4">
                    <small class="text-muted">Question ${this.currentQuestionIndex + 1} of ${this.totalQuestions}</small>
                </div>
                
                <div class="quiz-question mb-4">
                    ${question.question}
                </div>
                
                <div class="quiz-options">
                    ${question.options.map((option, index) => `
                        <label class="quiz-option" for="option-${index}">
                            <input type="radio" id="option-${index}" name="quiz-answer" value="${index + 1}">
                            <span>${option}</span>
                        </label>
                    `).join('')}
                </div>
                
                <div class="d-flex justify-content-between mt-4">
                    <button type="button" class="btn btn-outline-secondary" id="prev-btn" 
                            ${this.currentQuestionIndex === 0 ? 'disabled' : ''}>
                        <i class="bi bi-arrow-left"></i> Previous
                    </button>
                    
                    <button type="button" class="btn btn-primary" id="next-btn" disabled>
                        ${this.currentQuestionIndex === this.totalQuestions - 1 ? 'Submit Quiz' : 'Next'} 
                        <i class="bi bi-arrow-right"></i>
                    </button>
                </div>
            </div>
        `;
        
        // Restore previous answer if exists
        if (this.quizAnswers[this.currentQuestionIndex]) {
            const answerValue = this.quizAnswers[this.currentQuestionIndex];
            const radio = document.querySelector(`input[value="${answerValue}"]`);
            if (radio) {
                radio.checked = true;
                radio.closest('.quiz-option').classList.add('selected');
                document.getElementById('next-btn').disabled = false;
            }
        }
    }
    
    attachQuizEventListeners() {
        // Handle option selection
        document.addEventListener('change', (e) => {
            if (e.target.name === 'quiz-answer') {
                // Remove selected class from all options
                document.querySelectorAll('.quiz-option').forEach(option => {
                    option.classList.remove('selected');
                });
                
                // Add selected class to chosen option
                e.target.closest('.quiz-option').classList.add('selected');
                
                // Store answer
                this.quizAnswers[this.currentQuestionIndex] = parseInt(e.target.value);
                
                // Enable next button
                document.getElementById('next-btn').disabled = false;
            }
        });
        
        // Handle next button
        document.addEventListener('click', (e) => {
            if (e.target.id === 'next-btn') {
                if (this.currentQuestionIndex === this.totalQuestions - 1) {
                    this.submitQuiz();
                } else {
                    this.nextQuestion();
                }
            }
        });
        
        // Handle previous button
        document.addEventListener('click', (e) => {
            if (e.target.id === 'prev-btn') {
                this.previousQuestion();
            }
        });
    }
    
    nextQuestion() {
        if (this.currentQuestionIndex < this.totalQuestions - 1) {
            this.currentQuestionIndex++;
            this.renderQuiz();
        }
    }
    
    previousQuestion() {
        if (this.currentQuestionIndex > 0) {
            this.currentQuestionIndex--;
            this.renderQuiz();
        }
    }
    
    async submitQuiz() {
        // Show loading state
        const submitBtn = document.getElementById('next-btn');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<div class="spinner-border spinner-border-sm me-2"></div>Analyzing...';
        submitBtn.disabled = true;
        
        try {
            const response = await fetch('/api/quiz', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    answers: this.quizAnswers
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Show results
                this.showQuizResults(data);
            } else {
                throw new Error(data.error || 'Failed to submit quiz');
            }
        } catch (error) {
            console.error('Quiz submission error:', error);
            this.showError('Failed to submit quiz. Please try again.');
            
            // Restore button
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
    }
    
    showQuizResults(data) {
        const quizContainer = document.getElementById('quiz-container');
        const prediction = data.prediction;
        const breakdown = data.style_breakdown;
        
        quizContainer.innerHTML = `
            <div class="quiz-card p-5 text-center">
                <div class="mb-4">
                    <i class="bi bi-check-circle-fill text-success" style="font-size: 4rem;"></i>
                    <h2 class="mt-3 mb-3">Your Learning Style Analysis</h2>
                    <p class="text-muted">Based on your responses, here's your personalized learning profile:</p>
                </div>
                
                <div class="row g-4 mb-4">
                    <div class="col-md-4">
                        <div class="bg-info bg-opacity-10 rounded-3 p-3">
                            <i class="bi bi-eye text-info" style="font-size: 2rem;"></i>
                            <h4 class="text-info mt-2">Visual</h4>
                            <div class="style-percentage text-info">${breakdown.visual}%</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="bg-success bg-opacity-10 rounded-3 p-3">
                            <i class="bi bi-volume-up text-success" style="font-size: 2rem;"></i>
                            <h4 class="text-success mt-2">Auditory</h4>
                            <div class="style-percentage text-success">${breakdown.auditory}%</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="bg-warning bg-opacity-10 rounded-3 p-3">
                            <i class="bi bi-hand-index text-warning" style="font-size: 2rem;"></i>
                            <h4 class="text-warning mt-2">Kinesthetic</h4>
                            <div class="style-percentage text-warning">${breakdown.kinesthetic}%</div>
                        </div>
                    </div>
                </div>
                
                <div class="alert alert-primary" role="alert">
                    <h5 class="alert-heading">
                        <i class="bi bi-lightbulb"></i> Your Dominant Learning Style: ${prediction.dominant_style.charAt(0).toUpperCase() + prediction.dominant_style.slice(1)}
                    </h5>
                    <p class="mb-0">You learn best through ${this.getStyleDescription(prediction.dominant_style)} approaches.</p>
                </div>
                
                <div class="mt-4">
                    <a href="/dashboard" class="btn btn-primary btn-lg me-3">
                        <i class="bi bi-speedometer2"></i> View Dashboard
                    </a>
                    <a href="/chat" class="btn btn-success btn-lg">
                        <i class="bi bi-chat-dots"></i> Try AI Tutor
                    </a>
                </div>
            </div>
        `;
    }
    
    getStyleDescription(style) {
        const descriptions = {
            'visual': 'visual aids, diagrams, and seeing information',
            'auditory': 'listening, discussions, and hearing information',
            'kinesthetic': 'hands-on activities, movement, and doing'
        };
        return descriptions[style] || 'mixed learning approaches';
    }
    
    initializeDashboard() {
        console.log('Initializing dashboard...');
        this.loadPersonalizedContent();
        this.initializeProgressCharts();
    }
    
    async loadPersonalizedContent() {
        const contentContainer = document.getElementById('personalized-content');
        if (!contentContainer) return;
        
        try {
            const response = await fetch('/api/content');
            const data = await response.json();
            
            if (response.ok) {
                this.renderContent(data.content, contentContainer);
            }
        } catch (error) {
            console.error('Failed to load content:', error);
        }
    }
    
    renderContent(contentItems, container) {
        container.innerHTML = contentItems.map(item => `
            <div class="col-md-6 col-lg-4 mb-4">
                <div class="card content-card shadow-sm h-100">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start mb-3">
                            <span class="badge bg-primary">${item.content_type}</span>
                            <span class="content-difficulty difficulty-${item.difficulty_level}">
                                ${item.difficulty_level}
                            </span>
                        </div>
                        <h5 class="card-title">${item.title}</h5>
                        <p class="card-text text-muted">${item.description}</p>
                        <div class="mt-auto">
                            <button class="btn btn-outline-primary btn-sm" onclick="app.trackContent(${item.id})">
                                <i class="bi bi-play"></i> Start Learning
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    initializeProgressCharts() {
        // Initialize any progress charts or visualizations
        const progressElements = document.querySelectorAll('.progress-ring');
        progressElements.forEach(element => {
            const percentage = element.dataset.percentage;
            this.animateProgressRing(element, percentage);
        });
    }
    
    animateProgressRing(element, percentage) {
        const circle = element.querySelector('.progress');
        const radius = circle.r.baseVal.value;
        const circumference = radius * 2 * Math.PI;
        
        circle.style.strokeDasharray = `${circumference} ${circumference}`;
        circle.style.strokeDashoffset = circumference;
        
        const offset = circumference - percentage / 100 * circumference;
        circle.style.strokeDashoffset = offset;
    }
    
    initializeChat() {
        console.log('Initializing chat...');
        this.chatContainer = document.getElementById('chat-messages');
        this.chatInput = document.getElementById('chat-input');
        this.chatForm = document.getElementById('chat-form');
        
        if (this.chatForm) {
            this.chatForm.addEventListener('submit', (e) => this.sendMessage(e));
        }
        
        // Add welcome message
        this.addMessage("Hello! I'm your AI learning tutor. I'll adapt my responses to your learning style. What would you like to learn about today?", 'assistant');
    }
    
    async sendMessage(e) {
        e.preventDefault();
        
        const message = this.chatInput.value.trim();
        if (!message) return;
        
        // Add user message
        this.addMessage(message, 'user');
        this.chatInput.value = '';
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });
            
            const data = await response.json();
            
            // Remove typing indicator
            this.hideTypingIndicator();
            
            if (response.ok) {
                this.addMessage(data.response, 'assistant');
            } else {
                this.addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
            }
        } catch (error) {
            console.error('Chat error:', error);
            this.hideTypingIndicator();
            this.addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
        }
    }
    
    addMessage(message, sender) {
        if (!this.chatContainer) return;
        
        const messageElement = document.createElement('div');
        messageElement.className = `chat-message ${sender}`;
        messageElement.innerHTML = `
            <div class="chat-avatar ${sender}">
                <i class="bi bi-${sender === 'user' ? 'person' : 'robot'}"></i>
            </div>
            <div class="chat-bubble ${sender}">
                ${message}
            </div>
        `;
        
        this.chatContainer.appendChild(messageElement);
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }
    
    showTypingIndicator() {
        if (!this.chatContainer) return;
        
        const typingElement = document.createElement('div');
        typingElement.className = 'chat-message assistant typing-indicator';
        typingElement.innerHTML = `
            <div class="chat-avatar assistant">
                <i class="bi bi-robot"></i>
            </div>
            <div class="chat-bubble assistant">
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        
        this.chatContainer.appendChild(typingElement);
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }
    
    hideTypingIndicator() {
        const typingIndicator = document.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    async trackContent(contentId) {
        try {
            await fetch('/api/progress', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    content_id: contentId,
                    completion_status: 'started',
                    time_spent: 0
                })
            });
        } catch (error) {
            console.error('Failed to track content:', error);
        }
    }
    
    showError(message) {
        const alertContainer = document.querySelector('.container:first-of-type');
        if (!alertContainer) return;
        
        const alert = document.createElement('div');
        alert.className = 'alert alert-danger alert-dismissible fade show mt-3';
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        alertContainer.insertBefore(alert, alertContainer.firstChild);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.app = new LearnStyleApp();
});

// Add some CSS for typing indicator animation
const style = document.createElement('style');
style.textContent = `
    .typing-dots {
        display: flex;
        align-items: center;
        gap: 2px;
    }
    
    .typing-dots span {
        height: 6px;
        width: 6px;
        background: #6c757d;
        border-radius: 50%;
        animation: typing 1.5s infinite;
    }
    
    .typing-dots span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dots span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
        }
        30% {
            transform: translateY(-10px);
        }
    }
`;
document.head.appendChild(style);