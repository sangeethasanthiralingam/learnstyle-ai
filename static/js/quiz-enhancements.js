// Quiz Enhancements - Vanilla JavaScript
document.addEventListener('DOMContentLoaded', function() {
    initQuizEnhancements();
});

function initQuizEnhancements() {
    // Initialize quiz features
    initProgressTracking();
    initQuestionNavigation();
    initAnswerValidation();
    initSmoothTransitions();
    initAutoSave();
}

// Progress Tracking
function initProgressTracking() {
    const questions = document.querySelectorAll('.question-card');
    const progressBar = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');
    
    if (!progressBar || !progressText) return;
    
    const totalQuestions = questions.length;
    let answeredQuestions = 0;
    
    // Update progress when answers change
    document.addEventListener('change', function(e) {
        if (e.target.type === 'radio' && e.target.name.startsWith('question_')) {
            updateProgress();
        }
    });
    
    function updateProgress() {
        answeredQuestions = document.querySelectorAll('input[type="radio"]:checked').length;
        const percentage = (answeredQuestions / totalQuestions) * 100;
        
        progressBar.style.width = percentage + '%';
        progressText.textContent = `Question ${answeredQuestions} of ${totalQuestions}`;
        
        // Update question indicators
        updateQuestionIndicators();
    }
    
    function updateQuestionIndicators() {
        questions.forEach((question, index) => {
            const questionNumber = index + 1;
            const isAnswered = document.querySelector(`input[name="question_${questionNumber}"]:checked`);
            
            const indicator = question.querySelector('.question-indicator');
            if (indicator) {
                indicator.className = `question-indicator ${isAnswered ? 'answered' : 'unanswered'}`;
            }
        });
    }
    
    // Initial progress update
    updateProgress();
}

// Question Navigation
function initQuestionNavigation() {
    // Add navigation buttons if they don't exist
    addNavigationButtons();
    
    // Handle navigation
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('nav-previous')) {
            navigateToPrevious();
        } else if (e.target.classList.contains('nav-next')) {
            navigateToNext();
        }
    });
}

function addNavigationButtons() {
    const quizContainer = document.querySelector('.quiz-container');
    if (!quizContainer || document.querySelector('.quiz-navigation')) return;
    
    const navigation = document.createElement('div');
    navigation.className = 'quiz-navigation';
    navigation.innerHTML = `
        <button class="btn btn-outline-primary nav-previous" disabled>
            <i class="fas fa-arrow-left"></i> Previous
        </button>
        <button class="btn btn-primary nav-next">
            Next <i class="fas fa-arrow-right"></i>
        </button>
    `;
    
    quizContainer.appendChild(navigation);
}

function navigateToPrevious() {
    const currentQuestion = getCurrentQuestion();
    if (currentQuestion > 0) {
        showQuestion(currentQuestion - 1);
    }
}

function navigateToNext() {
    const currentQuestion = getCurrentQuestion();
    const totalQuestions = document.querySelectorAll('.question-card').length;
    
    if (currentQuestion < totalQuestions - 1) {
        showQuestion(currentQuestion + 1);
    } else {
        // Check if all questions are answered before submitting
        if (validateAllQuestions()) {
            submitQuiz();
        } else {
            showNotification('Please answer all questions before submitting', 'error');
        }
    }
}

function getCurrentQuestion() {
    const visibleQuestion = document.querySelector('.question-card:not(.hidden)');
    if (!visibleQuestion) return 0;
    
    const questions = Array.from(document.querySelectorAll('.question-card'));
    return questions.indexOf(visibleQuestion);
}

function showQuestion(questionIndex) {
    const questions = document.querySelectorAll('.question-card');
    
    questions.forEach((question, index) => {
        if (index === questionIndex) {
            question.classList.remove('hidden');
            question.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else {
            question.classList.add('hidden');
        }
    });
    
    updateNavigationButtons(questionIndex, questions.length);
}

function updateNavigationButtons(currentIndex, totalQuestions) {
    const prevBtn = document.querySelector('.nav-previous');
    const nextBtn = document.querySelector('.nav-next');
    
    if (prevBtn) {
        prevBtn.disabled = currentIndex === 0;
    }
    
    if (nextBtn) {
        if (currentIndex === totalQuestions - 1) {
            nextBtn.innerHTML = '<i class="fas fa-check"></i> Submit Quiz';
            nextBtn.classList.add('btn-success');
            nextBtn.classList.remove('btn-primary');
        } else {
            nextBtn.innerHTML = 'Next <i class="fas fa-arrow-right"></i>';
            nextBtn.classList.add('btn-primary');
            nextBtn.classList.remove('btn-success');
        }
    }
}

// Answer Validation
function initAnswerValidation() {
    // Real-time validation
    document.addEventListener('change', function(e) {
        if (e.target.type === 'radio') {
            validateQuestion(e.target.name);
        }
    });
    
    // Form submission validation
    const form = document.querySelector('form[action="/submit_quiz"]');
    if (form) {
        form.addEventListener('submit', function(e) {
            if (!validateAllQuestions()) {
                e.preventDefault();
                showNotification('Please answer all questions before submitting', 'error');
            }
        });
    }
}

function validateQuestion(questionName) {
    const questionNumber = questionName.split('_')[1];
    const questionCard = document.querySelector(`input[name="${questionName}"]`).closest('.question-card');
    
    if (questionCard) {
        const isAnswered = document.querySelector(`input[name="${questionName}"]:checked`);
        
        if (isAnswered) {
            questionCard.classList.remove('unanswered');
            questionCard.classList.add('answered');
        } else {
            questionCard.classList.remove('answered');
            questionCard.classList.add('unanswered');
        }
    }
}

function validateAllQuestions() {
    const totalQuestions = document.querySelectorAll('input[name^="question_"]').length / 3; // 3 options per question
    let answeredQuestions = 0;
    
    for (let i = 1; i <= totalQuestions; i++) {
        if (document.querySelector(`input[name="question_${i}"]:checked`)) {
            answeredQuestions++;
        }
    }
    
    return answeredQuestions === totalQuestions;
}

// Smooth Transitions
function initSmoothTransitions() {
    // Add transition styles
    const transitionStyles = `
        <style>
        .question-card {
            transition: all 0.3s ease;
        }
        
        .question-card.hidden {
            opacity: 0;
            transform: translateX(100%);
            pointer-events: none;
        }
        
        .question-card.answered {
            border-left: 4px solid #28a745;
            background: linear-gradient(135deg, #f8fff8 0%, #e8f5e8 100%);
        }
        
        .question-card.unanswered {
            border-left: 4px solid #ffc107;
            background: linear-gradient(135deg, #fffbf0 0%, #fff3cd 100%);
        }
        
        .option-label {
            transition: all 0.3s ease;
        }
        
        .option-label:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        .option-label.selected {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transform: scale(1.02);
        }
        
        .progress-fill {
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }
        </style>
    `;
    
    document.head.insertAdjacentHTML('beforeend', transitionStyles);
    
    // Add selection effects
    document.addEventListener('change', function(e) {
        if (e.target.type === 'radio') {
            // Remove previous selection
            const previousSelected = document.querySelector('.option-label.selected');
            if (previousSelected) {
                previousSelected.classList.remove('selected');
            }
            
            // Add selection to current option
            const currentLabel = e.target.closest('.option-label');
            if (currentLabel) {
                currentLabel.classList.add('selected');
            }
        }
    });
}

// Auto-save functionality
function initAutoSave() {
    // Save answers every 30 seconds
    setInterval(saveAnswers, 30000);
    
    // Save on page unload
    window.addEventListener('beforeunload', saveAnswers);
}

function saveAnswers() {
    const answers = {};
    const radioButtons = document.querySelectorAll('input[type="radio"]:checked');
    
    radioButtons.forEach(button => {
        answers[button.name] = button.value;
    });
    
    // Save to localStorage
    localStorage.setItem('quiz_answers', JSON.stringify(answers));
    
    // Send to server if possible
    if (navigator.onLine) {
        fetch('/api/save-quiz-progress', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(answers)
        }).catch(error => {
            console.log('Auto-save failed:', error);
        });
    }
}

// Load saved answers
function loadSavedAnswers() {
    const savedAnswers = localStorage.getItem('quiz_answers');
    if (savedAnswers) {
        const answers = JSON.parse(savedAnswers);
        
        Object.keys(answers).forEach(questionName => {
            const radioButton = document.querySelector(`input[name="${questionName}"][value="${answers[questionName]}"]`);
            if (radioButton) {
                radioButton.checked = true;
                validateQuestion(questionName);
            }
        });
        
        updateProgress();
    }
}

// Initialize saved answers
document.addEventListener('DOMContentLoaded', loadSavedAnswers);

// Quiz submission
function submitQuiz() {
    if (!validateAllQuestions()) {
        showNotification('Please answer all questions before submitting', 'error');
        return;
    }
    
    const form = document.querySelector('form[action="/submit_quiz"]');
    if (form) {
        // Show loading state
        const submitBtn = document.querySelector('.nav-next');
        if (submitBtn) {
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Submitting...';
            submitBtn.disabled = true;
        }
        
        // Clear saved answers
        localStorage.removeItem('quiz_answers');
        
        // Submit form
        form.submit();
    }
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}
