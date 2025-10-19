# 🏥 **System Health Monitoring - LearnStyle AI**

## 📊 **Current Implementation**

### **Basic System Health (Current)**
```python
# In app.py line 2957-2958
system_health = 98.5  # Hardcoded value
```

**Current Status:** ❌ **Static/Simulated** - The system health is currently hardcoded as `98.5%` and doesn't reflect actual system performance.

---

## 🔧 **Comprehensive System Health Monitoring**

### **1. Real-Time Metrics Collection**

#### **A. Server Performance Metrics**
```python
def calculate_server_health():
    """Calculate server performance health"""
    metrics = {
        'cpu_usage': psutil.cpu_percent(interval=1),
        'memory_usage': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'network_io': psutil.net_io_counters(),
        'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
    }
    
    # Calculate health score (0-100)
    health_score = 100
    if metrics['cpu_usage'] > 80: health_score -= 20
    if metrics['memory_usage'] > 85: health_score -= 25
    if metrics['disk_usage'] > 90: health_score -= 15
    if metrics['load_average'][0] > 4: health_score -= 20
    
    return min(100, max(0, health_score))
```

#### **B. Database Health Metrics**
```python
def calculate_database_health():
    """Calculate database performance health"""
    try:
        # Connection test
        db.session.execute('SELECT 1')
        
        # Query performance test
        start_time = time.time()
        User.query.count()
        query_time = time.time() - start_time
        
        # Connection pool status
        pool_size = db.engine.pool.size()
        checked_out = db.engine.pool.checkedout()
        
        # Calculate health score
        health_score = 100
        if query_time > 1.0: health_score -= 30  # Slow queries
        if checked_out / pool_size > 0.8: health_score -= 20  # Pool exhaustion
        
        return health_score
    except Exception as e:
        return 0  # Database down
```

#### **C. ML Models Health**
```python
def calculate_ml_models_health():
    """Calculate ML models performance health"""
    health_scores = []
    
    # Test each ML model
    models_to_test = [
        'learning_style_predictor',
        'content_generator', 
        'emotion_ai',
        'biometric_fusion',
        'predictive_analytics'
    ]
    
    for model_name in models_to_test:
        try:
            # Test model loading and prediction
            start_time = time.time()
            # ... model testing logic ...
            response_time = time.time() - start_time
            
            # Calculate model health
            model_health = 100
            if response_time > 2.0: model_health -= 40  # Slow response
            if response_time > 5.0: model_health -= 60  # Very slow
            
            health_scores.append(model_health)
        except Exception as e:
            health_scores.append(0)  # Model failed
    
    return sum(health_scores) / len(health_scores) if health_scores else 0
```

#### **D. API Endpoints Health**
```python
def calculate_api_health():
    """Calculate API endpoints performance health"""
    endpoints_to_test = [
        '/api/predict',
        '/api/content-recommendations',
        '/api/emotion-ai/analyze',
        '/api/biometric/fusion-analysis',
        '/api/minimal-tracking/data'
    ]
    
    health_scores = []
    for endpoint in endpoints_to_test:
        try:
            start_time = time.time()
            response = requests.get(f'http://localhost:5000{endpoint}', timeout=5)
            response_time = time.time() - start_time
            
            # Calculate endpoint health
            endpoint_health = 100
            if response.status_code != 200: endpoint_health -= 50
            if response_time > 3.0: endpoint_health -= 30
            if response_time > 5.0: endpoint_health -= 50
            
            health_scores.append(endpoint_health)
        except Exception as e:
            health_scores.append(0)  # Endpoint failed
    
    return sum(health_scores) / len(health_scores) if health_scores else 0
```

### **2. User Experience Metrics**

#### **A. Learning Session Quality**
```python
def calculate_learning_quality_health():
    """Calculate learning session quality health"""
    try:
        # Get recent learning sessions
        recent_sessions = UserProgress.query.filter(
            UserProgress.timestamp >= datetime.now() - timedelta(hours=24)
        ).all()
        
        if not recent_sessions:
            return 50  # No data
        
        # Calculate quality metrics
        completion_rate = sum(1 for s in recent_sessions if s.completion_status == 'completed') / len(recent_sessions)
        avg_score = sum(s.score for s in recent_sessions) / len(recent_sessions)
        avg_time = sum(s.time_spent for s in recent_sessions) / len(recent_sessions)
        
        # Calculate health score
        health_score = 0
        health_score += completion_rate * 40  # 40% weight for completion
        health_score += (avg_score / 100) * 30  # 30% weight for scores
        health_score += min(1.0, avg_time / 3600) * 30  # 30% weight for engagement time
        
        return min(100, health_score * 100)
    except Exception as e:
        return 0
```

#### **B. System Responsiveness**
```python
def calculate_responsiveness_health():
    """Calculate system responsiveness health"""
    response_times = []
    
    # Test critical endpoints
    critical_endpoints = [
        '/',
        '/dashboard',
        '/quiz',
        '/api/predict'
    ]
    
    for endpoint in critical_endpoints:
        try:
            start_time = time.time()
            response = requests.get(f'http://localhost:5000{endpoint}', timeout=10)
            response_time = time.time() - start_time
            response_times.append(response_time)
        except:
            response_times.append(10.0)  # Timeout penalty
    
    # Calculate health based on response times
    avg_response_time = sum(response_times) / len(response_times)
    health_score = max(0, 100 - (avg_response_time * 10))  # 10 points per second
    
    return health_score
```

### **3. Overall System Health Calculation**

```python
def calculate_overall_system_health():
    """Calculate comprehensive system health score"""
    
    # Weighted health components
    health_components = {
        'server_performance': calculate_server_health() * 0.25,      # 25% weight
        'database_health': calculate_database_health() * 0.20,       # 20% weight
        'ml_models_health': calculate_ml_models_health() * 0.20,     # 20% weight
        'api_health': calculate_api_health() * 0.15,                 # 15% weight
        'learning_quality': calculate_learning_quality_health() * 0.10,  # 10% weight
        'responsiveness': calculate_responsiveness_health() * 0.10   # 10% weight
    }
    
    # Calculate weighted average
    overall_health = sum(health_components.values())
    
    # Apply system-wide penalties
    if overall_health < 50:
        overall_health *= 0.8  # 20% penalty for critical issues
    
    return {
        'overall_health': min(100, max(0, overall_health)),
        'component_health': health_components,
        'status': get_health_status(overall_health),
        'recommendations': get_health_recommendations(health_components),
        'timestamp': datetime.now().isoformat()
    }

def get_health_status(health_score):
    """Get health status based on score"""
    if health_score >= 90:
        return "Excellent"
    elif health_score >= 75:
        return "Good"
    elif health_score >= 60:
        return "Fair"
    elif health_score >= 40:
        return "Poor"
    else:
        return "Critical"

def get_health_recommendations(components):
    """Get recommendations based on health components"""
    recommendations = []
    
    if components['server_performance'] < 70:
        recommendations.append("Consider scaling server resources")
    if components['database_health'] < 70:
        recommendations.append("Optimize database queries and connections")
    if components['ml_models_health'] < 70:
        recommendations.append("Check ML model performance and retrain if needed")
    if components['api_health'] < 70:
        recommendations.append("Review API endpoint performance")
    if components['learning_quality'] < 70:
        recommendations.append("Improve learning content and user experience")
    if components['responsiveness'] < 70:
        recommendations.append("Optimize system responsiveness")
    
    return recommendations
```

---

## 🚨 **Health Monitoring Features**

### **1. Real-Time Alerts**
- **Critical Issues** (< 40%): Immediate notification
- **Warning Issues** (40-60%): Email alert
- **Performance Issues** (60-75%): Dashboard warning
- **Good Performance** (75-90%): Green status
- **Excellent Performance** (90%+): Optimal status

### **2. Historical Tracking**
- **Health trends** over time
- **Performance degradation** detection
- **Peak usage** analysis
- **Maintenance scheduling** recommendations

### **3. Automated Actions**
- **Auto-scaling** based on load
- **Cache clearing** when memory usage is high
- **Database optimization** when queries are slow
- **Model retraining** when accuracy drops

---

## 📈 **Implementation Priority**

### **Phase 1: Basic Health Monitoring** ⭐
1. ✅ Replace hardcoded `system_health = 98.5`
2. ✅ Implement server performance metrics
3. ✅ Add database health checks
4. ✅ Create health status dashboard

### **Phase 2: Advanced Monitoring** ⭐⭐
1. ✅ ML models health testing
2. ✅ API endpoints monitoring
3. ✅ User experience metrics
4. ✅ Real-time alerts system

### **Phase 3: Predictive Health** ⭐⭐⭐
1. ✅ Historical trend analysis
2. ✅ Predictive maintenance
3. ✅ Automated optimization
4. ✅ Advanced reporting

---

## 🎯 **Current Status**

**System Health Monitoring:** ❌ **Not Implemented**
- Currently using hardcoded value: `98.5%`
- No real-time monitoring
- No performance tracking
- No automated alerts

**Next Steps:**
1. Implement comprehensive health monitoring
2. Replace static value with dynamic calculation
3. Add real-time dashboard updates
4. Create alert system for critical issues

---

**The system health monitoring needs to be implemented to provide real insights into application performance and user experience!** 🏥📊
