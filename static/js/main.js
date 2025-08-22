// Main JavaScript file for CVD Predictor

// Global variables
let currentUserData = {};
let currentPrediction = null;

// Utility functions
function formatPercentage(value) {
    return (value * 100).toFixed(1) + '%';
}

function formatNumber(value, decimals = 2) {
    return parseFloat(value).toFixed(decimals);
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

function validateForm(formData) {
    const errors = [];
    
    // Required field validation
    const requiredFields = ['age_years', 'sex', 'height_cm', 'weight_kg', 'smoking', 'physical_activity', 'cholesterol', 'general_health'];
    
    requiredFields.forEach(field => {
        if (!formData[field] || formData[field] === '') {
            errors.push(`${field.replace('_', ' ')} is required`);
        }
    });
    
    // Range validation
    if (formData.age_years && (formData.age_years < 18 || formData.age_years > 120)) {
        errors.push('Age must be between 18 and 120 years');
    }
    
    if (formData.height_cm && (formData.height_cm < 100 || formData.height_cm > 250)) {
        errors.push('Height must be between 100 and 250 cm');
    }
    
    if (formData.weight_kg && (formData.weight_kg < 30 || formData.weight_kg > 300)) {
        errors.push('Weight must be between 30 and 300 kg');
    }
    
    if (formData.cholesterol && (formData.cholesterol < 100 || formData.cholesterol > 600)) {
        errors.push('Cholesterol must be between 100 and 600 mg/dL');
    }
    
    return errors;
}

function calculateBMI(height, weight) {
    if (height && weight) {
        const heightM = height / 100;
        return weight / (heightM * heightM);
    }
    return null;
}

function getRiskCategory(probability) {
    if (probability <= 0.33) return 'Low';
    if (probability <= 0.67) return 'Moderate';
    return 'High';
}

function getRiskColor(riskCategory) {
    switch (riskCategory) {
        case 'Low': return 'success';
        case 'Moderate': return 'warning';
        case 'High': return 'danger';
        default: return 'secondary';
    }
}

// Chart utility functions
function createRiskComparisonChart(userRisk, populationRisk = 0.25) {
    const data = [{
        x: ['Your Risk', 'Population Average'],
        y: [userRisk * 100, populationRisk * 100],
        type: 'bar',
        marker: {
            color: [getRiskColor(getRiskCategory(userRisk)) === 'success' ? '#198754' : 
                   getRiskColor(getRiskCategory(userRisk)) === 'warning' ? '#ffc107' : '#dc3545', '#6c757d']
        }
    }];
    
    const layout = {
        title: 'Your Risk vs Population Average',
        xaxis: { title: 'Risk Level' },
        yaxis: { 
            title: 'Risk Percentage (%)',
            range: [0, Math.max(userRisk * 100, populationRisk * 100) * 1.2]
        },
        height: 400,
        margin: { t: 50, b: 50, l: 60, r: 20 }
    };
    
    return { data, layout };
}

function createRiskDistributionChart(riskProbability) {
    const data = [{
        values: [riskProbability * 100, (1 - riskProbability) * 100],
        labels: ['Your Risk', 'Remaining Risk'],
        type: 'pie',
        marker: {
            colors: ['#ff6b6b', '#4ecdc4']
        },
        textinfo: 'label+percent',
        textposition: 'outside'
    }];
    
    const layout = {
        title: 'Risk Distribution',
        height: 300,
        showlegend: false
    };
    
    return { data, layout };
}

// Form handling functions
function collectFormData(formElement) {
    const formData = new FormData(formElement);
    const data = {};
    
    formData.forEach((value, key) => {
        // Convert numeric values
        if (['age_years', 'height_cm', 'weight_kg', 'bmi', 'cholesterol', 'alcohol_consumption'].includes(key)) {
            data[key] = parseFloat(value) || 0;
        } else if (['sex', 'smoking', 'physical_activity', 'skin_cancer', 'general_health', 'age_category'].includes(key)) {
            data[key] = parseInt(value) || 0;
        } else {
            data[key] = value;
        }
    });
    
    return data;
}

function updateFormValidation(formElement) {
    const inputs = formElement.querySelectorAll('input, select');
    let isValid = true;
    
    inputs.forEach(input => {
        if (input.hasAttribute('required') && !input.value) {
            input.classList.add('is-invalid');
            isValid = false;
        } else {
            input.classList.remove('is-invalid');
        }
    });
    
    return isValid;
}

// Data persistence functions
function saveUserData(data) {
    try {
        localStorage.setItem('cvd_user_data', JSON.stringify(data));
        currentUserData = data;
    } catch (error) {
        console.error('Error saving user data:', error);
    }
}

function loadUserData() {
    try {
        const data = localStorage.getItem('cvd_user_data');
        if (data) {
            currentUserData = JSON.parse(data);
            return currentUserData;
        }
    } catch (error) {
        console.error('Error loading user data:', error);
    }
    return null;
}

function clearUserData() {
    try {
        localStorage.removeItem('cvd_user_data');
        currentUserData = {};
    } catch (error) {
        console.error('Error clearing user data:', error);
    }
}

// Navigation functions
function navigateToPage(page) {
    window.location.href = page;
}

function goBack() {
    window.history.back();
}

// Print functions
function printResults() {
    window.print();
}

function exportResults(format = 'pdf') {
    // Placeholder for export functionality
    showNotification('Export functionality coming soon!', 'info');
}

// Accessibility functions
function setHighContrast() {
    document.body.classList.toggle('high-contrast');
    const isHighContrast = document.body.classList.contains('high-contrast');
            localStorage.setItem('cvd_high_contrast', isHighContrast);
}

function setLargeText() {
    document.body.classList.toggle('large-text');
    const isLargeText = document.body.classList.contains('large-text');
            localStorage.setItem('cvd_large_text', isLargeText);
}

// Initialize accessibility settings
function initAccessibility() {
            const highContrast = localStorage.getItem('cvd_high_contrast') === 'true';
        const largeText = localStorage.getItem('cvd_large_text') === 'true';
    
    if (highContrast) document.body.classList.add('high-contrast');
    if (largeText) document.body.classList.add('large-text');
}

// Error handling
function handleError(error, context = '') {
    console.error(`Error in ${context}:`, error);
    
    let userMessage = 'An unexpected error occurred. Please try again.';
    
    if (error.message) {
        userMessage = error.message;
    } else if (error.status) {
        switch (error.status) {
            case 400:
                userMessage = 'Invalid request. Please check your input.';
                break;
            case 500:
                userMessage = 'Server error. Please try again later.';
                break;
            default:
                userMessage = `Error ${error.status}: ${error.statusText || 'Unknown error'}`;
        }
    }
    
    showNotification(userMessage, 'danger');
}

// Performance monitoring
function measurePerformance(operation, callback) {
    const start = performance.now();
    
    try {
        const result = callback();
        const end = performance.now();
        console.log(`${operation} took ${(end - start).toFixed(2)}ms`);
        return result;
    } catch (error) {
        const end = performance.now();
        console.error(`${operation} failed after ${(end - start).toFixed(2)}ms:`, error);
        throw error;
    }
}

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('CVD Predictor initialized');
    
    // Initialize accessibility settings
    initAccessibility();
    
    // Load any saved user data
    loadUserData();
    
    // Add global error handler
    window.addEventListener('error', function(event) {
        handleError(event.error, 'Global');
    });
    
    // Add unhandled promise rejection handler
    window.addEventListener('unhandledrejection', function(event) {
        handleError(event.reason, 'Promise');
    });
    
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Add smooth scrolling for anchor links
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
});

// Export functions for use in other scripts
window.CVDUtils = {
    formatPercentage,
    formatNumber,
    showNotification,
    validateForm,
    calculateBMI,
    getRiskCategory,
    getRiskColor,
    createRiskComparisonChart,
    createRiskDistributionChart,
    collectFormData,
    updateFormValidation,
    saveUserData,
    loadUserData,
    clearUserData,
    navigateToPage,
    goBack,
    printResults,
    exportResults,
    setHighContrast,
    setLargeText,
    handleError,
    measurePerformance
};

