# Scripts Folder

## Purpose
This folder contains Python scripts for model training, testing, analysis, and utility functions for the cardiovascular risk prediction system.

## Contents

### Model Creation Scripts

#### `create_normalized_model.py`
- **Purpose**: Creates the main normalized model with user-friendly risk ranges
- **Function**: Trains Random Forest on real data + synthetic data
- **Output**: 90%+ accuracy model with normalized predictions (15-25%, 45-55%, 70-80%)
- **Usage**: Primary model training script

#### `create_simple_normalized_model.py`
- **Purpose**: Simplified version of normalized model creation
- **Function**: Streamlined training process for quick model generation
- **Usage**: Development and testing of normalization approaches

#### `create_realistic_model.py`
- **Purpose**: Creates model trained on realistic cardiovascular data
- **Function**: Alternative training approach with different feature engineering
- **Usage**: Comparison with main model

#### `create_balanced_model.py`
- **Purpose**: Creates model with balanced class distribution
- **Function**: Addresses imbalanced dataset issues
- **Usage**: Alternative approach for skewed data

### Analysis and Testing Scripts

#### `test_normalized_presentation_cases.py`
- **Purpose**: Tests the normalized model with presentation test cases
- **Function**: Validates normalization function and user-friendly ranges
- **Output**: Before/after normalization comparisons
- **Usage**: Quality assurance and demonstration

#### `check_model_performance.py`
- **Purpose**: Comprehensive model performance analysis
- **Function**: Evaluates accuracy, precision, recall, F1-score, cross-validation
- **Output**: Performance metrics and comparison tables
- **Usage**: Model evaluation and validation

#### `check_realistic_features.py`
- **Purpose**: Inspects realistic model features and structure
- **Function**: Analyzes model attributes and feature names
- **Usage**: Debugging and feature analysis

### Chart Generation Scripts

#### `generate_normalized_model_charts.py`
- **Purpose**: Creates 5 core charts for the normalized model
- **Function**: Generates performance overview, feature importance, data diversity, normalization comparison, validation charts
- **Output**: PNG charts in charts/normalized_model/
- **Usage**: Presentation and documentation

#### `generate_additional_model_charts.py`
- **Purpose**: Creates additional performance and analysis charts
- **Function**: Generates confusion matrix, ROC curve, precision-recall, prediction distribution, model comparison, cross-validation charts
- **Output**: 6 additional PNG charts
- **Usage**: Comprehensive model analysis and presentation

### Utility Scripts

#### `app.py`
- **Purpose**: Flask application entry point
- **Function**: Web server for cardiovascular risk assessment
- **Usage**: Production deployment and user interface

#### `presentation_test_cases.py`
- **Purpose**: Test cases for presentation and demonstration
- **Function**: Validates model performance on specific scenarios
- **Usage**: Quality assurance and demonstrations

## Script Categories

### **Training Scripts**
- Model creation and training
- Feature engineering
- Data preprocessing

### **Analysis Scripts**
- Performance evaluation
- Model comparison
- Statistical analysis

### **Visualization Scripts**
- Chart generation
- Performance visualization
- Data representation

### **Testing Scripts**
- Model validation
- Test case execution
- Quality assurance

## Usage Patterns
```bash
# Train a new model
python scripts/create_normalized_model.py

# Test model performance
python scripts/test_normalized_presentation_cases.py

# Generate charts
python scripts/generate_normalized_model_charts.py

# Run Flask app
python scripts/app.py
```

## Author
Sarah Akhtar - COMP 193/293 AI in Healthcare
