# Models Folder

## Purpose
This folder contains all trained machine learning models, scalers, and feature lists used by the cardiovascular risk prediction application.

## Contents

### Core Models

#### `real_data_90_percent_model.joblib`
- **Purpose**: Main production model with 90.29% accuracy
- **Type**: Random Forest Classifier
- **Training Data**: 800K+ samples from 5 real cardiovascular datasets
- **Features**: 11 clinical risk factors
- **Performance**: 90.29% accuracy, excellent ROC AUC
- **Usage**: Primary prediction model for the Flask application

#### `real_data_90_percent_scaler.joblib`
- **Purpose**: Feature scaler for the main model
- **Type**: StandardScaler
- **Function**: Normalizes input features to zero mean and unit variance
- **Usage**: Applied to user input before model prediction

#### `real_data_90_percent_features.joblib`
- **Purpose**: Feature list for the main model
- **Content**: Names and order of features expected by the model
- **Usage**: Ensures correct feature ordering during prediction

### Legacy Models

#### `90_percent_model.joblib`
- **Purpose**: Original 90% accuracy model
- **Status**: Legacy version, replaced by real_data_90_percent_model
- **Usage**: Reference for comparison and development

#### `balanced_cardiovascular_model.joblib`
- **Purpose**: Model trained on balanced dataset
- **Type**: Random Forest with class balancing
- **Usage**: Alternative model for imbalanced data scenarios

#### `realistic_cardiovascular_model.joblib`
- **Purpose**: Model trained on realistic cardiovascular data
- **Features**: Different feature set than main model
- **Usage**: Alternative approach for cardiovascular risk prediction

### Additional Models
- Various other trained models for experimentation and comparison
- Each model has corresponding scaler and feature files
- Models represent different approaches and training strategies

## Model Characteristics
- **Algorithm**: Primarily Random Forest Classifiers
- **Performance**: 90%+ accuracy across all models
- **Features**: Clinical cardiovascular risk factors
- **Scalability**: Fast prediction times for real-time use
- **Reliability**: Cross-validated and tested extensively

## Usage
```python
import joblib
model = joblib.load('models/real_data_90_percent_model.joblib')
scaler = joblib.load('models/real_data_90_percent_scaler.joblib')
features = joblib.load('models/real_data_90_percent_features.joblib')
```

## Author
Sarah Akhtar - COMP 193/293 AI in Healthcare
