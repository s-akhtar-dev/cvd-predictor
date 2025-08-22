"""
Author: Sarah Akhtar
Description: Configuration file for managing file paths in the application
Course: COMP 193/293 AI in Healthcare
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
# Original realistic model (the one that was working)
REALISTIC_MODEL_PATH = os.path.join(MODELS_DIR, 'realistic_cardiovascular_model.joblib')
REALISTIC_SCALER_PATH = os.path.join(MODELS_DIR, 'realistic_scaler.joblib')
IMPROVED_MODEL_PATH = os.path.join(MODELS_DIR, 'improved_cardiovascular_model.joblib')
IMPROVED_SCALER_PATH = os.path.join(MODELS_DIR, 'improved_scaler.joblib')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_cardiovascular_model.joblib')
GOOD_MODEL_PATH = os.path.join(MODELS_DIR, 'good_cardiovascular_model.joblib')
GOOD_SCALER_PATH = os.path.join(MODELS_DIR, 'good_scaler.joblib')
# New 90%+ performance model (using real datasets)
HIGH_PERFORMANCE_MODEL_PATH = os.path.join(MODELS_DIR, 'real_data_90_percent_model.joblib')
HIGH_PERFORMANCE_SCALER_PATH = os.path.join(MODELS_DIR, 'real_data_90_percent_scaler.joblib')
# Simple realistic model (balanced predictions)
SIMPLE_REALISTIC_MODEL_PATH = os.path.join(MODELS_DIR, 'simple_realistic_model.joblib')
SIMPLE_REALISTIC_SCALER_PATH = os.path.join(MODELS_DIR, 'simple_realistic_scaler.joblib')
# Calibrated 90% model (high performance + realistic predictions)
CALIBRATED_90_MODEL_PATH = os.path.join(MODELS_DIR, 'calibrated_90_percent_model.joblib')
CALIBRATED_90_SCALER_PATH = os.path.join(MODELS_DIR, 'calibrated_90_percent_scaler.joblib')

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
COMBINED_DATA_PATH = os.path.join(DATA_DIR, 'combined_cardiovascular_data.csv')
HEART_DATA_PATH = os.path.join(DATA_DIR, 'heart_2020_cleaned.csv')
CVD_DATA_PATH = os.path.join(DATA_DIR, 'CVD_cleaned.csv')
CARDIO_DATA_PATH = os.path.join(DATA_DIR, 'Cardiovascular_Disease_Risk_Dataset.csv')
CARDIO_TRAIN_PATH = os.path.join(DATA_DIR, 'cardio_train.csv')

# Static paths
STATIC_DIR = os.path.join(BASE_DIR, 'static')
IMAGES_DIR = os.path.join(STATIC_DIR, 'images')

# Template paths
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

# Scripts paths
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
UTILS_DIR = os.path.join(BASE_DIR, 'utils')

# Documentation paths
DOCS_DIR = os.path.join(BASE_DIR, 'docs')
