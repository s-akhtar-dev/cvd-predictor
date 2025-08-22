#!/usr/bin/env python3
"""
Author: Sarah Akhtar
Description: Test the preprocessing function to ensure it creates exactly 17 features
Course: COMP 193/293 AI in Healthcare
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import preprocess_user_input

def test_preprocessing():
    """Test the preprocessing function"""
    print("🧪 TESTING PREPROCESSING FUNCTION")
    print("=" * 50)
    
    # Test data
    user_data = {
        'age_years': 45,
        'sex': 1,
        'BMI': 25,
        'smoking': 0,
        'physical_activity': 1,
        'cholesterol': 200,
        'diabetes': 'No',
        'alcohol_consumption': 5,
        'general_health': 2
    }
    
    print(f"Input data: {user_data}")
    
    # Get features
    features = preprocess_user_input(user_data)
    
    print(f"\nFeatures shape: {features.shape}")
    print(f"Number of features: {features.shape[1]}")
    
    # Expected feature names
    expected_features = [
        'age_normalized', 'gender', 'bmi_normalized', 'smoking', 'physical_activity',
        'cholesterol_normalized', 'diabetes', 'alcohol_normalized', 'health_normalized',
        'age_bmi_interaction', 'age_smoking_interaction', 'bmi_cholesterol_interaction',
        'smoking_diabetes_interaction', 'age_squared', 'bmi_squared', 'cholesterol_squared',
        'risk_score'
    ]
    
    print(f"\nExpected features ({len(expected_features)}):")
    for i, feature_name in enumerate(expected_features):
        print(f"  {i+1:2d}. {feature_name}: {features[0, i]:.4f}")
    
    if features.shape[1] == 17:
        print(f"\n✅ SUCCESS! Preprocessing creates exactly 17 features")
    else:
        print(f"\n❌ ERROR! Expected 17 features, got {features.shape[1]}")
    
    return features.shape[1] == 17

if __name__ == "__main__":
    test_preprocessing()

