#!/usr/bin/env python3
"""
Author: Sarah Akhtar
Description: Debug the preprocessing function to see why it's only creating 13 features
Course: COMP 193/293 AI in Healthcare
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_preprocessing():
    """Debug the preprocessing function step by step"""
    print("🔍 DEBUGGING PREPROCESSING FUNCTION")
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
    
    # Simulate the preprocessing step by step
    features = []
    
    print(f"\n🔧 Building features step by step:")
    
    # 1. Age
    age_years = user_data.get('age_years', 45)
    age_normalized = (age_years - 18) / (100 - 18)
    features.append(age_normalized)
    print(f"  1. age_normalized: {age_normalized:.4f}")
    
    # 2. Gender
    gender = user_data.get('sex', 0)
    features.append(gender)
    print(f"  2. gender: {gender}")
    
    # 3. BMI
    bmi = user_data.get('BMI', 25)
    if bmi < 16:
        bmi = 16
    elif bmi > 50:
        bmi = 50
    bmi_normalized = (bmi - 16) / (50 - 16)
    features.append(bmi_normalized)
    print(f"  3. bmi_normalized: {bmi_normalized:.4f}")
    
    # 4. Smoking
    smoking = user_data.get('smoking', 0)
    features.append(smoking)
    print(f"  4. smoking: {smoking}")
    
    # 5. Physical activity
    physical_activity = user_data.get('physical_activity', 1)
    features.append(physical_activity)
    print(f"  5. physical_activity: {physical_activity}")
    
    # 6. Cholesterol
    cholesterol = user_data.get('cholesterol', 200)
    if cholesterol < 100:
        cholesterol = 100
    elif cholesterol > 400:
        cholesterol = 400
    cholesterol_normalized = (cholesterol - 100) / (400 - 100)
    features.append(cholesterol_normalized)
    print(f"  6. cholesterol_normalized: {cholesterol_normalized:.4f}")
    
    # 7. Diabetes
    diabetes = user_data.get('diabetes', 'No')
    if isinstance(diabetes, str):
        diabetes = 1 if diabetes == 'Yes' else 0
    features.append(diabetes)
    print(f"  7. diabetes: {diabetes}")
    
    # 8. Alcohol
    alcohol_consumption = user_data.get('alcohol_consumption', 0)
    alcohol_normalized = min(alcohol_consumption / 30.0, 1.0)
    features.append(alcohol_normalized)
    print(f"  8. alcohol_normalized: {alcohol_normalized:.4f}")
    
    # 9. Health
    general_health = user_data.get('general_health', 2)
    health_normalized = general_health / 4.0
    features.append(health_normalized)
    print(f"  9. health_normalized: {health_normalized:.4f}")
    
    # 10. Age-BMI interaction
    age_bmi_interaction = age_normalized * bmi_normalized
    features.append(age_bmi_interaction)
    print(f" 10. age_bmi_interaction: {age_bmi_interaction:.4f}")
    
    # 11. Age-smoking interaction
    age_smoking_interaction = age_normalized * smoking
    features.append(age_smoking_interaction)
    print(f" 11. age_smoking_interaction: {age_smoking_interaction:.4f}")
    
    # 12. BMI-cholesterol interaction
    bmi_cholesterol_interaction = bmi_normalized * cholesterol_normalized
    features.append(bmi_cholesterol_interaction)
    print(f" 12. bmi_cholesterol_interaction: {bmi_cholesterol_interaction:.4f}")
    
    # 13. Smoking-diabetes interaction
    smoking_diabetes_interaction = smoking * diabetes
    features.append(smoking_diabetes_interaction)
    print(f" 13. smoking_diabetes_interaction: {smoking_diabetes_interaction:.4f}")
    
    # 14. Age squared
    age_squared = age_normalized ** 2
    features.append(age_squared)
    print(f" 14. age_squared: {age_squared:.4f}")
    
    # 15. BMI squared
    bmi_squared = bmi_normalized ** 2
    features.append(bmi_squared)
    print(f" 15. bmi_squared: {bmi_squared:.4f}")
    
    # 16. Cholesterol squared
    cholesterol_squared = cholesterol_normalized ** 2
    features.append(cholesterol_squared)
    print(f" 16. cholesterol_squared: {cholesterol_squared:.4f}")
    
    # 17. Risk score
    risk_score = (
        age_normalized * 0.3 +
        gender * 0.15 +
        bmi_normalized * 0.25 +
        smoking * 0.35 +
        (1 - physical_activity) * 0.2 +
        cholesterol_normalized * 0.2 +
        diabetes * 0.4 +
        smoking_diabetes_interaction * 0.3 +
        age_smoking_interaction * 0.25
    )
    features.append(risk_score)
    print(f" 17. risk_score: {risk_score:.4f}")
    
    print(f"\n📊 Summary:")
    print(f"  Total features created: {len(features)}")
    print(f"  Expected: 17")
    
    if len(features) == 17:
        print(f"  ✅ SUCCESS! All 17 features created")
    else:
        print(f"  ❌ ERROR! Missing {17 - len(features)} features")
    
    return len(features) == 17

if __name__ == "__main__":
    debug_preprocessing()

