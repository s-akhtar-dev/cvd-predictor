#!/usr/bin/env python3
"""
Author: Sarah Akhtar
Description: 3 Perfect Test Cases for Presentation
This script provides exactly the examples you need:
- Low Risk: 15-25%
- Moderate Risk: 45-55% 
- High Risk: 70-80%
Course: COMP 193/293 AI in Healthcare
"""

import requests
import json

def test_presentation_cases():
    """Test 3 specific cases perfect for the presentation"""
    print("🎯 PERFECT PRESENTATION TEST CASES")
    print("=" * 60)
    print("🎯 Goal: Low, Moderate, and High Risk Examples")
    print("=" * 60)
    
    # Test Case 1: LOW RISK - Young, healthy person
    low_risk_case = {
        'sex': 0,  # Female
        'age_years': 28,
        'height_cm': 165,
        'weight_kg': 58,
        'BMI': 21.3,
        'smoking': 0,
        'physical_activity': 1,
        'cholesterol': 170,
        'diabetes': 'No',
        'alcohol_consumption': 3,
        'general_health': 2
    }
    
    # Test Case 2: MODERATE RISK - Middle-aged with some risk factors
    moderate_risk_case = {
        'sex': 1,  # Male
        'age_years': 52,
        'height_cm': 175,
        'weight_kg': 82,
        'BMI': 26.8,
        'smoking': 0,
        'physical_activity': 1,
        'cholesterol': 240,
        'diabetes': 'No',
        'alcohol_consumption': 10,
        'general_health': 3
    }
    
    # Test Case 3: HIGH RISK - Elderly with multiple risk factors
    high_risk_case = {
        'sex': 1,  # Male
        'age_years': 68,
        'height_cm': 170,
        'weight_kg': 88,
        'BMI': 30.4,
        'smoking': 1,
        'physical_activity': 0,
        'cholesterol': 280,
        'diabetes': 'Yes',
        'alcohol_consumption': 18,
        'general_health': 4
    }
    
    test_cases = [
        (low_risk_case, "LOW RISK - Young Healthy Female", "15-25%"),
        (moderate_risk_case, "MODERATE RISK - Middle-aged Male", "45-55%"),
        (high_risk_case, "HIGH RISK - Elderly Male with Risk Factors", "70-80%")
    ]
    
    print(f"{'Case':<45} {'Prediction':<15} {'Target Range':<15} {'Status':<15}")
    print("-" * 95)
    
    results = []
    
    for test_data, description, target_range in test_cases:
        try:
            # Make prediction request
            response = requests.post(
                'http://localhost:5003/predict',
                json=test_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                prob = result.get('risk_probability', 0)
                category = result.get('risk_category', 'Unknown')
                
                # Assess if prediction is in target range
                if "15-25%" in target_range:
                    status = "✅ Perfect" if 0.15 <= prob <= 0.25 else "⚠️ Adjust"
                elif "45-55%" in target_range:
                    status = "✅ Perfect" if 0.45 <= prob <= 0.55 else "⚠️ Adjust"
                elif "70-80%" in target_range:
                    status = "✅ Perfect" if 0.70 <= prob <= 0.80 else "⚠️ Adjust"
                else:
                    status = "Unknown"
                
                print(f"{description:<45} {prob:.1%}           {target_range:<15} {status:<15}")
                
                # Store results for summary
                results.append({
                    'case': description,
                    'prediction': prob,
                    'target': target_range,
                    'status': status,
                    'data': test_data
                })
                
            else:
                print(f"{description:<45} {'ERROR':<15} {'Failed':<15} {'Request Failed':<15}")
                
        except Exception as e:
            print(f"{description:<45} {'ERROR':<15} {'Exception':<15} {str(e)[:15]:<15}")
    
    print(f"\n🎯 PRESENTATION TESTING COMPLETED!")
    
    # Print detailed results for presentation
    print(f"\n📊 PRESENTATION-READY RESULTS:")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['case']}")
        print(f"   📊 Risk Prediction: {result['prediction']:.1%}")
        print(f"   🎯 Target Range: {result['target']}")
        print(f"   ✅ Status: {result['status']}")
        print(f"   📋 Key Factors:")
        
        data = result['data']
        if i == 1:  # Low risk
            print(f"      • Age: {data['age_years']} years (young)")
            print(f"      • BMI: {data['BMI']:.1f} (healthy weight)")
            print(f"      • Non-smoker, physically active")
            print(f"      • Good cholesterol: {data['cholesterol']} mg/dL")
        elif i == 2:  # Moderate risk
            print(f"      • Age: {data['age_years']} years (middle-aged)")
            print(f"      • BMI: {data['BMI']:.1f} (slightly overweight)")
            print(f"      • Elevated cholesterol: {data['cholesterol']} mg/dL")
            print(f"      • Some lifestyle risk factors")
        else:  # High risk
            print(f"      • Age: {data['age_years']} years (elderly)")
            print(f"      • BMI: {data['BMI']:.1f} (obese)")
            print(f"      • Smoker, sedentary lifestyle")
            print(f"      • High cholesterol: {data['cholesterol']} mg/dL")
            print(f"      • Diabetes present")
    
    print(f"\n🎉 READY FOR PRESENTATION!")
    print(f"✅ You now have 3 representative examples showing:")
    print(f"   • Low Risk: {results[0]['prediction']:.1%} (Young healthy person)")
    print(f"   • Moderate Risk: {results[1]['prediction']:.1%} (Middle-aged with risk factors)")
    print(f"   • High Risk: {results[2]['prediction']:.1%} (Elderly with multiple risk factors)")

if __name__ == "__main__":
    test_presentation_cases()
