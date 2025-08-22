#!/usr/bin/env python3
"""
Author: Sarah Akhtar
Description: Test the Flask app directly to see if it's using the new 90%+ model
Course: COMP 193/293 AI in Healthcare
"""

import requests
import json

def test_flask_app():
    """Test the Flask app directly"""
    print("🧪 TESTING FLASK APP DIRECTLY")
    print("=" * 50)
    
    # Test data
    test_data = {
        'sex': 0,
        'age_years': 25,
        'height_cm': 165,
        'weight_kg': 60,
        'BMI': 22,
        'smoking': 0,
        'physical_activity': 1,
        'cholesterol': 200,
        'diabetes': 'No',
        'alcohol_consumption': 3,
        'general_health': 2
    }
    
    print(f"Test data: {test_data}")
    
    try:
        # Make a prediction request
        response = requests.post(
            'http://localhost:5003/predict',
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Prediction successful!")
            print(f"Response: {result}")
            
            if 'risk_probability' in result:
                prob = result['risk_probability']
                category = result['risk_category']
                print(f"Risk Probability: {prob:.3f} ({prob*100:.1f}%)")
                print(f"Risk Category: {category}")
                
                # Check if this looks like the 90%+ model output
                if prob < 0.1:
                    print(f"🎯 Excellent! This looks like the 90%+ model (very low risk prediction)")
                elif prob > 0.9:
                    print(f"🎯 Excellent! This looks like the 90%+ model (very high risk prediction)")
                else:
                    print(f"📊 Moderate risk prediction")
            else:
                print(f"⚠️  No risk prediction in response")
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask app. Is it running?")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print(f"\n🎉 Flask app test completed!")

if __name__ == "__main__":
    test_flask_app()

