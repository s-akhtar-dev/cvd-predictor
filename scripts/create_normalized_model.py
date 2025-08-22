#!/usr/bin/env python3
"""
Author: Sarah Akhtar
Description: Create a normalized version of the real data 90% model with user-friendly risk ranges
Course: COMP 193/293 AI in Healthcare
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class NormalizedCardiovascularModel:
    """A wrapper class that normalizes predictions to user-friendly ranges"""
    
    def __init__(self, base_model, scaler, feature_names):
        self.base_model = base_model
        self.scaler = scaler
        self.feature_names = feature_names
        
        # Define target ranges for normalization
        self.target_ranges = {
            'low': (0.10, 0.15),      # 10-15%
            'moderate': (0.50, 0.60),  # 50-60%
            'high': (0.70, 0.80)      # 70-80%
        }
        
        # Calibrate the model to understand its current output range
        self._calibrate_ranges()
    
    def _calibrate_ranges(self):
        """Calibrate the model to understand its current output distribution"""
        print("🔧 Calibrating model output ranges...")
        
        # Generate a range of test cases to understand current output distribution
        test_cases = self._generate_calibration_cases()
        
        # Get predictions for calibration cases
        predictions = []
        for case in test_cases:
            try:
                # Scale features
                case_scaled = self.scaler.transform(case.reshape(1, -1))
                
                # Get base prediction
                if hasattr(self.base_model, 'predict_proba'):
                    pred = self.base_model.predict_proba(case_scaled)[0, 1]
                else:
                    pred = self.base_model.predict(case_scaled)[0]
                
                predictions.append(pred)
            except Exception as e:
                print(f"Warning: Error in calibration case: {e}")
                continue
        
        if predictions:
            self.min_pred = min(predictions)
            self.max_pred = max(predictions)
            self.range_pred = self.max_pred - self.min_pred
            
            print(f"📊 Current model output range: {self.min_pred:.3f} to {self.max_pred:.3f}")
            print(f"📊 Range span: {self.range_pred:.3f}")
        else:
            # Fallback values if calibration fails
            self.min_pred = 0.05
            self.max_pred = 0.35
            self.range_pred = 0.30
            print("⚠️ Using fallback calibration values")
    
    def _generate_calibration_cases(self):
        """Generate diverse test cases for calibration"""
        cases = []
        
        # Age variations
        for age in [25, 35, 45, 55, 65, 75]:
            case = np.array([age, 1, 25, 0, 1, 0, 0, 200, 130, 80, 100])
            cases.append(case)
        
        # BMI variations
        for bmi in [20, 25, 30, 35, 40]:
            case = np.array([50, 1, bmi, 0, 1, 0, 0, 200, 130, 80, 100])
            cases.append(case)
        
        # Smoking variations
        for smoking in [0, 1]:
            case = np.array([50, 1, 28, smoking, 1, 0, 0, 200, 130, 80, 100])
            cases.append(case)
        
        # Diabetes variations
        for diabetes in [0, 1]:
            case = np.array([50, 1, 28, 0, 1, diabetes, 0, 200, 130, 80, 100])
            cases.append(case)
        
        # High risk combinations
        high_risk_case = np.array([70, 1, 35, 1, 0, 1, 1, 300, 160, 100, 150])
        cases.append(high_risk_case)
        
        return np.array(cases)
    
    def normalize_prediction(self, raw_prediction):
        """Normalize a raw prediction to user-friendly ranges"""
        # First, normalize to 0-1 range
        normalized = (raw_prediction - self.min_pred) / self.range_pred
        normalized = np.clip(normalized, 0, 1)
        
        # Map to target ranges based on risk level
        if normalized <= 0.33:  # Low risk
            # Map 0-0.33 to 10-15%
            mapped = 0.10 + (normalized / 0.33) * 0.05
        elif normalized <= 0.67:  # Moderate risk
            # Map 0.33-0.67 to 50-60%
            mapped = 0.50 + ((normalized - 0.33) / 0.34) * 0.10
        else:  # High risk
            # Map 0.67-1.0 to 70-80%
            mapped = 0.70 + ((normalized - 0.67) / 0.33) * 0.10
        
        return np.clip(mapped, 0.10, 0.80)
    
    def predict_proba(self, X):
        """Get normalized probability predictions"""
        # Get base predictions
        if hasattr(self.base_model, 'predict_proba'):
            base_probs = self.base_model.predict_proba(X)
            # Normalize only the positive class (index 1)
            normalized_probs = base_probs.copy()
            normalized_probs[:, 1] = np.array([self.normalize_prediction(p) for p in base_probs[:, 1]])
            # Adjust negative class to maintain sum = 1
            normalized_probs[:, 0] = 1 - normalized_probs[:, 1]
            return normalized_probs
        else:
            # For models without predict_proba
            base_preds = self.base_model.predict(X)
            normalized_preds = np.array([self.normalize_prediction(p) for p in base_preds])
            return normalized_preds
    
    def predict(self, X):
        """Get binary predictions based on normalized probabilities"""
        probs = self.predict_proba(X)
        if len(probs.shape) > 1:
            # If probs is 2D, use the positive class probability
            return (probs[:, 1] > 0.5).astype(int)
        else:
            # If probs is 1D, use threshold
            return (probs > 0.5).astype(int)

def create_normalized_model():
    """Create and save a normalized version of the real data 90% model"""
    print("🔧 Creating Normalized Cardiovascular Risk Model")
    print("=" * 60)
    
    try:
        # Load the base model and scaler
        print("📥 Loading base model and scaler...")
        base_model = joblib.load('models/real_data_90_percent_model.joblib')
        scaler = joblib.load('models/real_data_90_percent_scaler.joblib')
        feature_names = joblib.load('models/real_data_90_percent_features.joblib')
        
        print("✅ Base model loaded successfully")
        
        # Create normalized model wrapper
        print("🔧 Creating normalized model wrapper...")
        normalized_model = NormalizedCardiovascularModel(base_model, scaler, feature_names)
        
        # Test the normalized model
        print("🧪 Testing normalized model...")
        
        # Test case 1: Low risk (should map to 10-15%)
        test_low = np.array([25, 0, 22, 0, 1, 0, 0, 180, 120, 75, 90])
        test_low_scaled = scaler.transform(test_low.reshape(1, -1))
        
        # Get base prediction
        base_pred_low = base_model.predict_proba(test_low_scaled)[0, 1]
        norm_pred_low = normalized_model.predict_proba(test_low_scaled)[0, 1]
        
        print(f"📊 Low Risk Test Case:")
        print(f"   Base prediction: {base_pred_low:.1%}")
        print(f"   Normalized prediction: {norm_pred_low:.1%}")
        
        # Test case 2: Moderate risk (should map to 50-60%)
        test_mod = np.array([55, 1, 28, 1, 1, 0, 0, 240, 140, 85, 110])
        test_mod_scaled = scaler.transform(test_mod.reshape(1, -1))
        
        base_pred_mod = base_model.predict_proba(test_mod_scaled)[0, 1]
        norm_pred_mod = normalized_model.predict_proba(test_mod_scaled)[0, 1]
        
        print(f"📊 Moderate Risk Test Case:")
        print(f"   Base prediction: {base_pred_mod:.1%}")
        print(f"   Normalized prediction: {norm_pred_mod:.1%}")
        
        # Test case 3: High risk (should map to 70-80%)
        test_high = np.array([70, 1, 32, 1, 0, 1, 1, 280, 160, 100, 140])
        test_high_scaled = scaler.transform(test_high.reshape(1, -1))
        
        base_pred_high = base_model.predict_proba(test_high_scaled)[0, 1]
        norm_pred_high = normalized_model.predict_proba(test_high_scaled)[0, 1]
        
        print(f"📊 High Risk Test Case:")
        print(f"   Base prediction: {base_pred_high:.1%}")
        print(f"   Normalized prediction: {norm_pred_high:.1%}")
        
        # Save the normalized model
        print("\n💾 Saving normalized model...")
        joblib.dump(normalized_model, 'models/normalized_90_percent_model.joblib')
        joblib.dump(scaler, 'models/normalized_90_percent_scaler.joblib')
        joblib.dump(feature_names, 'models/normalized_90_percent_features.joblib')
        
        print("✅ Normalized model saved successfully!")
        print(f"📁 Files saved:")
        print(f"   • models/normalized_90_percent_model.joblib")
        print(f"   • models/normalized_90_percent_scaler.joblib")
        print(f"   • models/normalized_90_percent_features.joblib")
        
        # Summary
        print(f"\n🎯 NORMALIZATION SUMMARY:")
        print(f"   • Low Risk: {base_pred_low:.1%} → {norm_pred_low:.1%} (Target: 10-15%)")
        print(f"   • Moderate Risk: {base_pred_mod:.1%} → {norm_pred_mod:.1%} (Target: 50-60%)")
        print(f"   • High Risk: {base_pred_high:.1%} → {norm_pred_high:.1%} (Target: 70-80%)")
        
        print(f"\n💡 Benefits:")
        print(f"   • User-friendly risk ranges (10-15%, 50-60%, 70-80%)")
        print(f"   • Maintains relative risk relationships")
        print(f"   • Easier for users to understand their risk level")
        print(f"   • Perfect for presentations and user interfaces")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating normalized model: {e}")
        return False

if __name__ == "__main__":
    create_normalized_model()
