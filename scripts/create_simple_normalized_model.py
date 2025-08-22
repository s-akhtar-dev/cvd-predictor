#!/usr/bin/env python3
"""
Author: Sarah Akhtar
Description: Create a simple normalized model using function wrapper
Course: COMP 193/293 AI in Healthcare
"""

import joblib
import numpy as np
import pickle

def create_simple_normalized_model():
    """Create a simple normalized model wrapper"""
    print("🔧 Creating Simple Normalized Cardiovascular Risk Model")
    print("=" * 60)
    
    try:
        # Load the base model and scaler
        print("📥 Loading base model and scaler...")
        base_model = joblib.load('models/real_data_90_percent_model.joblib')
        scaler = joblib.load('models/real_data_90_percent_scaler.joblib')
        feature_names = joblib.load('models/real_data_90_percent_features.joblib')
        
        print("✅ Base model loaded successfully")
        
        # Create a simple normalization function
        def normalize_prediction(raw_prediction):
            """Normalize prediction to user-friendly ranges"""
            # Current model range: 0.10 to 0.38
            min_pred = 0.10
            max_pred = 0.38
            range_pred = 0.28
            
            # Normalize to 0-1 range
            normalized = (raw_prediction - min_pred) / range_pred
            normalized = np.clip(normalized, 0, 1)
            
            # Map to target ranges
            if normalized <= 0.33:  # Low risk
                # Map 0-0.33 to 15-25%
                mapped = 0.15 + (normalized / 0.33) * 0.10
            elif normalized <= 0.67:  # Moderate risk
                # Map 0.33-0.67 to 45-55%
                mapped = 0.45 + ((normalized - 0.33) / 0.34) * 0.10
            else:  # High risk
                # Map 0.67-1.0 to 70-80%
                mapped = 0.70 + ((normalized - 0.67) / 0.33) * 0.10
            
            return np.clip(mapped, 0.15, 0.80)
        
        # Create a wrapper function that can be pickled
        def normalized_predict_proba(X):
            """Get normalized probability predictions"""
            # Get base predictions
            base_probs = base_model.predict_proba(X)
            # Normalize only the positive class (index 1)
            normalized_probs = base_probs.copy()
            normalized_probs[:, 1] = np.array([normalize_prediction(p) for p in base_probs[:, 1]])
            # Adjust negative class to maintain sum = 1
            normalized_probs[:, 0] = 1 - normalized_probs[:, 1]
            return normalized_probs
        
        def normalized_predict(X):
            """Get binary predictions based on normalized probabilities"""
            probs = normalized_predict_proba(X)
            return (probs[:, 1] > 0.5).astype(int)
        
        # Create a simple model object with the normalized functions
        class SimpleNormalizedModel:
            def __init__(self, base_model, scaler, feature_names, normalize_func):
                self.base_model = base_model
                self.scaler = scaler
                self.feature_names = feature_names
                self.normalize_func = normalize_func
            
            def predict_proba(self, X):
                return normalized_predict_proba(X)
            
            def predict(self, X):
                return normalized_predict(X)
        
        # Create the normalized model
        normalized_model = SimpleNormalizedModel(base_model, scaler, feature_names, normalize_prediction)
        
        # Test the normalized model
        print("🧪 Testing normalized model...")
        
        # Test case 1: Low risk (should map to 15-25%)
        test_low = np.array([25, 0, 22, 0, 1, 0, 0, 180, 120, 75, 90])
        test_low_scaled = scaler.transform(test_low.reshape(1, -1))
        
        # Get base prediction
        base_pred_low = base_model.predict_proba(test_low_scaled)[0, 1]
        norm_pred_low = normalized_model.predict_proba(test_low_scaled)[0, 1]
        
        print(f"📊 Low Risk Test Case:")
        print(f"   Base prediction: {base_pred_low:.1%}")
        print(f"   Normalized prediction: {norm_pred_low:.1%}")
        
        # Test case 2: Moderate risk (should map to 45-55%)
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
        joblib.dump(normalized_model, 'models/simple_normalized_90_percent_model.joblib')
        joblib.dump(scaler, 'models/simple_normalized_90_percent_scaler.joblib')
        joblib.dump(feature_names, 'models/simple_normalized_90_percent_features.joblib')
        
        print("✅ Simple normalized model saved successfully!")
        print(f"📁 Files saved:")
        print(f"   • models/simple_normalized_90_percent_model.joblib")
        print(f"   • models/simple_normalized_90_percent_scaler.joblib")
        print(f"   • models/simple_normalized_90_percent_features.joblib")
        
        # Summary
        print(f"\n🎯 NORMALIZATION SUMMARY:")
        print(f"   • Low Risk: {base_pred_low:.1%} → {norm_pred_low:.1%} (Target: 15-25%)")
        print(f"   • Moderate Risk: {base_pred_mod:.1%} → {norm_pred_mod:.1%} (Target: 45-55%)")
        print(f"   • High Risk: {base_pred_high:.1%} → {norm_pred_high:.1%} (Target: 70-80%)")
        
        print(f"\n💡 Benefits:")
        print(f"   • User-friendly risk ranges (15-25%, 45-55%, 70-80%)")
        print(f"   • Maintains relative risk relationships")
        print(f"   • Easier for users to understand their risk level")
        print(f"   • Perfect for presentations and user interfaces")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating simple normalized model: {e}")
        return False

if __name__ == "__main__":
    create_simple_normalized_model()
