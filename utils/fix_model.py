#!/usr/bin/env python3
"""
Author: Sarah Akhtar
Description: Fix the cardiovascular risk prediction model
Creates a properly calibrated model with realistic risk predictions
Course: COMP 193/293 AI in Healthcare
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the cardiovascular data"""
    print("Loading and preparing data...")
    
    try:
        # Load the combined dataset
        df = pd.read_csv('combined_cardiovascular_data.csv')
        print(f"Dataset loaded: {df.shape}")
        
        # Check the target variable distribution
        if 'heart_disease_1' in df.columns:
            target_col = 'heart_disease_1'
        elif 'heart_disease' in df.columns:
            target_col = 'heart_disease'
        else:
            # Look for any column that might be the target
            target_candidates = [col for col in df.columns if 'heart' in col.lower() or 'disease' in col.lower()]
            if target_candidates:
                target_col = target_candidates[0]
            else:
                raise ValueError("Could not find target variable")
        
        print(f"Target column: {target_col}")
        print(f"Target distribution:\n{df[target_col].value_counts()}")
        
        # Handle missing values
        df = df.fillna(df.median())
        
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Remove any non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y, target_col
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def create_balanced_features(X, y):
    """Create balanced features and handle class imbalance"""
    print("Creating balanced features...")
    
    # Check class balance
    class_counts = y.value_counts()
    print(f"Original class distribution: {class_counts}")
    
    # If severely imbalanced, we'll use techniques to balance it
    if class_counts.min() / class_counts.max() < 0.1:
        print("Severe class imbalance detected. Using balanced sampling...")
        
        # Use RandomForest with class_weight='balanced'
        from sklearn.ensemble import RandomForestClassifier
        from imblearn.over_sampling import SMOTE
        
        # Apply SMOTE to balance the classes
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print(f"After SMOTE - Class distribution: {pd.Series(y_balanced).value_counts()}")
        return X_balanced, y_balanced
    else:
        return X, y

def train_improved_model(X, y):
    """Train an improved, properly calibrated model"""
    print("Training improved model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
    }
    
    # Train and evaluate each model
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        # Check prediction distribution
        risk_levels = pd.cut(y_proba, bins=[0, 0.33, 0.67, 1.0], labels=['Low', 'Moderate', 'High'])
        print(f"Risk distribution: {risk_levels.value_counts()}")
        
        if auc > best_score:
            best_score = auc
            best_model = model
    
    # Calibrate the best model for better probability estimates
    print(f"\nCalibrating best model (AUC: {best_score:.4f})...")
    
    # Use CalibratedClassifierCV for better probability calibration
    calibrated_model = CalibratedClassifierCV(
        best_model, 
        cv=5, 
        method='isotonic'
    )
    
    calibrated_model.fit(X_train_scaled, y_train)
    
    # Final evaluation
    y_pred_cal = calibrated_model.predict(X_test_scaled)
    y_proba_cal = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    final_accuracy = (y_pred_cal == y_test).mean()
    final_auc = roc_auc_score(y_test, y_proba_cal)
    
    print(f"\nFinal Calibrated Model:")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"AUC: {final_auc:.4f}")
    
    # Check final risk distribution
    risk_levels_final = pd.cut(y_proba_cal, bins=[0, 0.33, 0.67, 1.0], labels=['Low', 'Moderate', 'High'])
    print(f"Final Risk Distribution:")
    print(risk_levels_final.value_counts())
    
    # Save the improved model and scaler
    joblib.dump(calibrated_model, 'improved_cardiovascular_model.joblib')
    joblib.dump(scaler, 'improved_scaler.joblib')
    
    print("\n✅ Improved model saved as 'improved_cardiovascular_model.joblib'")
    print("✅ Scaler saved as 'improved_scaler.joblib'")
    
    return calibrated_model, scaler, X_test_scaled, y_test

def create_sample_predictions(model, scaler, X_test, y_test):
    """Create sample predictions to verify the model works properly"""
    print("\nCreating sample predictions...")
    
    # Get some sample predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Create a sample user profile (low risk)
    sample_low_risk = np.array([[
        25,    # age_years
        0,     # sex (female)
        165,   # height_cm
        60,    # weight_kg
        22.0,  # BMI
        0,     # smoking
        1,     # physical_activity
        180,   # cholesterol
        0,     # skin_cancer
        1,     # general_health (good)
        1,     # age_category
        0,     # alcohol_consumption
        0,     # diabetes
        0      # heart_disease
    ]])
    
    # Pad with zeros to match the expected feature count
    if sample_low_risk.shape[1] < X_test.shape[1]:
        padding = np.zeros((1, X_test.shape[1] - sample_low_risk.shape[1]))
        sample_low_risk = np.hstack([sample_low_risk, padding])
    
    # Scale the sample
    sample_scaled = scaler.transform(sample_low_risk)
    
    # Make prediction
    low_risk_prob = model.predict_proba(sample_scaled)[0, 1]
    low_risk_category = 'Low' if low_risk_prob <= 0.33 else 'Moderate' if low_risk_prob <= 0.67 else 'High'
    
    print(f"Sample Low-Risk Profile (25yo, healthy):")
    print(f"  Risk Probability: {low_risk_prob:.3f} ({low_risk_prob*100:.1f}%)")
    print(f"  Risk Category: {low_risk_category}")
    
    # Create a sample user profile (high risk)
    sample_high_risk = np.array([[
        65,    # age_years
        1,     # sex (male)
        175,   # height_cm
        95,    # weight_kg
        31.0,  # BMI
        1,     # smoking
        0,     # physical_activity
        280,   # cholesterol
        0,     # skin_cancer
        4,     # general_health (poor)
        9,     # age_category
        15,    # alcohol_consumption
        1,     # diabetes
        1      # heart_disease
    ]])
    
    # Pad with zeros to match the expected feature count
    if sample_high_risk.shape[1] < X_test.shape[1]:
        padding = np.zeros((1, X_test.shape[1] - sample_high_risk.shape[1]))
        sample_high_risk = np.hstack([sample_high_risk, padding])
    
    # Scale the sample
    sample_scaled = scaler.transform(sample_high_risk)
    
    # Make prediction
    high_risk_prob = model.predict_proba(sample_scaled)[0, 1]
    high_risk_category = 'Low' if high_risk_prob <= 0.33 else 'Moderate' if high_risk_prob <= 0.67 else 'High'
    
    print(f"\nSample High-Risk Profile (65yo, unhealthy):")
    print(f"  Risk Probability: {high_risk_prob:.3f} ({high_risk_prob*100:.1f}%)")
    print(f"  Risk Category: {high_risk_category}")
    
    # Show overall distribution
    print(f"\nOverall Test Set Risk Distribution:")
    risk_levels = pd.cut(y_proba, bins=[0, 0.33, 0.67, 1.0], labels=['Low', 'Moderate', 'High'])
    print(risk_levels.value_counts())

def main():
    """Main function to fix the model"""
    print("🔧 Fixing Cardiovascular Risk Prediction Model")
    print("=" * 50)
    
    # Load and prepare data
    X, y, target_col = load_and_prepare_data()
    if X is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Create balanced features
    X_balanced, y_balanced = create_balanced_features(X, y)
    
    # Train improved model
    model, scaler, X_test, y_test = train_improved_model(X_balanced, y_balanced)
    
    # Create sample predictions
    create_sample_predictions(model, scaler, X_test, y_test)
    
    print("\n🎉 Model fixed successfully!")
    print("\nNext steps:")
    print("1. Update app.py to use 'improved_cardiovascular_model.joblib'")
    print("2. Update app.py to use 'improved_scaler.joblib'")
    print("3. Restart the Flask application")

if __name__ == "__main__":
    main()

