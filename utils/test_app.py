#!/usr/bin/env python3
"""
Author: Sarah Akhtar
Description: Simple test script for CVD Predictor
Tests basic functionality without starting the full web server
Course: COMP 193/293 AI in Healthcare
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import flask
        print("✅ Flask imported successfully")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import joblib
        print("✅ Joblib imported successfully")
    except ImportError as e:
        print(f"❌ Joblib import failed: {e}")
        return False
    
    try:
        import plotly.graph_objs as go
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if the trained model can be loaded"""
    print("\nTesting model loading...")
    
    model_path = "best_cardiovascular_model.joblib"
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        return False
    
    try:
        import joblib
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        
        # Test if model has predict method
        if hasattr(model, 'predict'):
            print("✅ Model has predict method")
        else:
            print("⚠️  Model missing predict method")
            
        if hasattr(model, 'predict_proba'):
            print("✅ Model has predict_proba method")
        else:
            print("⚠️  Model missing predict_proba method")
            
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_data_loading():
    """Test if the dataset can be loaded"""
    print("\nTesting dataset loading...")
    
    data_path = "combined_cardiovascular_data.csv"
    if not os.path.exists(data_path):
        print(f"⚠️  Dataset file not found: {data_path}")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(data_path)
        print("✅ Dataset loaded successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def test_flask_app():
    """Test if the Flask app can be created"""
    print("\nTesting Flask app creation...")
    
    try:
        # Temporarily modify sys.path to import app
        sys.path.insert(0, os.getcwd())
        
        # Import the app module
        from app import app
        
        print("✅ Flask app created successfully")
        print(f"   App name: {app.name}")
        print(f"   Debug mode: {app.debug}")
        
        # Test routes
        routes = []
        for rule in app.url_map.iter_rules():
            routes.append(f"{rule.methods} {rule.rule}")
        
        print(f"   Available routes: {len(routes)}")
        for route in routes[:5]:  # Show first 5 routes
            print(f"     {route}")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask app creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 CVD Predictor - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_loading,
        test_data_loading,
        test_flask_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application should work correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

