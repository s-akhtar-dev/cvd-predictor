#!/usr/bin/env python3
"""
Author: Sarah Akhtar
Description: Main entry point for the Cardiovascular Risk Prediction Flask application
Course: COMP 193/293 AI in Healthcare
"""

from app import app

if __name__ == '__main__':
    print("🚀 Starting Cardiovascular Risk Prediction Application...")
    print("📍 Access your website at: http://localhost:5003")
    print("🔄 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5003)
