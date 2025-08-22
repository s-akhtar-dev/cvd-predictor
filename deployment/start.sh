#!/bin/bash

# Simple startup script for CVD Predictor
echo "🚀 Starting CVD Predictor..."
echo "📍 Opening http://localhost:5003 in your browser..."

# Start the Flask app in the background using system Python
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 app.py &

# Wait a moment for the app to start
sleep 3

# Open in default browser
open http://localhost:5003

echo "✅ Application started successfully!"
echo "🛑 To stop the app, press Ctrl+C or close this terminal"
echo ""

# Keep the script running to show the Flask output
wait

