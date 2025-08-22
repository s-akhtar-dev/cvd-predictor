#!/bin/bash

# CVD Predictor Deployment Script
# This script sets up and runs the cardiovascular risk prediction web application

echo "🚀 CVD Predictor - Deployment Script"
echo "============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python version $PYTHON_VERSION is too old. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3."
    exit 1
fi

echo "✅ pip3 detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
if [ -f "../project/requirements.txt" ]; then
    pip install -r ../project/requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed"
else
    echo "❌ requirements.txt not found in ../project/"
    exit 1
fi

# Check if required files exist
echo "🔍 Checking required files..."

if [ ! -f "best_cardiovascular_model.joblib" ]; then
    echo "⚠️  Warning: best_cardiovascular_model.joblib not found"
    echo "   The application may not function properly without the trained model"
fi

if [ ! -f "combined_cardiovascular_data.csv" ]; then
    echo "⚠️  Warning: combined_cardiovascular_data.csv not found"
    echo "   The application may not function properly without the dataset"
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found. Please ensure you're in the correct directory."
    exit 1
fi

echo "✅ All required files found"

# Set environment variables
export FLASK_ENV=development
export FLASK_DEBUG=1

echo ""
echo "🎯 Starting CVD Predictor..."
echo "📍 Application will be available at: http://localhost:5001"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

# Start the application
python app.py

# Deactivate virtual environment when done
deactivate
echo ""
echo "👋 Application stopped. Virtual environment deactivated."
