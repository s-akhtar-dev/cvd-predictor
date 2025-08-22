#!/bin/bash

echo "🚀 Deploying CVD Predictor to Render..."
echo "============================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git repository not found. Initializing..."
    git init
    git add .
    git commit -m "Initial commit for Render deployment"
fi

echo "✅ Git repository ready"

# Check if remote exists
if ! git remote get-url origin &> /dev/null; then
    echo "🔗 Please add your Render Git remote:"
    echo "   git remote add origin <YOUR_RENDER_GIT_URL>"
    echo ""
    echo "📝 You can get this URL from your Render dashboard after creating a new web service"
    echo "   and connecting it to this Git repository."
    echo ""
    echo "💡 After adding the remote, run: git push -u origin main"
    exit 1
fi

echo "✅ Remote origin found"

# Push to Render
echo "🚀 Pushing to Render..."
git push origin main

echo ""
echo "✅ Deployment initiated!"
echo "🌐 Check your Render dashboard for deployment status"
echo "🔗 Your app will be available at: https://cvd-predictor.onrender.com"
echo ""
echo "💡 Render will automatically redeploy when you push changes to main branch"
