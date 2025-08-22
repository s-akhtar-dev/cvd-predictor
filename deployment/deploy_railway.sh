#!/bin/bash

echo "🚀 Deploying CVD Predictor to Railway..."
echo "============================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI is not installed. Installing now..."
    npm install -g @railway/cli
fi

echo "✅ Railway CLI detected"

# Login to Railway (if not already logged in)
echo "🔐 Logging into Railway..."
railway login

# Initialize Railway project (if not already initialized)
if [ ! -f ".railway" ]; then
    echo "📦 Initializing Railway project..."
    railway init
fi

# Deploy to Railway
echo "🚀 Deploying to Railway..."
railway up

echo ""
echo "✅ Deployment complete!"
echo "🌐 Your app should now be live on Railway!"
echo "🔗 Check your Railway dashboard for the live URL"
echo ""
echo "💡 To view logs: railway logs"
echo "💡 To open the app: railway open"
