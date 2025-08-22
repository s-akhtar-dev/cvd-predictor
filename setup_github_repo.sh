#!/bin/bash

echo "🚀 Setting up GitHub Repository for CVD Predictor"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${YELLOW}⚠️  GitHub CLI (gh) is not installed.${NC}"
    echo -e "${BLUE}📥 Installing GitHub CLI...${NC}"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install gh
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
        sudo apt update && sudo apt install gh
    else
        echo -e "${RED}❌ Unsupported OS. Please install GitHub CLI manually:${NC}"
        echo "   https://cli.github.com/"
        exit 1
    fi
fi

echo -e "${GREEN}✅ GitHub CLI detected${NC}"

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo -e "${YELLOW}🔐 Not authenticated with GitHub. Please login:${NC}"
    gh auth login
fi

echo -e "${GREEN}✅ GitHub authentication verified${NC}"

# Get repository name from user
echo -e "${BLUE}📝 Enter the name for your GitHub repository:${NC}"
echo -e "${YELLOW}   (e.g., cvd-predictor, cardiovascular-risk-predictor)${NC}"
read -p "Repository name: " REPO_NAME

# Create the repository
echo -e "${BLUE}🏗️  Creating GitHub repository: ${REPO_NAME}${NC}"
gh repo create "$REPO_NAME" \
    --public \
    --description "Cardiovascular Disease Risk Prediction using Machine Learning" \
    --homepage "https://$REPO_NAME.onrender.com" \
    --source=. \
    --remote=origin \
    --push

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Repository created successfully!${NC}"
else
    echo -e "${RED}❌ Failed to create repository. Please check your GitHub permissions.${NC}"
    exit 1
fi

# Verify Git LFS is properly configured
echo -e "${BLUE}🔍 Verifying Git LFS configuration...${NC}"
git lfs track "*.joblib"
git lfs track "*.csv"
git lfs track "*.png"
git lfs track "*.jpg"
git lfs track "*.jpeg"
git lfs track "*.pkl"

# Add all files (excluding venv)
echo -e "${BLUE}📁 Adding all files to Git (excluding venv)...${NC}"
git add .

# Commit all changes
echo -e "${BLUE}💾 Committing all files...${NC}"
git commit -m "Initial commit: Complete CVD Predictor application with ML models and datasets"

# Push to GitHub with LFS
echo -e "${BLUE}🚀 Pushing to GitHub with Git LFS...${NC}"
git push -u origin main

if [ $? -eq 0 ]; then
    echo -e "${GREEN}🎉 Successfully pushed to GitHub!${NC}"
    echo ""
    echo -e "${BLUE}📋 Repository Information:${NC}"
    echo -e "   Name: ${REPO_NAME}"
    echo -e "   URL: https://github.com/$(gh api user --jq .login)/${REPO_NAME}"
    echo ""
    echo -e "${BLUE}🔗 Next Steps:${NC}"
    echo -e "   1. Visit: https://github.com/$(gh api user --jq .login)/${REPO_NAME}"
    echo -e "   2. Go to Render.com and create a new Web Service"
    echo -e "   3. Connect to this GitHub repository"
    echo -e "   4. Deploy your CVD Predictor app!"
    echo ""
    echo -e "${GREEN}✅ Your repository is ready for Render deployment!${NC}"
else
    echo -e "${RED}❌ Failed to push to GitHub. Please check your connection and try again.${NC}"
    exit 1
fi
