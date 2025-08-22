# 🚀 GitHub Repository Setup Guide

## 🌟 **Option 1: Automated Setup (Recommended)**

Run the automated script:
```bash
./setup_github_repo.sh
```

This script will:
- ✅ Install GitHub CLI if needed
- ✅ Authenticate with GitHub
- ✅ Create a new repository
- ✅ Configure Git LFS for large files
- ✅ Push all files to GitHub

## 🔧 **Option 2: Manual Setup**

### **Step 1: Install GitHub CLI**
```bash
# macOS
brew install gh

# Linux
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install gh
```

### **Step 2: Authenticate with GitHub**
```bash
gh auth login
```

### **Step 3: Create Repository**
```bash
gh repo create cvd-predictor \
    --public \
    --description "Cardiovascular Disease Risk Prediction using Machine Learning" \
    --homepage "https://cvd-predictor.onrender.com" \
    --source=. \
    --remote=origin \
    --push
```

### **Step 4: Verify Git LFS Configuration**
```bash
git lfs track "*.joblib"
git lfs track "*.csv"
git lfs track "*.png"
git lfs track "*.jpg"
git lfs track "*.jpeg"
git lfs track "*.pkl"
```

### **Step 5: Add and Commit All Files**
```bash
git add .
git commit -m "Initial commit: Complete CVD Predictor application with ML models and datasets"
```

### **Step 6: Push to GitHub**
```bash
git push -u origin main
```

## 📁 **What Gets Pushed to GitHub**

### **✅ Included Files:**
- **Application**: `app.py`, `requirements.txt`, `Procfile`, `runtime.txt`
- **Models**: All `.joblib` files (handled by Git LFS)
- **Datasets**: All `.csv` files (handled by Git LFS)
- **Templates**: HTML templates for the web interface
- **Static Files**: CSS, JavaScript, images
- **Charts**: Generated visualization files
- **Documentation**: README, deployment guides
- **Configuration**: Render deployment config

### **❌ Excluded Files:**
- **Virtual Environment**: `venv/` folder (not needed for deployment)
- **Cache Files**: `__pycache__/`, `.pyc` files
- **OS Files**: `.DS_Store`, `Thumbs.db`
- **IDE Files**: `.vscode/`, `.idea/`

## 🔍 **Git LFS File Tracking**

The following file types are tracked by Git LFS:
- `*.joblib` - Machine learning models (874MB total)
- `*.csv` - Dataset files
- `*.png`, `*.jpg`, `*.jpeg` - Chart images
- `*.pkl` - Pickle files

## 🚨 **Important Notes**

1. **Large Files**: Git LFS handles files over 50MB automatically
2. **Repository Size**: Total size will be ~1GB due to ML models
3. **Push Time**: First push may take 10-20 minutes due to large files
4. **Bandwidth**: Ensure stable internet connection for upload

## 🌐 **After GitHub Setup**

1. **Visit your repository**: `https://github.com/yourusername/cvd-predictor`
2. **Go to Render.com** and create a new Web Service
3. **Connect to GitHub** and select your repository
4. **Deploy automatically** - Render will build and deploy your app

## 🆘 **Troubleshooting**

### **Git LFS Issues**
```bash
# Reinstall Git LFS
git lfs uninstall
git lfs install

# Re-track files
git lfs track "*.joblib"
git add .gitattributes
git commit -m "Fix Git LFS tracking"
```

### **Large File Push Issues**
```bash
# Check LFS status
git lfs ls-files

# Force push if needed
git push --force-with-lease origin main
```

### **Authentication Issues**
```bash
# Re-authenticate
gh auth logout
gh auth login
```

## 📞 **Support**

- **GitHub CLI**: [cli.github.com](https://cli.github.com/)
- **Git LFS**: [git-lfs.github.com](https://git-lfs.github.com/)
- **GitHub Pages**: [pages.github.com](https://pages.github.com/)
