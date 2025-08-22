# 🚀 Deployment Guide for CVD Predictor

## 🌐 **Render Deployment (Recommended)**

### **Step 1: Create Render Account**
1. Go to [render.com](https://render.com)
2. Sign up for a free account
3. Verify your email

### **Step 2: Create New Web Service**
1. Click "New +" → "Web Service"
2. Connect your GitHub repository (or create one first)
3. Choose the `cvd-predictor` repository
4. Set the following:
   - **Name**: `cvd-predictor`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: `Free`

### **Step 3: Deploy**
1. Click "Create Web Service"
2. Render will automatically:
   - Clone your repository
   - Install dependencies
   - Build the application
   - Start the service

### **Step 4: Access Your App**
- Your app will be available at: `https://cvd-predictor.onrender.com`
- Render provides automatic HTTPS
- Free tier includes 750 hours/month

## 🔧 **Alternative: GitHub + Render Integration**

### **Step 1: Create GitHub Repository**
```bash
# Create a new repo on GitHub, then:
git remote add origin https://github.com/yourusername/cvd-predictor.git
git push -u origin main
```

### **Step 2: Connect to Render**
1. In Render, choose "Connect to GitHub"
2. Select your repository
3. Render will auto-deploy on every push

## 📁 **Files Included for Deployment**

- ✅ `app.py` - Main Flask application
- ✅ `requirements.txt` - Python dependencies
- ✅ `render.yaml` - Render configuration
- ✅ `Procfile` - Process configuration
- ✅ `runtime.txt` - Python version
- ✅ `models/*.joblib` - ML model files (via Git LFS)
- ✅ `templates/` - HTML templates
- ✅ `static/` - CSS, JS, images

## 🚨 **Important Notes**

- **Model Files**: Large ML models are tracked with Git LFS
- **Free Tier**: 750 hours/month, auto-sleeps after 15 min inactivity
- **Cold Start**: First request after sleep may take 10-30 seconds
- **File Size**: Render handles large files much better than Railway

## 🔍 **Troubleshooting**

### **Build Failures**
- Check `requirements.txt` for correct dependencies
- Ensure Python 3.12 compatibility
- Check build logs in Render dashboard

### **Runtime Errors**
- Check application logs in Render dashboard
- Verify model file paths in `app.py`
- Ensure all required files are committed to Git

## 📞 **Support**

- **Render Docs**: [docs.render.com](https://docs.render.com)
- **Render Status**: [status.render.com](https://status.render.com)
- **Git LFS**: [git-lfs.github.com](https://git-lfs.github.com)
