# Deployment Files

This folder contains all files related to deploying the Cardiovascular Risk Prediction application.

## Files

- **`deploy.sh`** - Main deployment script for general deployment
- **`deploy_railway.sh`** - Specific deployment script for Railway platform
- **`start.sh`** - Application startup script
- **`Procfile`** - Process file for Heroku/Railway deployment
- **`runtime.txt`** - Python runtime version specification

## Usage

To deploy the application:

1. **Railway Deployment**: Run `./deploy_railway.sh`
2. **General Deployment**: Run `./deploy.sh`
3. **Local Start**: Run `./start.sh`

## Notes

- Make sure scripts have execute permissions: `chmod +x *.sh`
- Update runtime.txt if Python version changes
- Procfile defines the web process for platform deployment
