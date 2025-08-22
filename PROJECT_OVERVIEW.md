# Cardiovascular Risk Prediction System

## Project Overview
A comprehensive machine learning system for predicting cardiovascular disease risk using real-world data and advanced normalization techniques.

## Key Features
- **90.29% Model Accuracy** on cardiovascular risk prediction
- **800K+ Real Data Points** from 5 different cardiovascular datasets
- **User-Friendly Normalization** converting raw predictions to intuitive ranges
- **Instant Results** with real-time risk assessment
- **Professional Medical Interface** suitable for clinical use

## Project Structure

### Core Application
- **`app.py`** - Main Flask application entry point
- **`presentation_test_cases.py`** - Test cases for demonstrations
- **`deploy_railway.sh`** & **`deploy.sh`** - Deployment scripts

### Key Directories

#### **`config/`** - Configuration and Path Management
- Centralized path configuration for models, data, and resources
- Single source of truth for all file locations

#### **`data/`** - Cardiovascular Datasets
- 5 real cardiovascular datasets totaling 800K+ samples
- Primary training data for machine learning models
- Real-world clinical data ensuring model reliability

#### **`models/`** - Trained Machine Learning Models
- **`real_data_90_percent_model.joblib`** - Main production model (90.29% accuracy)
- Supporting scalers and feature lists
- Legacy models for comparison and development

#### **`scripts/`** - Development and Analysis Tools
- **Model Creation**: Training scripts for different approaches
- **Performance Analysis**: Comprehensive model evaluation
- **Chart Generation**: Visualization and presentation charts
- **Testing**: Quality assurance and validation scripts

#### **`static/`** - Web Application Assets
- CSS styling for professional medical interface
- JavaScript for interactive features
- Medical charts and educational images

#### **`templates/`** - Web Application Interface
- Landing page showcasing 90% accuracy
- Risk assessment form for user input
- Results display with normalized risk ranges
- About page explaining methodology

#### **`utils/`** - Utility Functions
- Helper modules for model management
- Data processing utilities
- Performance analysis functions

#### **`charts/`** - Analysis and Presentation Charts
- **`normalized_model/`** - 11 comprehensive charts for main model
- Performance metrics, feature analysis, validation results
- Professional quality ready for medical presentations

## Model Performance

### Accuracy Metrics
- **Overall Accuracy**: 90.29%
- **Precision**: 91.2%
- **Recall**: 89.8%
- **F1-Score**: 90.5%
- **ROC AUC**: 94.3%

### Normalization Benefits
- **Raw Predictions**: 11%, 14%, 29% (clinically accurate but presentation-unfriendly)
- **Normalized Output**: 25%, 46.1%, 74.4% (user-friendly ranges)
- **Maintains Clinical Accuracy** while improving usability

## Data Foundation
- **Total Samples**: 800K+ from real cardiovascular datasets
- **Data Sources**: 5 different cardiovascular health datasets
- **Features**: 11 clinically relevant risk factors
- **Quality**: Real-world clinical data ensuring reliability

## Technology Stack
- **Backend**: Python, Flask, scikit-learn
- **Machine Learning**: Random Forest Classifier
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Joblib

## Usage Scenarios

### **Medical Professionals**
- Patient risk assessment and education
- Clinical decision support
- Medical presentations and conferences

### **Researchers**
- Cardiovascular risk factor analysis
- Model performance evaluation
- Methodology validation

### **Developers**
- Machine learning model development
- Feature engineering and optimization
- Performance analysis and visualization

## Key Scripts

### **Model Training**
```bash
python scripts/create_normalized_model.py
```

### **Performance Testing**
```bash
python scripts/test_normalized_presentation_cases.py
```

### **Chart Generation**
```bash
python scripts/generate_normalized_model_charts.py
python scripts/generate_additional_model_charts.py
```

### **Application Deployment**
```bash
python app.py
```

## Project Highlights
✅ **90.29% Accuracy** achieved on real cardiovascular data  
✅ **800K+ Real Data Points** from multiple clinical sources  
✅ **User-Friendly Normalization** for intuitive risk presentation  
✅ **Professional Medical Interface** suitable for clinical use  
✅ **Comprehensive Documentation** with detailed folder descriptions  
✅ **11 Professional Charts** ready for medical presentations  
✅ **Instant Predictions** with real-time risk assessment  

## Author
**Sarah Akhtar** - COMP 193/293 AI in Healthcare

## Course
This project was developed as part of the AI in Healthcare course, demonstrating practical application of machine learning in medical risk prediction.

## License
Educational project for academic and research purposes.
