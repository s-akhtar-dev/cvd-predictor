# CVD Predictor: AI-Powered Cardiovascular Risk Assessment System

**Author:** Sarah Akhtar  
**Department:** Computer Science, University of the Pacific  
**Location:** Stockton, CA, USA  
**Email:** s.akhtar@pacific.edu  

---

## Project Overview

This project develops an intelligent system for predicting cardiovascular disease risk using machine learning techniques. The system analyzes multiple health factors to provide accurate risk assessments, helping healthcare professionals make informed decisions about patient care.

## Research Context

Cardiovascular diseases remain a significant global health challenge, affecting millions of people worldwide. Traditional methods for assessing cardiovascular risk often require extensive testing and may not capture the complex interactions between various risk factors. This project addresses these limitations by creating a data-driven approach that combines multiple information sources to deliver comprehensive risk evaluations.

## Technical Approach

The system integrates several large-scale healthcare datasets to create a robust prediction model. By combining clinical records, demographic information, and behavioral data, the model can identify patterns that might be missed by conventional assessment methods.

### Data Integration Strategy

The project combines five different cardiovascular datasets, creating a comprehensive training set of over 800,000 patient samples. This diverse data foundation helps ensure the model performs well across different population groups and reduces bias that can occur when using limited datasets.

### Machine Learning Implementation

A Random Forest classifier serves as the core prediction engine, chosen for its ability to handle complex, non-linear relationships in medical data. The model processes eleven key health indicators including age, body mass index, blood pressure readings, cholesterol levels, and lifestyle factors.

### Model Performance

The system achieves strong predictive accuracy, with cross-validation results showing consistent performance across different data subsets. The model demonstrates reliable discrimination between low, moderate, and high-risk patients, making it suitable for clinical decision support.

## System Features

### Real-Time Assessment
Healthcare providers can input patient information through an intuitive web interface and receive immediate risk assessments. The system processes multiple risk factors simultaneously, providing comprehensive evaluations in seconds.

### Clinical Interpretability
Risk predictions are presented in familiar clinical terms, with clear categorization into low, moderate, and high-risk groups. This approach helps bridge the gap between complex machine learning outputs and practical clinical decision-making.

### Interactive Visualization
The system includes dynamic charts and graphs that help users understand risk factors and their relative importance. These visualizations support patient education and clinical discussions.

## Technical Architecture

### Web Application Framework
The system is deployed as a Flask-based web application, ensuring accessibility across different devices and platforms. The architecture supports real-time predictions while maintaining security and privacy standards.

### Data Processing Pipeline
Input data undergoes rigorous validation and preprocessing before reaching the prediction model. This includes range checking, data type validation, and feature scaling to ensure optimal model performance.

### Model Deployment
Trained models are serialized and loaded efficiently, enabling fast response times for clinical use. The system maintains model performance while providing the scalability needed for healthcare environments.

## Clinical Applications

### Primary Care Settings
The system can support routine health assessments, helping identify patients who may benefit from additional monitoring or preventive interventions.

### Risk Stratification
Healthcare providers can use the system to categorize patients by risk level, supporting resource allocation and treatment planning decisions.

### Patient Education
The interactive interface helps patients understand their cardiovascular risk factors and the importance of lifestyle modifications.

## Research Contributions

This project advances the field of cardiovascular risk prediction in several ways:

- **Data Scale:** Integration of multiple large datasets improves model generalizability
- **Clinical Integration:** Production-ready deployment enables immediate clinical use
- **Interpretability:** User-friendly risk presentation maintains clinical accuracy
- **Validation:** Comprehensive testing ensures reliability across diverse populations

## Future Development

The system provides a foundation for continued research in healthcare AI applications. Potential enhancements include:

- Integration with electronic health record systems
- Real-time learning from new clinical data
- Expansion to additional cardiovascular conditions
- Mobile application development for patient use

## Technical Requirements

### Dependencies
- Python 3.8+
- Scikit-learn 1.0+
- Flask 2.0+
- Pandas and NumPy
- Additional visualization and data processing libraries

### System Requirements
- Modern web browser for user interface
- Python environment for backend processing
- Sufficient memory for model loading and prediction

## Getting Started

### Local Development
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Production Deployment
```bash
chmod +x deploy_railway.sh
./deploy_railway.sh
```

## Project Structure

```
project/
├── app.py                          # Main Flask application
├── config/                         # Configuration management
├── data/                           # Cardiovascular datasets
├── models/                         # Trained machine learning models
├── scripts/                        # Model training and analysis scripts
├── static/                         # Web application assets
├── templates/                      # HTML templates for web interface
├── utils/                          # Utility functions and testing
└── charts/                         # Performance analysis and charts
```

## Performance Summary

The system demonstrates strong predictive capabilities across multiple evaluation metrics:

| Metric | Performance |
|--------|-------------|
| Accuracy | High reliability for clinical use |
| Precision | Low false positive rate |
| Recall | Good detection of high-risk patients |
| F1-Score | Balanced precision and recall |
| ROC AUC | Excellent discrimination ability |

## Contact and Support

**Project Repository:** https://github.com/s-akhtar-dev/cvd-predictor  
**Live Application:** https://cvd-predictor-production-f8be.up.railway.app/  

For questions or collaboration opportunities, please contact the development team.

## License

This project is developed for educational and research purposes. The system demonstrates the potential of machine learning in healthcare applications while maintaining appropriate privacy and security standards.

---

**Acknowledgments:** Special thanks to Dr. Gao for guidance and support throughout this research project.
