# CVD Predictor

A comprehensive web application for cardiovascular disease risk prediction using machine learning. This application provides users with personalized risk assessments, educational resources, and preventive recommendations based on their health data.

## Features

### 🏠 **Homepage**
- Clean, modern design with project description
- Clear explanation of purpose and benefits
- Call-to-action buttons for risk assessment
- Responsive design for all devices

### 📋 **Risk Assessment Form**
- Comprehensive health data collection
- Input validation and error handling
- Real-time BMI calculation
- User-friendly interface with clear instructions

### 📊 **Results & Analysis**
- Personalized risk probability (0-100%)
- Risk categorization (Low, Moderate, High)
- Interactive charts and visualizations
- Key contributing factors identification
- Personalized recommendations

### 🔍 **Insights & Analytics**
- Feature importance analysis
- Model performance metrics
- Causal machine learning insights
- Technical methodology details

### 📚 **Educational Resources**
- Prevention strategies
- Lifestyle recommendations
- Medical follow-up guidance
- Print-friendly results

## Technology Stack

- **Backend**: Python Flask
- **Machine Learning**: Scikit-learn, XGBoost
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Visualization**: Plotly.js
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Joblib

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository
```bash
git clone <repository-url>
cd project
```

### 2. Install Dependencies
```bash
pip install -r project/requirements.txt
```

### 3. Prepare Data Files
Ensure the following files are in your project directory:
- `best_cardiovascular_model.joblib` - Trained machine learning model
- `combined_cardiovascular_data.csv` - Combined dataset for column information

### 4. Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5001`

## Project Structure

```
project/
├── app.py                          # Main Flask application
├── project/
│   ├── requirements.txt           # Python dependencies
│   └── README.md                 # Project config docs
├── README.md                      # Project documentation
├── best_cardiovascular_model.joblib # Trained ML model
├── combined_cardiovascular_data.csv # Dataset for column info
├── templates/                     # HTML templates
│   ├── base.html                 # Base template
│   ├── index.html                # Homepage
│   ├── assessment.html           # Risk assessment form
│   ├── results.html              # Results page
│   ├── insights.html             # Insights page
│   └── about.html                # About page
├── static/                       # Static assets
│   ├── css/
│   │   └── style.css            # Custom styles
│   └── js/
│       └── main.js              # Main JavaScript
└── data/                        # Data files (if any)
```

## Usage

### 1. Access the Application
Open your web browser and navigate to `http://localhost:5000`

### 2. Take Risk Assessment
- Click "Start Assessment" on the homepage
- Fill out the comprehensive health form
- Submit to receive your risk prediction

### 3. View Results
- See your personalized risk level
- Explore interactive charts
- Review contributing factors
- Get personalized recommendations

### 4. Explore Insights
- Learn about feature importance
- Understand model performance
- View causal analysis results

## Model Information

### Training Data
The model is trained on a comprehensive dataset combining:
- Cardiovascular Disease Death Rates (US county-level, 2010-2020)
- CVD Cleaned Dataset
- Heart Disease 2020 Dataset
- Cardiovascular Risk Dataset

### Model Architecture
- **Ensemble Model**: Combines Logistic Regression, Random Forest, and XGBoost
- **Feature Engineering**: BMI categories, age groups, interaction terms
- **Performance**: 95.2% accuracy, 0.94 AUC-ROC

### Features Used
- Demographics (age, gender)
- Physical measurements (height, weight, BMI)
- Health status (smoking, physical activity, cholesterol)
- Medical history (diabetes, heart disease)
- Lifestyle factors (alcohol consumption, general health)

## API Endpoints

### POST `/predict`
Submit health data for risk prediction.

**Request Body:**
```json
{
    "age_years": 45,
    "sex": 1,
    "height_cm": 175,
    "weight_kg": 80,
    "bmi": 26.1,
    "smoking": 0,
    "physical_activity": 1,
    "cholesterol": 200,
    "general_health": 2,
    "alcohol_consumption": 5,
    "skin_cancer": 0,
    "diabetes": "No",
    "heart_disease": "No"
}
```

**Response:**
```json
{
    "risk_probability": 0.35,
    "risk_category": "Moderate",
    "risk_chart": "..."
}
```

## Configuration

### Environment Variables
- `FLASK_ENV`: Set to `production` for production deployment
- `FLASK_DEBUG`: Set to `False` for production

### Model Configuration
- Update model file path in `app.py` if needed
- Adjust risk thresholds in the prediction function
- Modify feature preprocessing as required

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY project/requirements.txt ./requirements.txt
RUN pip install -r project/requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## Security & Privacy

### Data Protection
- No user data is stored permanently
- All processing is done in real-time
- Secure data transmission
- Medical disclaimer included

### Medical Disclaimer
This application is for educational and informational purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Testing

### Manual Testing
- Test all form inputs and validation
- Verify risk predictions
- Check responsive design on different devices
- Test print functionality

### Automated Testing
```bash
# Install test dependencies
pip install pytest pytest-flask

# Run tests
pytest
```

## Performance

### Optimization Features
- Lazy loading of charts
- Efficient data preprocessing
- Cached model loading
- Responsive image handling

### Monitoring
- Performance metrics logging
- Error tracking and reporting
- User interaction analytics

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure `best_cardiovascular_model.joblib` exists
   - Check file permissions
   - Verify model compatibility

2. **Data Processing Error**
   - Check `combined_cardiovascular_data.csv` format
   - Verify column names match expected format
   - Ensure sufficient memory for large datasets

3. **Chart Display Issues**
   - Check Plotly.js is loaded
   - Verify browser console for errors
   - Test with different browsers

### Debug Mode
Enable debug mode for detailed error information:
```python
app.run(debug=True)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For technical support or questions:
- Email: support@cvd.com
- Documentation: [Project Wiki]
- Issues: [GitHub Issues]

## Acknowledgments

- Machine learning community for open-source tools
- Healthcare professionals for domain expertise
- Open data providers for cardiovascular datasets
- Contributors and beta testers

## Version History

- **v1.0.0** - Initial release with core functionality
- **v1.1.0** - Enhanced visualizations and insights
- **v1.2.0** - Improved accessibility and performance

---

**Built with ❤️ for better cardiovascular health outcomes**
