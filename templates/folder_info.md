# Templates Folder

## Purpose
This folder contains HTML templates for the Flask web application, providing the user interface for the cardiovascular risk assessment system.

## Contents

### Core Templates

#### `base.html`
- **Purpose**: Base template that other templates extend
- **Function**: Defines common layout, navigation, and structure
- **Features**: Header, footer, navigation menu, common styling
- **Usage**: Foundation template for consistent page structure

#### `index.html`
- **Purpose**: Landing page and main entry point
- **Function**: Introduces the application and displays key statistics
- **Features**: 90% accuracy showcase, 800K+ data points, instant results
- **Usage**: First page users see when visiting the application

#### `assessment.html`
- **Purpose**: Cardiovascular risk assessment form
- **Function**: Collects user health data for risk prediction
- **Features**: Input fields for age, BMI, blood pressure, cholesterol, etc.
- **Usage**: Main form where users enter their health information

#### `about.html`
- **Purpose**: Information about the application and methodology
- **Function**: Explains how the system works and its scientific basis
- **Features**: Model description, data sources, methodology explanation
- **Usage**: Educational content for users and medical professionals

#### `results.html`
- **Purpose**: Displays risk assessment results
- **Function**: Shows predicted cardiovascular risk with explanations
- **Features**: Risk percentage, risk level, recommendations, visual indicators
- **Usage**: Results page after form submission

## Template Features

### **Responsive Design**
- Mobile-friendly layouts
- Adaptive navigation
- Flexible content areas

### **Medical Interface**
- Professional medical appearance
- Clear information hierarchy
- Accessible design elements

### **Interactive Elements**
- Dynamic form validation
- Real-time feedback
- Smooth user experience

## Template Structure
```
base.html (base template)
├── index.html (landing page)
├── assessment.html (input form)
├── results.html (results display)
└── about.html (information page)
```

## Usage
- **Flask Integration**: Templates are rendered by Flask routes
- **Dynamic Content**: Python variables can be passed to templates
- **Consistent Design**: All pages share common styling and layout
- **User Experience**: Provides intuitive navigation and clear information

## Author
Sarah Akhtar - COMP 193/293 AI in Healthcare
