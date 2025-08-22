# Data Folder

## Purpose
This folder contains all the cardiovascular datasets used to train and validate the machine learning models.

## Contents

### `cardio_train.csv`
- **Purpose**: Primary cardiovascular training dataset
- **Size**: ~70,000 samples
- **Features**: Age, gender, height, weight, BMI, blood pressure, cholesterol, glucose, smoking, alcohol, physical activity
- **Target**: Cardiovascular disease presence (binary)
- **Source**: Public cardiovascular dataset

### `Cardiovascular_Disease_Death_Rates__Trends__and_Excess_Death_Rates_Among_US_Adults__35___by_County_and_Age_Group___2010-2020.csv`
- **Purpose**: US county-level cardiovascular death rate data
- **Size**: ~319,000 samples
- **Features**: Geographic, demographic, and temporal cardiovascular mortality data
- **Usage**: Supplementary data for understanding regional patterns

### `Cardiovascular_Disease_Risk_Dataset.csv`
- **Purpose**: Comprehensive cardiovascular risk assessment dataset
- **Size**: ~308,000 samples
- **Features**: Clinical risk factors, lifestyle indicators, medical history
- **Target**: Risk stratification for cardiovascular events

### Additional CSV files
- **Purpose**: Various cardiovascular datasets for comprehensive model training
- **Total Combined**: 800K+ data points from 5 different sources
- **Benefits**: Robust training data foundation, reduces overfitting, improves generalization

## Usage
- **Training**: Models are trained on these datasets
- **Validation**: Used for cross-validation and performance assessment
- **Feature Engineering**: Basis for creating interaction and polynomial features
- **Synthetic Data**: Combined with generated data for enhanced training

## Data Quality
- **Real-world data**: All datasets contain actual cardiovascular health information
- **Diverse sources**: Multiple datasets ensure robust model performance
- **Clinical relevance**: Features align with known cardiovascular risk factors

## Author
Sarah Akhtar - COMP 193/293 AI in Healthcare
