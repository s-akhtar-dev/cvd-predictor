# Utils Folder

## Purpose
This folder contains utility functions and helper modules that support the main cardiovascular risk prediction system.

## Contents

### Core Utility Modules

#### `__init__.py`
- **Purpose**: Python package initialization file
- **Function**: Makes the utils directory a Python package
- **Usage**: Required for importing utility modules

#### `create_realistic_model.py`
- **Purpose**: Creates realistic cardiovascular risk prediction models
- **Function**: Alternative model training approach with different feature engineering
- **Features**: Realistic data generation, feature selection, model training
- **Usage**: Development and comparison with main models

#### `fix_model.py`
- **Purpose**: Utility for fixing and updating existing models
- **Function**: Model repair, feature alignment, compatibility fixes
- **Usage**: Maintenance and troubleshooting of trained models

### Utility Functions

#### Model Management
- Model loading and saving utilities
- Feature alignment functions
- Model validation helpers

#### Data Processing
- Feature engineering utilities
- Data normalization functions
- Input validation helpers

#### Performance Analysis
- Model evaluation utilities
- Statistical analysis functions
- Performance metric calculations

## Usage Patterns
```python
from utils.create_realistic_model import create_realistic_model
from utils.fix_model import fix_model_features

# Create alternative model
model = create_realistic_model()

# Fix existing model
fixed_model = fix_model_features(model_path)
```

## Benefits
- **Modularity**: Separates utility functions from main application logic
- **Reusability**: Common functions can be used across different scripts
- **Maintainability**: Centralized location for utility functions
- **Testing**: Easier to test individual utility functions

## Integration
- **Scripts**: Used by various scripts in the scripts/ folder
- **Application**: Supports the main Flask application
- **Development**: Facilitates model development and testing

## Author
Sarah Akhtar - COMP 193/293 AI in Healthcare
