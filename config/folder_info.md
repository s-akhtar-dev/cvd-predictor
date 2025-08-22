# Config Folder

## Purpose
This folder contains configuration files and settings for the cardiovascular risk prediction application.

## Contents

### `__init__.py`
- **Purpose**: Python package initialization file
- **Function**: Makes the config directory a Python package
- **Usage**: Required for importing config modules

### `paths.py`
- **Purpose**: Centralized path configuration
- **Function**: Defines file paths for models, data, and other resources
- **Usage**: Imported by other modules to access consistent file paths
- **Benefits**: Single source of truth for all file locations, easy to maintain

## Usage
```python
from config.paths import MODEL_PATH, DATA_PATH
# Use these paths throughout the application
```

## Author
Sarah Akhtar - COMP 193/293 AI in Healthcare
