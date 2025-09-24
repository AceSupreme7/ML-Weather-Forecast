# ML-Weather-Forecast
A machine learning-powered weather prediction system that forecasts foremost temperature, then precipitation, and basic weather conditions using historical data analysis and feature engineering.

## System Architecture

### Core Components

| Module | Purpose | Key Features |
|--------|---------|--------------|
| **main.py** | Application entry point | Orchestrates the prediction pipeline |
| **config.py** | Configuration management | API settings, location coordinates, time ranges |
| **weather_predictor.py** | ML prediction engine | Random Forest classifier, Linear Regression, feature engineering |
| **weather_fetcher.py** | Data acquisition | Weather API integration, data parsing |
| **visualizer.py** | Results visualization | Matplotlib charts, crossing pattern detection |

## Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib requests
```

## ⚙️ Configuration

Edit `config.py` to set your location and API key:

```python
# Set your location from available options
LOCATION = {
    "latitude": COO["New York"][0],  # Change to your city
    "longitude": COO["New York"][1]
}

# Add your WeatherAPI key
API_KEY = "your_actual_api_key_here"
```
