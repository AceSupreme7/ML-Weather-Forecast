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
