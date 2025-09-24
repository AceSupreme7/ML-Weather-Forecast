from datetime import datetime, timedelta

# === API Configuration ===
API_KEY = "YOUR_API_KEY"  # Replace with your valid API
BASE_URL = "http://api.weatherapi.com/v1" 

# === Location Settings ===

COO = {
    "Paris": [48.864716, 2.349014],
    "Moscow": [55.751244, 37.618423],
    "Chad": [15.4542, 18.7322],
    "New York": [40.730610, -73.935242],
}

LOCATION = {
    "latitude": COO["Chad"][0],
    "longitude": COO["Chad"][1]
}

# === Query Parameters ===

WEATHER_PARAMETERS = "temp_c,precip_mm" 

# === Time Ranges ===

today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
HISTORICAL_START_DATE = (today - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ") # Last 7 days
HISTORICAL_END_DATE = (today).strftime("%Y-%m-%dT%H:%M:%SZ")

now = datetime.utcnow()
next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
PREDICTION_START_DATE = next_hour.strftime("%Y-%m-%dT%H:%M:%SZ")
PREDICTION_END_DATE = (next_hour + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

