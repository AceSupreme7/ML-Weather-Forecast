import config
from weather_predictor import WeatherPredictor
from visualizer import visualize_weather_predictions

def main():
    # Retrieve configuration values from config.py
    latitude = config.LOCATION['latitude']
    longitude = config.LOCATION['longitude']
    start_date = config.PREDICTION_START_DATE
    end_date = config.PREDICTION_END_DATE
    api_key = config.API_KEY  # Retrieve API key for WeatherAPI

    # Initialize the WeatherPredictor with the configuration params
    predictor = WeatherPredictor(latitude, longitude, start_date, end_date)

    predictor.load_historical_data() 
    predictor.feature_engineering() 
    predictor.run_prediction()

if __name__ == "__main__":
    main()
