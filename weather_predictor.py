import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from weather_fetcher import get_historical_weather
from datetime import timedelta
import math

class WeatherPredictor:
    def __init__(self, lat, lon, start_date, end_date):
        self.lat = lat
        self.lon = lon
        self.start_date = start_date
        self.end_date = end_date
        self.weather_df = None
        self.condition_model = None
        self.precipitation_model = None
        self.temperature_model = None
        self.label_encoder = LabelEncoder()
        self.condition_classes = []

    def load_historical_data(self):
        data = get_historical_weather(self.lat, self.lon, self.start_date, self.end_date)
        if not data:
            raise ValueError("No historical weather data available.")
        df = pd.DataFrame.from_dict(data, orient='index')
        df.reset_index(drop=True, inplace=True)
        df['datetime'] = pd.date_range(start=self.start_date, periods=len(df), freq='H')
        df['hour'] = df['datetime'].dt.hour
        df.dropna(inplace=True)
        self.weather_df = df
        print("1- OK! Historical weather data loaded.")
        return data

    def feature_engineering(self):
        df = self.weather_df.copy()
        df['condition_encoded'] = self.label_encoder.fit_transform(df['weather_condition'].astype(str))
        self.condition_classes = list(self.label_encoder.classes_)

        df['prev_temp'] = df['avg_temperature'].shift(1).bfill()
        df['prev_precip'] = df['total_precipitation'].shift(1).bfill()
        df['prev_condition_encoded'] = df['condition_encoded'].shift(1).bfill()

        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['season'] = ((df['day_of_year'] // 91) % 4).astype(int)

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        recency_weights = np.linspace(1, 0.1, len(df))
        df['recency_weight'] = recency_weights

        df['net_radiation'] = self.net_radiation(df)
        df['heat_index'] = self.heat_index(df)
        df['wind_chill'] = self.wind_chill(df)
        df['relative_humidity'] = self.relative_humidity(df)
        df['dew_point'] = self.dew_point(df)
        df['sensible_heat_flux'] = self.sensible_heat_flux(df)
        df['latent_heat_flux'] = self.latent_heat_flux(df)

        self.weather_df = df
        print("2- OK! Feature engineering complete.")

    def net_radiation(self, df):
        Q_sw_down = 1000
        alpha = 0.3
        theta = np.pi / 4
        Q_sw_up = Q_sw_down * alpha
        Q_sw = Q_sw_down * (1 - alpha) * np.cos(theta)

        Q_clear = 300
        cloud_fraction = df['avg_cloud'] / 100.0
        Q_lw_down = Q_clear * (1 + 0.25 * cloud_fraction ** 2)

        sigma = 5.67e-8
        Q_lw_up = 0.95 * sigma * (df['avg_temperature'] + 273.15) ** 4

        return Q_sw - Q_sw_up + Q_lw_down - Q_lw_up

    def heat_index(self, df):
        T = df['avg_temperature']
        RH = df['avg_humidity']
        return -8.784 + 1.611 * T + 2.338 * RH - 0.146 * T * RH

    def wind_chill(self, df):
        T = df['avg_temperature']
        V = df['avg_wind_kph']
        return 13.12 + 0.6215 * T - 11.37 * V ** 0.16 + 0.3965 * T * V ** 0.16

    def relative_humidity(self, df):
        e = df['avg_dewpoint_c'].apply(lambda Td: 6.112 * np.exp((17.67 * Td) / (Td + 243.5)))
        es = df['avg_temperature'].apply(lambda T: 6.112 * np.exp((17.67 * T) / (T + 243.5)))
        return (e / es * 100).clip(0, 100)

    def dew_point(self, df):
        RH = df['avg_humidity']
        T = df['avg_temperature']
        return 243.5 * (np.log(RH / 100) + (17.67 * T) / (T + 243.5)) / (17.67 - np.log(RH / 100))

    def sensible_heat_flux(self, df):
        rho = 1.225
        cp = 1005
        CH = 0.01
        U = df['avg_wind_kph'] / 3.6
        Ts = df['avg_temperature']
        Ta = df['avg_dewpoint_c']
        return rho * cp * CH * U * (Ts - Ta)

    def latent_heat_flux(self, df):
        rho = 1.225
        Lv = 2.5e6
        CE = 0.01
        U = df['avg_wind_kph'] / 3.6
        qs = df['avg_humidity'] / 100
        qa = df['avg_dewpoint_c'] / 100
        return rho * Lv * CE * U * (qs - qa)

    def prepare_classification_data(self):
        features = self.weather_df.drop(columns=['weather_condition', 'datetime'])
        target = self.weather_df['condition_encoded']
        return features, target

    def prepare_regression_data(self, target):
        features = self.weather_df.drop(columns=['weather_condition', 'datetime', target])
        y = self.weather_df[target]
        return features, y

    def train_classifier(self, X, y):
        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(X, y)
        return clf

    def train_regressor(self, X, y):
        model = LinearRegression()
        model.fit(X, y)
        model.feature_names_in_ = X.columns
        return model

    def analyze_location_climate(self):
        df = self.weather_df.copy()
        
        if len(df) == 0:
            # Default fallback patterns if no historical data
            return {
                'hourly_avg': pd.Series([15] * 24),  # Default 15C for all hours
                'daily_min_hour': 5,
                'daily_max_hour': 14,
                'daily_range': 10,
                'seasonal_avg': pd.Series([15] * 4),
                'overall_avg': 15,
                'temp_std': 5,
                'min_historical': 0,
                'max_historical': 30
            }
        
        # Get average temperatures by hour to understand daily pattern
        hourly_avg = df.groupby('hour')['avg_temperature'].mean()
        
        daily_min_hour = hourly_avg.idxmin()
        daily_max_hour = hourly_avg.idxmax()
        daily_range = hourly_avg.max() - hourly_avg.min()
        
        # Get seasonal averages
        seasonal_avg = df.groupby('season')['avg_temperature'].mean()
        
        # Get overall statistics for realistic bounds
        overall_avg = df['avg_temperature'].mean()
        temp_std = df['avg_temperature'].std()
        min_historical = df['avg_temperature'].min()
        max_historical = df['avg_temperature'].max()
        
        # Realistic bounds
        realistic_min = min_historical - 5
        realistic_max = max_historical + 5
        
        print(f"INFO! Location Analysis (Lat: {self.lat}, Lon: {self.lon})")
        print(f"   Historical temp range: {min_historical:.1f} C to {max_historical:.1f} C")
        print(f"   Daily pattern: Min at {daily_min_hour:02d}:00, Max at {daily_max_hour:02d}:00")
        print(f"   Typical daily range: {daily_range:.1f}°C")
        
        return {
            'hourly_avg': hourly_avg,
            'daily_min_hour': daily_min_hour,
            'daily_max_hour': daily_max_hour,
            'daily_range': daily_range,
            'seasonal_avg': seasonal_avg,
            'overall_avg': overall_avg,
            'temp_std': temp_std,
            'min_historical': min_historical,
            'max_historical': max_historical,
            'realistic_min': realistic_min,
            'realistic_max': realistic_max
        }

    def generate_location_based_temperature(self, hour, day_of_year, climate):
        """Generate temperature based on this specific location's climate patterns"""
        
        # Use the location's historical patterns
        overall_avg = climate['overall_avg']
        daily_range = climate['daily_range']
        min_hour = climate['daily_min_hour']
        max_hour = climate['daily_max_hour']
        
        # Seasonal adjustment based on latitude (more extreme seasons at higher latitudes)
        lat_factor = abs(self.lat) / 90.0 
        seasonal_intensity = 15 * lat_factor # more seasonal variation at higher latitudes
        
        seasonal_shift = seasonal_intensity * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        seasonal_base = overall_avg + seasonal_shift
        
        # Daily oscillation based on this location's pattern
        phase_shift = (min_hour / 24) * 2 * np.pi
        
        # Smooth daily oscillation using sine wave
        daily_variation = (daily_range / 2) * np.sin(2 * np.pi * hour / 24 - phase_shift - np.pi/2)
        
        # Add some randomness based on this location's temperature variability
        random_variation = np.random.normal(0, climate['temp_std'] * 0.1)

        temperature = seasonal_base + daily_variation + random_variation
        temperature = np.clip(temperature, climate['realistic_min'], climate['realistic_max'])
        
        return round(temperature, 1)

    def generate_future_data(self, hours=168):
        base_date = self.weather_df.iloc[-1]['datetime']
        last_values = self.weather_df.iloc[-1]
        future_data = []

        climate = self.analyze_location_climate()
        
        print(f"--> Generating predictions for location (Lat: {self.lat}, Lon: {self.lon})")

        for i in range(hours):
            dt = base_date + timedelta(hours=i + 1)
            hour = dt.hour
            doy = dt.timetuple().tm_yday
            
            temp = self.generate_location_based_temperature(hour, doy, climate)
            
            # Cloud cover pattern varies by location
            if abs(self.lat) < 30: # Tropical regions
                base_cloud = 40 + np.random.normal(0, 25) # More variable clouds
            else: # Temperate regions
                if 10 <= hour <= 16: # Daytime
                    base_cloud = 50 + np.random.normal(0, 20)
                else: # Nighttime
                    base_cloud = 30 + np.random.normal(0, 15)
            
            cloud_variation = np.clip(base_cloud, 0, 100)
            
            # Humidity varies with temperature and location
            if abs(self.lat) < 30: # Tropical: generally more humid
                base_humidity = 75 - (temp - climate['overall_avg']) * 0.8
            else:
                base_humidity = 65 - (temp - climate['overall_avg']) * 1.2
            
            humidity_variation = np.clip(base_humidity + np.random.normal(0, 5), 20, 95)
            
            # Precipitation probability based on location and conditions
            if abs(self.lat) < 30: # Tropical: higher rain probability
                precip_prob = 0.15
            else:
                precip_prob = 0.08
                
            if cloud_variation > 75:
                precip_prob *= 2
            if cloud_variation > 85 and humidity_variation > 80:
                precip_prob *= 3
                
            precipitation = np.random.exponential(0.1) if np.random.random() < precip_prob else 0

            # Wind patterns based on latitude
            if abs(self.lat) > 60: # Polar regions: often windier
                base_wind = 15 + np.random.normal(0, 5)
            elif abs(self.lat) < 20: # Tropical: variable winds
                base_wind = 10 + np.random.normal(0, 6)
            else: # Temperate: moderate winds
                base_wind = 12 + np.random.normal(0, 4)

            row = {
                'datetime': dt,
                'hour': hour,
                'avg_temperature': temp,
                'avg_cloud': round(cloud_variation, 1),
                'avg_humidity': round(humidity_variation, 1),
                'avg_uv': max(0, min(12, (3 + abs(self.lat)/90*8) + np.random.normal(0, 1))), # UV varies with latitude
                'total_precipitation': round(precipitation, 2),
                'avg_wind_kph': max(0, base_wind),
                'avg_dewpoint_c': round(temp - np.random.uniform(2, 8), 1),
                'prev_condition_encoded': self.label_encoder.transform([last_values['weather_condition']])[0],
                'condition_encoded': np.random.randint(0, len(self.condition_classes)),
                'avg_pressure_mb': 1013 + np.random.normal(0, 5),
                'prev_temp': temp,
                'prev_precip': round(precipitation, 2),
                'day_of_year': doy,
                'season': (doy // 91) % 4,
                'hour_sin': np.sin(2 * np.pi * hour / 24),
                'hour_cos': np.cos(2 * np.pi * hour / 24),
                'recency_weight': 1
            }

            # Calculate derived features
            temp_df = pd.DataFrame([row])
            temp_df['net_radiation'] = self.net_radiation(temp_df)
            temp_df['heat_index'] = self.heat_index(temp_df)
            temp_df['wind_chill'] = self.wind_chill(temp_df)
            temp_df['relative_humidity'] = self.relative_humidity(temp_df)
            temp_df['dew_point'] = self.dew_point(temp_df)
            temp_df['sensible_heat_flux'] = self.sensible_heat_flux(temp_df)
            temp_df['latent_heat_flux'] = self.latent_heat_flux(temp_df)

            future_data.append(temp_df.iloc[0])

        return pd.DataFrame(future_data)

    def run_prediction(self):
        self.feature_engineering()

        # Train condition classifier
        X_clf, y_clf = self.prepare_classification_data()
        self.condition_model = self.train_classifier(X_clf, y_clf)

        future_df = self.generate_future_data()

        future_df['condition_encoded'] = self.condition_model.predict(future_df[self.condition_model.feature_names_in_])
        future_df['predicted_condition'] = self.label_encoder.inverse_transform(future_df['condition_encoded'].astype(int))

        # Use our location-based temperature model
        future_df['predicted_temperature'] = future_df['avg_temperature']

        # Precipitation based on predicted conditions
        wet_conditions = [c for c in self.condition_classes if any(word in c.lower() for word in ['rain', 'storm', 'drizzle', 'shower', 'thunder'])]
        
        future_df['predicted_precipitation'] = future_df['total_precipitation']
        # Set precipitation to 0 if conditions don't indicate wet weather
        future_df.loc[~future_df['predicted_condition'].isin(wet_conditions), 'predicted_precipitation'] = 0

        future_df['day'] = future_df['datetime'].dt.day_name()

        self.visualize_predictions(future_df)

        print(f"-->  First 24 hours of predictions for ({self.lat}, {self.lon}):")
        print(future_df[['datetime', 'hour', 'predicted_temperature', 'predicted_condition', 'predicted_precipitation']].head(24))

    def visualize_predictions(self, df):
        from visualizer import visualize_weather_predictions
        visualize_weather_predictions(df)
