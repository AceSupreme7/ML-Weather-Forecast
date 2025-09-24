import requests
from config import API_KEY, BASE_URL, WEATHER_PARAMETERS, HISTORICAL_START_DATE, HISTORICAL_END_DATE, LOCATION

def get_historical_weather(lat, lon, start_date, end_date):
    location = f"{lat},{lon}"
    url = f"{BASE_URL}/history.json"

    params = {
        "key": API_KEY,
        "q": location,
        "dt": start_date,
        "end_dt": end_date,
        "include": "hours",
        "contentType": "json"
    }

    print(f"[INFO] Requesting weather data from: {url}")
    print(f"[DEBUG] Params: {params}")

    try:
        response = requests.get(url, params=params)
        print(f"[DEBUG] Response Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            weather_data = {}

            for entry in data.get('forecast', {}).get('forecastday', []):
                date = entry.get('date')
                daily_stats = {
                    'temperature_sum': 0.0,
                    'humidity_sum': 0.0,
                    'precipitation': 0.0,
                    'cloud_sum': 0.0,
                    'wind_sum': 0.0,
                    'pressure_sum': 0.0,
                    'dewpoint_sum': 0.0,
                    'uv_sum': 0.0,
                    'conditions': [],
                    'count': 0
                }

                for hour in entry.get('hour', []):
                    temp = hour.get('temp_c')
                    humidity = hour.get('humidity')
                    precip = hour.get('precip_mm', 0.0)
                    cloud = hour.get('cloud')
                    wind = hour.get('wind_kph')
                    pressure = hour.get('pressure_mb')
                    dewpoint = hour.get('dewpoint_c')
                    uv = hour.get('uv')
                    condition = hour.get('condition', {}).get('text', 'Unknown')

                    if temp is not None:
                        daily_stats['temperature_sum'] += temp
                        daily_stats['humidity_sum'] += humidity or 0.0
                        daily_stats['cloud_sum'] += cloud or 0.0
                        daily_stats['wind_sum'] += wind or 0.0
                        daily_stats['pressure_sum'] += pressure or 0.0
                        daily_stats['dewpoint_sum'] += dewpoint or 0.0
                        daily_stats['uv_sum'] += uv or 0.0
                        daily_stats['precipitation'] += precip
                        daily_stats['conditions'].append(condition)
                        daily_stats['count'] += 1

                count = daily_stats['count']
                if count == 0:
                    continue  # skip empty days

                # Most frequent condition
                condition_mode = max(set(daily_stats['conditions']), key=daily_stats['conditions'].count)

                weather_data[date] = {
                    'avg_temperature': round(daily_stats['temperature_sum'] / count, 2),
                    'total_precipitation': round(daily_stats['precipitation'], 2),
                    'avg_humidity': round(daily_stats['humidity_sum'] / count, 2),
                    'avg_cloud': round(daily_stats['cloud_sum'] / count, 2),
                    'avg_wind_kph': round(daily_stats['wind_sum'] / count, 2),
                    'avg_pressure_mb': round(daily_stats['pressure_sum'] / count, 2),
                    'avg_dewpoint_c': round(daily_stats['dewpoint_sum'] / count, 2),
                    'avg_uv': round(daily_stats['uv_sum'] / count, 2),
                    'weather_condition': condition_mode
                }

            return weather_data

        else:
            print(f"[ERROR] Failed to fetch data: Status {response.status_code}")
            print(f"[DETAIL] Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"[EXCEPTION] Request failed: {e}")
        return None
    except Exception as e:
        print(f"[EXCEPTION] Unexpected error: {e}")
        return None


def main():
    lat = LOCATION["latitude"]
    lon = LOCATION["longitude"]
    start_date = HISTORICAL_START_DATE
    end_date = HISTORICAL_END_DATE

    data = get_historical_weather(lat, lon, start_date, end_date)
    # print('======================DEBUG======================\nWeather Data', data)
    if data:
        print(f"\nWeather Data for {lat}, {lon} from {start_date} to {end_date}:")
        for date, values in sorted(data.items()):
            print(f"Date: {date} | Avg Temp: {values['avg_temperature']}°C | Total Precip: {values['total_precipitation']} mm")
    else:
        print("No data retrieved.")


if __name__ == "__main__":
    main()
