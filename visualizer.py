import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_weather_predictions(predictions_df):

    one_day_df = predictions_df.head(24).copy() # 24h Prediction
    
    if 'hour' not in one_day_df.columns and 'datetime' in one_day_df.columns:
        one_day_df['hour'] = one_day_df['datetime'].dt.hour

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    temperatures = one_day_df['predicted_temperature'].values
    hours = one_day_df['hour'].values

    avg_temp = np.mean(temperatures)
    
    sorted_indices = np.argsort(hours)
    sorted_hours = hours[sorted_indices]
    sorted_temperatures = temperatures[sorted_indices]
    
    # Plot with connected lines showing the temperature trend
    ax1.plot(sorted_hours, sorted_temperatures, color='red', marker='o', linewidth=2, 
            markersize=5, label='Temperature Prediction', alpha=0.8)
    
    ax1.axhline(y=avg_temp, color='blue', linestyle='--', linewidth=1.5, 
               label=f'Average ({avg_temp:.1f} C)', alpha=0.7)

    ax1.set_title(f'24-Hour Temperature Forecast')
    ax1.set_ylabel('Temperature (C)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xlim(0, 23)
    ax1.legend()

    precip_values = one_day_df['predicted_precipitation'].values
    ax2.bar(hours, precip_values, color='blue', alpha=0.7, label='Precipitation')

    ax2.set_title('24-Hour Precipitation Forecast')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Precipitation (mm)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xlim(0, 23)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(f"\n24-Hour Weather Forecast:")
    print(f"Average Temperature: {avg_temp:.1f} C")
    print("-" * 40)
    
    for hour, row in one_day_df.iterrows():
        print(f"Hour {int(row['hour']):2d}: {row['predicted_temperature']:5.1f}°C | {row['predicted_precipitation']:4.1f}mm")
