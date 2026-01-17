import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Настройка случайных чисел для повторяемости
np.random.seed(42)

def generate_aqi_data(samples=1000):
    start_date = datetime(2023, 1, 1)
    
    data = {
        'timestamp': [start_date + timedelta(hours=i) for i in range(samples)],
        'temperature': np.random.uniform(-10, 35, samples), # Температура от -10 до +35
        'humidity': np.random.uniform(20, 90, samples),    # Влажность в %
        'wind_speed': np.random.uniform(0, 15, samples),   # Скорость ветра м/с
        'traffic_index': np.random.uniform(0, 100, samples) # Плотность трафика
    }
    
    df = pd.DataFrame(data)
    
    # Создаем зависимость для AQI (наша целевая переменная)
    # AQI растет от трафика, падает от ветра и немного зависит от температуры
    df['AQI'] = (
        df['traffic_index'] * 0.8 + 
        df['humidity'] * 0.2 - 
        df['wind_speed'] * 5 + 
        50 + np.random.normal(0, 5, samples)
    )
    
    # Ограничим AQI разумными пределами (от 0 до 500)
    df['AQI'] = df['AQI'].clip(lower=0, upper=500)
    
    return df

# Создаем данные
air_quality_df = generate_aqi_data(2000)

# Сохраняем в CSV файл
air_quality_df.to_csv('air_quality_data.csv', index=False)

print("Файл 'air_quality_data.csv' успешно создан! В нем 2000 строк.")
print(air_quality_df.head()) # Показывает первые 5 строк
