import joblib
import pandas as pd
import os

# 1. Проверка наличия файлов модели
if not os.path.exists('aqi_model.pkl') or not os.path.exists('features.pkl'):
    print("Ошибка: Файлы модели не найдены! Сначала запусти TrainModel.py.")
    exit()

# 2. Загрузка обученной модели и списка признаков
model = joblib.load('aqi_model.pkl')
features = joblib.load('features.pkl')

print("=== Прогноз качества воздуха (AQI) ===")
print("Введите параметры окружающей среды для получения прогноза:")

try:
    # 3. Сбор данных от пользователя
    temp = float(input("1. Температура воздуха (°C): "))
    hum = float(input("2. Влажность воздуха (%): "))
    wind = float(input("3. Скорость ветра (м/с): "))
    traffic = float(input("4. Интенсивность трафика (от 0 до 100): "))
    hour = int(input("5. Час суток (0-23): "))

    # 4. Превращение ввода в формат, понятный модели (DataFrame)
    user_input = pd.DataFrame([[temp, hum, wind, traffic, hour]], columns=features)

    # 5. Получение предсказания
    prediction = model.predict(user_input)[0]

    # 6. Вывод результата с расшифровкой
    print(f"\n" + "="*30)
    print(f"ПРОГНОЗ AQI: {prediction:.2f}")
    
    if prediction <= 50:
        print("Статус: ОТЛИЧНО (Воздух чистый)")
    elif prediction <= 100:
        print("Статус: УМЕРЕННО (Можно находиться на улице)")
    elif prediction <= 150:
        print("Status: ВРЕДНО (Для чувствительных групп)")
    else:
        print("Статус: ОПАСНО (Лучше остаться дома)")
    print("="*30)

except ValueError:
    print("\nОшибка: Пожалуйста, вводите только числа. Для дробных чисел используйте точку (например, 22.5).")
