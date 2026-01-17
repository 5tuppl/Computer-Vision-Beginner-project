import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import os

# 1. Загрузка и подготовка данных
if not os.path.exists('air_quality_data.csv'):
    print("Ошибка: Файл 'air_quality_data.csv' не найден. Сначала запусти скрипт генерации данных!")
    exit()

df = pd.read_csv('air_quality_data.csv')

# Преобразуем колонку времени, чтобы модель могла извлечь пользу из часа суток
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour

# Выбираем признаки (факторы влияния) и целевую переменную (AQI)
features = ['temperature', 'humidity', 'wind_speed', 'traffic_index', 'hour']
X = df[features]
y = df['AQI']

# 2. Разделение данных на обучающую и тестовую выборки (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Создание и обучение модели
# Используем Random Forest — это мощный алгоритм, состоящий из множества "деревьев решений"
model = RandomForestRegressor(n_estimators=100, random_state=42)
print("Начинаю обучение модели... Это может занять несколько секунд.")
model.fit(X_train, y_train)

# 4. Проверка качества модели
y_pred = model.predict(X_test)

# Математические метрики
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Результаты анализа ===")
print(f"Средняя ошибка (MAE): {mae:.2f}") 
print(f"Коэффициент детерминации (R2): {r2:.4f}") 

# 5. Визуализация 1: Сравнение прогноза и реальности
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2, linestyle='--')
plt.title('Прогноз vs Реальные данные (AQI)')
plt.xlabel('Реальный AQI')
plt.ylabel('Предсказанный AQI')
plt.grid(True)

# Сохраняем график в файл
plt.savefig('model_performance.png')
plt.close() # Важно закрыть фигуру
print("\n[Инфо] График точности сохранен как 'model_performance.png'")

# 6. Визуализация 2: Влияние признаков (SHAP)
print("Рассчитываю влияние факторов (SHAP)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Создаем график SHAP без вывода на экран
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('feature_importance_shap.png')
plt.close()
print("[Инфо] График важности признаков сохранен как 'feature_importance_shap.png'")

print("\nВсе этапы завершены успешно!")

import joblib

# Сохраняем модель в файл
joblib.dump(model, 'aqi_model.pkl')
# Сохраняем список признаков, чтобы не забыть порядок
joblib.dump(features, 'features.pkl')

print("Модель успешно сохранена в файл 'aqi_model.pkl'!")



import joblib

features = ['temperature', 'humidity', 'wind_speed', 'traffic_index', 'hour']

joblib.dump(model, 'aqi_model.pkl')
joblib.dump(features, 'features.pkl')

print("Файлы aqi_model.pkl и features.pkl обновлены!")
