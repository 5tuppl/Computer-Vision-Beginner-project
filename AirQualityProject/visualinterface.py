import streamlit as st
import pandas as pd
import joblib
import requests
import os
import random
from datetime import datetime

# --- 1. НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="AQI Predictor Pro", page_icon="OO", layout="wide")

# --- 2. КОНФИГУРАЦИЯ API ---
# Вставь свой ключ сюда. Если он еще не активирован, программа перейдет в демо-режим.
API_KEY = "318c0d0a7f93e0c4299cc55d8b5c204f" 


# --- 3. ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_model_files():
    if os.path.exists('aqi_model.pkl') and os.path.exists('features.pkl'):
        model = joblib.load('aqi_model.pkl')
        features = joblib.load('features.pkl')
        return model, features
    return None, None

model, features = load_model_files()

# --- 4. ФУНКЦИЯ ПОЛУЧЕНИЯ ПОГОДЫ ---
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            return res.json(), False  # Возвращаем данные и флаг ошибки (False)
        else:
            # Если 401 или другая ошибка — включаем демо-режим
            demo_data = {
                'main': {'temp': random.uniform(15, 25), 'humidity': random.randint(30, 70)},
                'wind': {'speed': random.uniform(1, 7)},
                'name': f"{city} (Demo)"
            }
            return demo_data, True # Флаг Demo = True
    except:
        return None, True

# --- 5. ИНТЕРФЕЙС (БОКОВАЯ ПАНЕЛЬ) ---
st.sidebar.header("Настройки данных")
mode = st.sidebar.radio("Режим ввода:", ["Реальный город (API)", "Ручной ввод"])

# Значения по умолчанию
temp, hum, wind, traffic, hour = 20.0, 50, 3.0, 30, datetime.now().hour

if mode == "Реальный город (API)":
    city_input = st.sidebar.text_input("Введите город:", "Almaty")
    if st.sidebar.button("Обновить погоду"):
        data, is_demo = get_weather(city_input)
        if data:
            temp = data['main']['temp']
            hum = data['main']['humidity']
            wind = data['wind']['speed']
            if is_demo:
                st.sidebar.warning("API ключ еще не активен. Работает демо-режим.")
            else:
                st.sidebar.success(f"Данные для {city_input} получены!")

# Слайдеры для точной настройки
st.sidebar.subheader("Уточнение параметров")
temp = st.sidebar.slider("Температура (°C)", -20.0, 45.0, float(temp))
hum = st.sidebar.slider("Влажность (%)", 0, 100, int(hum))
wind = st.sidebar.slider("Ветер (м/с)", 0.0, 20.0, float(wind))
traffic = st.sidebar.slider("Трафик (0-100)", 0, 100, 30)
hour = st.sidebar.slider("Час (0-23)", 0, 23, int(hour))

# --- 6. ОСНОВНОЙ ЭКРАН ---
st.title("🌍 Система мониторинга качества воздуха")

if model is None:
    st.error("Файлы модели не найдены! Запусти сначала TrainModel.py")
else:
    # Расчет текущего прогноза
    input_df = pd.DataFrame([[temp, hum, wind, traffic, hour]], columns=features)
    current_prediction = model.predict(input_df)[0]

    # Вывод результата
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Предсказанный индекс AQI", value=f"{current_prediction:.2f}")
        if current_prediction <= 50:
            st.success("Статус: Чистый воздух 🌱")
        elif current_prediction <= 100:
            st.warning("Статус: Умеренное загрязнение 🟡")
        else:
            st.error("Статус: Опасно для здоровья! 🔴")

    with col2:
        st.write("**Текущие условия:**")
        st.write(f"🌡 Температура: {temp}°C | 💧 Влажность: {hum}%")
        st.write(f"💨 Ветер: {wind} м/с | 🚗 Трафик: {traffic}%")

    # --- 7. ГРАФИК ПРОГНОЗА НА 24 ЧАСА ---
    st.divider()
    st.subheader(" Динамика AQI в течение суток")
    
    forecast_data = []
    for h in range(24):
        # Имитация: утром и вечером трафик выше, днем теплее
        h_traffic = 85 if h in [8, 9, 17, 18, 19] else 30
        h_temp = temp + (3 if 10 <= h <= 17 else -2)
        
        h_df = pd.DataFrame([[h_temp, hum, wind, h_traffic, h]], columns=features)
        forecast_data.append(model.predict(h_df)[0])

    chart_df = pd.DataFrame({'Час': range(24), 'AQI': forecast_data})
    st.line_chart(chart_df.set_index('Час'))
    st.caption("Этот график показывает, как будет меняться загрязнение в зависимости от времени и пробок.")

    # --- 8. АНАЛИЗ ФАКТОРОВ ---
    if st.checkbox("Показать анализ важности факторов (SHAP)"):
        if os.path.exists('feature_importance.png'):
            st.image('feature_importance.png')
        else:
            st.info("Файл feature_importance.png не найден.")
