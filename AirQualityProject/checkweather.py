import requests

API_KEY = "318c0d0a7f93e0c4299cc55d8b5c204f"  # Вставь сюда свой ключ

CITY = "Almaty" # Или любой другой город

# Ссылка для запроса текущей погоды
url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print(f"Город: {data['name']}")
    print(f"Температура: {data['main']['temp']}°C")
    print(f"Влажность: {data['main']['humidity']}%")
    print(f"Скорость ветра: {data['wind']['speed']} м/с")
else:
    print(f"Ошибка! Код: {response.status_code}")
    print("Возможно, ключ еще не активировался. Подожди 15-30 минут.")
