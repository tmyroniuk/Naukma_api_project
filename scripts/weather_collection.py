import requests
import datetime
import pickle
from dateutil import parser
import json

def get_weather(location: str, contentType: str = "application/json"):
    LOCAL_WEATHER_SERVICE_KEY = "123"

    url = "http://3.127.248.223:8000/content/api/v1/integration/weather-forecast"
    headers = {"Content-Type": "application/json"}
    data = {
        "token": LOCAL_WEATHER_SERVICE_KEY,
        "requester_name": __name__,
        "location": location
    }

    response = requests.post(url, headers=headers, json=data)

    print(response.status_code)
    return json.loads(response.text)