import requests
import datetime
from dateutil import parser
import json

def get_weather_json(location: str, contentType: str = "application/json"):
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
    return response.json()

def get_weather_dict(weather_json):
    weather_dict = {
    "day_tempmin": 0,
    "day_tempmax": 0,
    "day_temp": weather_json.get('currentConditions').get('temp'),
    "day_dew": ,
    "day_humidity": ,
    "day_precip": ,
    "day_precipcover": ,
    "day_solarradiation": ,
    "day_solarenergy": ,
    "day_uvindex": ,
    "day_sunrise": ,
    "day_sunset": ,
    "day_moonphase": ,
    "hour_datetime": ,
    "hour_temp": ,
    "hour_humidity": ,
    "hour_dew": ,
    "hour_precip": ,
    "hour_precipprob": ,
    "hour_snow": ,
    "hour_snowdepth": ,
    "hour_preciptype": ,
    "hour_windgust": ,
    "hour_windspeed": ,
    "hour_winddir": ,
    "hour_pressure": ,
    "hour_visibility": ,
    "hour_cloudcover": ,
    "hour_solarradiation": ,
    "hour_solarenergy": ,
    "hour_uvindex": ,
    "hour_severerisk": ,
    "hour_conditions": }
    return weather_dict