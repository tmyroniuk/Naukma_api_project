"""
DEPRECATED
"""

import datetime as dt
from dateutil import parser
import json

import requests

from flask import Flask, jsonify, request

WEATHER_API_KEY=""
MY_API_TOKEN="123"

app = Flask(__name__)

class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/")
def home_page():
    return "<p><h2>KMA: Forcasting service.</h2></p>"

def verify_token(json_data):
    if json_data.get("token") is None:
        raise InvalidUsage("token is required", status_code=400)

    token = json_data.get("token")

    if token != MY_API_TOKEN:
        raise InvalidUsage("wrong API token", status_code=403)

def load_weather(location: str, contentType: str = "json"):
    url_base_url = "https://weather.visualcrossing.com"
    url_api = "VisualCrossingWebServices/rest/services/timeline"
    url_endpoint = f"{location}/next24hours"
    url_querry_params = f"unitGroup=metric&include=hours%2Cdays&key={WEATHER_API_KEY}&contentType={contentType}"

    url = f"{url_base_url}/{url_api}/{url_endpoint}?{url_querry_params}"
    print(url)

    payload = {}
    headers = {"Authorization": WEATHER_API_KEY}

    response = requests.request("GET", url, headers=headers, data=payload)
    print(response.status_code)
    return json.loads(response.text)

def is_date_suitable(datetime: str, tzoffset: float, timeSpan: int):
    utc_datetime = (parser.parse(datetime) - dt.timedelta(hours=tzoffset)).astimezone(dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    return now - dt.timedelta(hours=1) < utc_datetime <= now + dt.timedelta(hours=timeSpan-1)

def prepare_forecast_data(weather_json, location_name: str, timeSpan: int):
    result_data = {
        "location": location_name,
        "forecast": []
    }
    for day in weather_json["days"]:
        for hour in day["hours"]:
            hour_data = {}

            # Check if date is within timeSpan
            datetime = f"{day['datetime']} {hour['datetime']}"
            if not is_date_suitable(datetime, weather_json['tzoffset'], timeSpan):
                continue

            hour_data["day_datetime"] = day["datetime"]
            hour_data["day_tempmin"] = day["tempmin"]
            hour_data["day_tempmax"] = day["tempmax"]
            hour_data["day_temp"] = day["temp"]
            hour_data["day_dew"] = day["dew"]
            hour_data["day_humidity"] = day["humidity"]
            hour_data["day_precip"] = day["precip"]
            hour_data["day_precipcover"] = day["precipcover"]
            hour_data["day_solarradiation"] = day["solarradiation"]
            hour_data["day_solarenergy"] = day["solarenergy"]
            hour_data["day_uvindex"] = day["uvindex"]
            hour_data["day_sunrise"] = day["sunrise"]
            hour_data["day_sunset"] = day["sunset"]
            hour_data["day_moonphase"] = day["moonphase"]
            hour_data["hour_datetime"] = hour["datetime"]
            hour_data["hour_temp"] = hour["temp"]
            hour_data["hour_humidity"] = hour["humidity"]
            hour_data["hour_dew"] = hour["dew"]
            hour_data["hour_precip"] = hour["precip"]
            hour_data["hour_precipprob"] = hour["precipprob"]
            hour_data["hour_snow"] = hour["snow"]
            hour_data["hour_snowdepth"] = hour["snowdepth"]
            hour_data["hour_preciptype"] = hour["preciptype"]
            hour_data["hour_windgust"] = hour["windgust"]
            hour_data["hour_windspeed"] = hour["windspeed"]
            hour_data["hour_winddir"] = hour["winddir"]
            hour_data["hour_pressure"] = hour["pressure"]
            hour_data["hour_visibility"] = hour["visibility"]
            hour_data["hour_cloudcover"] = hour["cloudcover"]
            hour_data["hour_solarradiation"] = hour["solarradiation"]
            hour_data["hour_solarenergy"] = hour["solarenergy"]
            hour_data["hour_uvindex"] = hour["uvindex"]
            hour_data["hour_severerisk"] = hour["severerisk"]
            hour_data["hour_conditions"] = hour["conditions"]

            result_data["forecast"].append(hour_data)

    return result_data


@app.route(
    "/content/api/v1/integration/weather-forecast",
    methods=["POST"],
)
def forecast_endpoint():
    json_data = request.get_json()
    verify_token(json_data)

    location = ""
    if json_data.get("location"):
        location = json_data.get("location")

    weather_json = load_weather(location)
    prepared_data = prepare_forecast_data(weather_json, location, 12)

    return prepared_data