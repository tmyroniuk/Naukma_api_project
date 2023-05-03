import datetime as dt
import json
import pandas as pd
import requests

from dateutil import parser
from flask import Flask, jsonify, request

from config import PREDICTIONS_FILE, REGIONS_DATASET, MY_API_TOKEN, STATE_FILE
import generate_predictions_script

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
    return "<p><h2>KMA: ALARM Forcasting service.</h2></p>"

def verify_token(json_data):
    if json_data.get("token") is None:
        raise InvalidUsage("token is required", status_code=400)

    token = json_data.get("token")

    if token != MY_API_TOKEN:
        raise InvalidUsage("wrong API token", status_code=403)

def read_prediction_file():
    df = pd.read_csv(PREDICTIONS_FILE, sep=';')
    df['date_time'] = df['date_time'].apply(pd.to_datetime)
    df['alarm_marker'] = df['alarm_marker'].astype('bool')
    return df

def read_regions_file():
    df = pd.read_csv(REGIONS_DATASET, sep=',')
    return df

def get_region_id(region_name, regions_df):
    if((region_name == 'all') | (region_name == None)):
        print('all regions')
        return regions_df['region_id'].values

    region_row = (regions_df[regions_df['region'] == region_name])
    if(region_row.empty):
        return [-1]

    return [region_row['region_id'].iloc[0]]

def get_reg_name(reg_id, regions_df):
    return (regions_df[regions_df['region_id'] == reg_id])['region'].iloc[0]

def get_status():
    try:
        with open(STATE_FILE, 'r') as handle:
            return json.load(handle)
    except:
        return {
            "last_prediciotn_time": ""
        }

def get_alarms_for_regions(region_ids, regions_df, predict_df):

    all_regions_forecast = {}
    time_forecast = {}
    for reg_id in region_ids:
        df_region = predict_df[predict_df['region_id'] == reg_id]
        time_forecast = {}
        for index, row in df_region.iterrows():
            time = row['date_time'].strftime('%Y-%m-%d %H:%M')
            time_forecast[time] = row['alarm_marker']

        all_regions_forecast[get_reg_name(reg_id, regions_df)] = time_forecast

    return all_regions_forecast

@app.route(
    "/content/api/alarm-forecast",
    methods=["POST"],
)
def load_forecast_endpoint():
    predict_df = read_prediction_file()
    regions_df = read_regions_file()
    json_data = request.get_json()

    region_ids = get_region_id(json_data.get('region'), regions_df)
    result_payload = {}
    if(region_ids[0] == -1):
        result_payload = {
            "Error": "The specified region is not supported"
        }
    else:
        # Load forecast
        time_forecast = get_alarms_for_regions(region_ids, regions_df, predict_df)
        # Load status
        status = get_status()

        result_payload = {
            'last_prediciotn_time': status['last_prediciotn_time'],
            'regions_forecast': time_forecast
        }

    return result_payload

@app.route(
    "/content/api/relaunch-forecast",
    methods=["POST"],
)
def relaunch_forecast_endpoint():
    generate_predictions_script.main()
    return "Forecast regenerated sucessfully"
