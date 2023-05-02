"""
This module provides methods used for data colletion
"""

import requests
import json
import datetime
import os

from config import WEATHER_API_KEY, ALERTS_API_KEY

# Define a list of month names
__month_names = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']

def __save_page(url, file_name):
    # Downloads html page by given url and saves as file_name

    # Send a GET request to the URL and retrieve the response
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the response content to a file
        with open(f"{file_name}.html", "wb") as f:
            f.write(response.content)
        return "Page downloaded successfully."
    else:
        return f"Error downloading page. Status code: {response.status_code}"

def save_by_date(date):
    # Download ISW report from given date as datetime.datetime object

    # Parse month and day
    month_name = __month_names[date.month - 1]
    day = int(date.strftime("%d"))
    name = date.strftime("%Y-%m-%d")

    print(name)
    if os.path.exists(f"./Reports/{name}"):
        return "Page allready downloaded."

    if date.strftime("%Y")== "2022":
        if date == datetime.date(2022, 5, 5):
            url = "https://www.understandingwar.org/backgrounder/russian-campaign-assessment-may-5"
        elif date == datetime.date(2022, 7, 11):
            url = "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-update-july-11"
        elif date == datetime.date(2022, 8, 12):
            url = "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-august-12-0"
        elif date > datetime.date(2022, 2, 28):
            url = ("https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-%s-%d"%(month_name, day))
        elif date == datetime.date(2022, 2, 24):
            url = "https://www.understandingwar.org/backgrounder/russia-ukraine-warning-update-initial-russian-offensive-campaign-assessment"
        elif date == datetime.date(2022, 2, 25):
            url = "https://www.understandingwar.org/backgrounder/russia-ukraine-warning-update-russian-offensive-campaign-assessment-february-25-2022"
        elif date == datetime.date(2022, 2, 28):
            url = "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-28-2022"
        else:
            url = ("https://www.understandingwar.org/backgrounder/russia-ukraine-warning-update-russian-offensive-campaign-assessment-%s-%d"%(month_name, day))
    else:
        if date == datetime.date(2023, 2, 5):
            url = "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-update-february-5-2023"
        else:
            year = date.strftime("%Y")
            url = ("https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-%s-%d-%s"%(month_name, day, year))
    return __save_page(url, f"Reports/{name}")

def save_all():
    # Downloads all ISW reports from Feb 22 2022 till today

    # Set the start date to February 24, 2022
    start_date = datetime.date(2022, 2, 24)

    # Get the current date
    end_date = datetime.date.today()

    # Define a list of month names
    month_names = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']

    # Iterate over the dates from the start date to the end date
    for date in (start_date + datetime.timedelta(n) for n in range((end_date - start_date).days)):
        save_by_date(date)


def load_alerts(contentType: str = "application/json"):
    # Loads list of active alerts in States

    url_base_url = "https://api.ukrainealarm.com"
    url_api = "api/v3"
    url_endpoint = "alerts"
    url_querry_params = ""

    url = f"{url_base_url}/{url_api}/{url_endpoint}?{url_querry_params}"

    payload = {}
    headers = {
        "Authorization": ALERTS_API_KEY,
        "accept": contentType
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    alarms = [alarm['activeAlerts'] for alarm in json.loads(response.text) if alarm['regionType'] == 'State']
    alarms = sum(alarms, [])
    return [alarm for alarm in alarms if alarm['type'] == 'AIR']

def load_history(regionId: int, contentType: str = "application/json"):
    # Loads last 25 alerts in given region (uses alerts-api id system)

    url_base_url = "https://api.ukrainealarm.com"
    url_api = "api/v3"
    url_endpoint = "alerts/regionHistory"
    url_querry_params = f"regionId={regionId}"

    url = f"{url_base_url}/{url_api}/{url_endpoint}?{url_querry_params}"

    payload = {}
    headers = {
        "Authorization": ALERTS_API_KEY,
        "accept": contentType
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    return [alarm for alarm in json.loads(response.text)[0]['alarms'] if alarm['alertType'] == "AIR"]

def load_states(contentType: str = "application/json"):
    # Loads info on all State regions including their names and ids

    url_base_url = "https://api.ukrainealarm.com"
    url_api = "api/v3"
    url_endpoint = "regions"
    url_querry_params = ""

    url = f"{url_base_url}/{url_api}/{url_endpoint}?{url_querry_params}"

    payload = {}
    headers = {
        "Authorization": ALERTS_API_KEY,
        "accept": contentType
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    regions_dict = json.loads(response.text)
    for state in regions_dict["states"]:
        state.pop("regionChildIds")
    return regions_dict['states']

def load_weather(location: str, contentType: str = "json"):
    # Loads weaher forecast for next 24 hours in given location

    url_base_url = "https://weather.visualcrossing.com"
    url_api = "VisualCrossingWebServices/rest/services/timeline"
    url_endpoint = f"{location}/next24hours"
    url_querry_params = f"unitGroup=metric&include=hours%2Cdays&key={WEATHER_API_KEY}&contentType={contentType}"

    url = f"{url_base_url}/{url_api}/{url_endpoint}?{url_querry_params}"

    payload = {}
    headers = {"Authorization": WEATHER_API_KEY}

    response = requests.request("GET", url, headers=headers, data=payload)
    return json.loads(response.text)

print(load_states())

