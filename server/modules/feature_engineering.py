"""
This module provides methods used for creating new features
"""

import datetime as dt
import pandas as pd
import numpy as np
import pytz

from dateutil import parser

from modules.data_collection import load_history, load_alerts, load_states
from config import *

def __isNaN(num):
    return num != num

__history_cache = {}

def __load_history(region_id: int):
    # Load region alarms history
    if region_id in __history_cache:
        # Found in cache, extract cache
        region_history = __history_cache[region_id]
        return region_history
    else:
        # History for this region hasn't been downloaded yet
        region_history = load_history(region_id)
        __history_cache[region_id] = region_history
        return region_history

# Resets cache for region alarms history
def reset_cache():
    __history_cache = {}

def event_holiday_is_near(holiday_df, row):
    datetime = parser.parse(f"{row['day_datetime']} {row['hour_datetime']}")
    value = (np.abs((holiday_df['date'] - datetime) / np.timedelta64(1, 'h')) <= 3).any()
    return 1.0 if value and not __isNaN(value) else 0.0

def calc_region_alarms_history(region_name: str, states_list: list, datetime_now):
    # Get region id as in alarms api
    region_id = next((state for state in states_list if region_name in state['regionName']), None)['regionId']
    region_id = int(region_id)
    if region_id is None:
        return None
    # Get timezone and current time
    tz = pytz.timezone('Europe/Kyiv')
    now = tz.localize(datetime_now)

    region_history = __load_history(region_id)
    hours_from_last_alarm = 0.0
    alarm_count = 0.0

    # Calculate number of distinct alarms in past 24 hours
    for alarm in reversed(region_history):
        if  alarm['isContinue']:
            alarm_count += 1
        else:
            alarm_end_date = tz.localize(parser.parse(alarm['endDate']))
            if alarm_end_date > now:
                continue
            if  alarm['isContinue'] or (now - alarm_end_date) < dt.timedelta(hours=24):
                alarm_count += 1
            else:
                break
    # Calculate hours from last alarm
    last_air_alert = region_history[-1]
    if not last_air_alert['isContinue']:
        hours_from_last_alarm = (now - tz.localize(parser.parse(last_air_alert['endDate']))).total_seconds() / 3600

    return pd.Series([alarm_count, hours_from_last_alarm])

# Returns number of alarms active now
def calc_simultaneous_alarms():
    return float(len(load_alerts()))

def generate_features_dumb(df):
    # Num of separate alarms in past 24 hours

    # Load state regions metadata from alerts API
    states = load_states()
    # Clear cache if any
    reset_cache()
    df[['event_alarms_past_24', 'event_hours_from_last_alarm']] = df.apply(lambda row: calc_region_alarms_history(row['region'], states, row['date_time']), axis=1)
    # Num of state regions with alarms at the moment
    df['event_simultaneous_alarms'] = calc_simultaneous_alarms()
    return df
