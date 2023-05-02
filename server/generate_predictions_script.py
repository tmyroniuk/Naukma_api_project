import pickle
import sys

import datetime as dt
import pandas as pd
import numpy as np

from dateutil import parser

from config import *
from modules.data_collection import save_by_date, load_states
from modules.data_preprocessing import get_report_tfidf_vector, get_weather_forecast_df
from modules.feature_engineering import *

def isNaN(num):
    return num != num

# Generate predictions file for 12 hours in all regions

def get_yesterday_report(day_str):
    # Load yesterday ISW reports
    date = parser.parse(day_str) - dt.timedelta(days=1)
    while 'Error' in save_by_date(date):
        date -= dt.timedelta(days=1)
    tfidf_vector = get_report_tfidf_vector(f"./Reports/{date.strftime('%Y-%m-%d')}.html")
    return pd.concat([pd.DataFrame([day_str], columns=['date_tomorrow_datetime']), tfidf_vector], axis=1)

def get_prediction(df, model):
    # Generate predictions

    # Normalize
    scaler = pickle.load(open('model/scaler_v1.pkl', 'rb'))
    # Separate float values
    df_float_values = df[scaler.get_feature_names_out()]
    if df_float_values.to_numpy().ndim < 2:
        df_float_values = pd.DataFrame({1: df_float_values.sparse.to_dense()}).T
    # Scale values
    df_float_values_scaled = pd.DataFrame(scaler.transform(df_float_values), columns=df_float_values.columns)
    df_float_values_scaled['day_of_week'] = df['day_of_week']

    # Generate prediction
    prediction = model.predict(df_float_values_scaled)
    return prediction

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

def generate_features_smart(df, model):
    # Generates features based on model predictions for prevous days

    # Generate dumb features and prediction
    df = generate_features_dumb(df)
    df['alarm_marker'] = pd.Series(get_prediction(df.copy(), model))

    # Create new column to account for predicted alarms
    df['event_alarms_past_24_modifier'] = 0

    # Recalculate features
    hours = forecast_hours
    regions = len(df['region_id'].unique())

    # Go through each hour and update values
    for h in range(1, hours):
        marker_sum = 0.0

        # Go through each region
        for r in range(0, regions):
            df.loc[r * hours + h, 'event_alarms_past_24_modifier'] = df.loc[r * hours + h - 1, 'event_alarms_past_24_modifier'] + df.loc[r * hours + h - 1, 'alarm_marker']
            # Reset alarm cooldown timer if new alarm predicted
            df.loc[r * hours + h, 'event_hours_from_last_alarm'] = df.loc[r * hours + h - 1, 'event_hours_from_last_alarm'] + 1 if df.loc[r * hours + h - 1, 'alarm_marker'] == 0 else 0.0
            # Calculate num of predicted simultanious alarms
            marker_sum += df.loc[r * hours + h - 1, 'alarm_marker']

        # Iterate once more
        for r in range(0, regions):
            # Write calculated values
            df.loc[r * hours + h, 'event_simultaneous_alarms'] = marker_sum
            df.loc[r * hours + h, 'event_alarms_past_24'] += df.loc[r * hours + h, 'event_alarms_past_24_modifier']
            # Update prediction with updated features
            df.loc[r * hours + h, 'alarm_marker'] = get_prediction(df.iloc[r * hours + h].copy(), model)[0]

    df = df.drop('event_alarms_past_24_modifier', axis=1)

    return df

if __name__ == '__main__':
    # Load weather forecast for 12 hours
    df = get_weather_forecast_df(forecast_hours)

    # Custom made dataset with most "important" russian hollidays
    holiday_df = pd.read_csv(HOLIDAY_DATASET, sep=';')
    holiday_df['date'] = holiday_df['date'].apply(pd.to_datetime)
    holiday_df = holiday_df.sort_values(by=['date'])
    holiday_df = holiday_df.set_index('date')

    # Dedicated datetime column
    df['date_time'] = df.apply(lambda row: parser.parse(f"{row['day_datetime']}T{row['hour_datetime']}"), axis=1)
    # Additional region id
    df['region_id_int'] = df['region_id'].astype(int)
    # Metadata as features
    df['day_of_week'] = df['date_time'].dt.dayofweek
    # If within 3 days from holiday
    df['event_holiday_is_near'] = df.apply(lambda row: event_holiday_is_near(holiday_df, row), axis=1)

    # Convet time to float
    df['day_sunset'] = df['day_sunset'].apply(lambda x:
        (parser.parse(x) - dt.datetime.strptime("00:00:00", "%H:%M:%S")).total_seconds()
    )
    df['day_sunrise'] = df['day_sunrise'].apply(lambda x:
        (parser.parse(x) - dt.datetime.strptime("00:00:00", "%H:%M:%S")).total_seconds()
    )
    df['hour_datetime'] = df['hour_datetime'].apply(lambda x:
        (parser.parse(x) - dt.datetime.strptime("00:00:00", "%H:%M:%S")).total_seconds()//3600
    )

    # Encode categorical values
    df['hour_preciptype'] = df['hour_preciptype'].apply(lambda a: str(a) if a else np.nan)
    le = pickle.load(open(f'./{MODEL_FOLDER}/{preciptype_encoder_model}_{preciptype_encoder_verssion}.pkl', 'rb'))
    df['hour_preciptype'] = le.transform(df['hour_preciptype']).astype(float)

    le = pickle.load(open(f'./{MODEL_FOLDER}/{conditions_encoder_model}_{conditions_encoder_version}.pkl', 'rb'))
    df['hour_conditions'] = le.transform(df['hour_conditions']).astype(float)

    # Fillna
    df.fillna(0.0, inplace=True)

    # Parse TF-IDF
    df_tfidf = []
    for day in df['day_datetime'].unique():
        df_tfidf.append(get_yesterday_report(day))
    df_tfidf = pd.concat(df_tfidf, axis=0, ignore_index=True)

    # Merge weather events dataset with yesterday report tfidf matrix (takes 2m to execute)
    df = df.merge(df_tfidf.add_prefix("isw_"),
                                    how="left",
                                    left_on="day_datetime",
                                    right_on="isw_date_tomorrow_datetime")

    # Load Model
    model = pickle.load(open(f'./{MODEL_FOLDER}/{model_file_name}.pkl', 'rb'))
    # Generate features
    df = generate_features_smart(df, model)

    # Save prediction to .csv
    res = df[['date_time', 'region_id_int', 'alarm_marker']].copy()
    res.to_csv(PREDICTIONS_FILE, sep=';')
