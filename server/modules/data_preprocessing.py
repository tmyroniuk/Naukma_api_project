"""
This module provides methods used for data preprocessing
"""

import re
import pickle
import pytz

import datetime as dt
import pandas as pd
import numpy as np

from dateutil import parser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from num2words import num2words

from bs4 import BeautifulSoup

from config import *
from modules.data_collection import load_weather, load_weather_at

def __remove_one_letter_word(data):
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if len(w) > 1:
            new_text = new_text + ' ' + w
    return new_text

def __convert_lower_case(data):
      return np.char.lower(data)

def __remove_stop_words(data):
    stop_words = set(stopwords.words('english'))
    stop_stop_words = {"no","not"}
    stop_words = stop_words - stop_stop_words

    words = word_tokenize(str(data))

    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text +" "+ w
    return new_text

def __remove_punctuation(data):
    symbols = "!\"#$%&()*+—./:;<=>7@[\]^_'{|}~\n"

    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")

    data = np.char.replace(data, ',', "")

    return data

def __remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def __remove_map_line(data):
    return np.char.replace(data, "Click here to expand the map below.", "")

def __remove_map_line_v2(data):
    return np.char.replace(data, "Click here to see ISW’s interactive map of the Russian invasion of Ukraine. This map is updated daily alongside the static maps present in this report.", "")

def __convert_numbers(data):

    tokens = word_tokenize(str(data))
    new_text = " "
    for w in tokens:
        if w.isdigit():
            if int(w)<1000000000000:
                w = num2words (w)
            else:
                w = ''
        new_text = new_text +" " + w
    new_text = np.char.replace(new_text, "-", " ")

    return new_text

def __stemming(data):
    stemmer= PorterStemmer()

    tokens = word_tokenize(str(data))

    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def __lemmatizing(data):
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + lemmatizer.lemmatize(w)
    return new_text

def __isNaN(num):
    return num != num

def __preprocess(data, word_root_algo="lemm"):
    data = __remove_map_line(data)
    data = __remove_map_line_v2(data)
    data = __remove_one_letter_word(data)
    data = __convert_lower_case(data)
    data = __remove_punctuation(data) #remove comma seperately
    data = __remove_apostrophe (data)
    data = __remove_stop_words(data)
    data = __convert_numbers(data)
    data = __stemming(data)
    data = __remove_punctuation(data)
    data = __convert_numbers (data)

    if word_root_algo == "lemm":
        print ("lennatizing")
        data = __lemmatizing(data) #needed again as we need to lemmatize the words
    else:
        print("stemming")
        data = __stemming(data) #needed again as we need to stem the words

    data = __remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = __remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one

    return data

def __get_report_lemm(report_file_path):
    lemm = ''
    # Open the HTML file
    with open(report_file_path, encoding="utf8") as file:
        soup = BeautifulSoup(file, 'html.parser')

        # Extract the text from the HTML
        text = soup.get_text()

        # Find the index of the first occurrence of "ET"
        index = text.find("ET")

        # Extract the text after the first occurrence of "ET"
        text = text[2+index:]
        index = text.rfind("[1]")
        text =text[:index]
        text = re.sub(r'\[.*?\]', '', text)

        lemm = __preprocess(text)

    return lemm

def __is_date_suitable(datetime: str, tzoffset: float, time_span: int):
    # Checks if given date is withing timespan from now using tz hour offset value
    utc_datetime = pytz.utc.localize(parser.parse(datetime))
    now = dt.datetime.now(pytz.utc) + dt.timedelta(hours=tzoffset)
    return now < utc_datetime < now + dt.timedelta(hours=time_span + 1)

def prepare_weather_data(day: dict, hour: dict):
    # Creates N hours forecast from forecast API responce for given region
    hour_data = {}

    # Write nessesary data into the responce
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
    try:
        hour_data["hour_severerisk"] = hour["severerisk"]
    except:
        hour_data["hour_severerisk"] = None
    hour_data["hour_conditions"] = hour["conditions"]

    return hour_data

def __prepare_if_sutable(weather_json, time_span: int):
    # Creates N hours forecast from forecast API responce for given region
    forecast = []
    for day in weather_json["days"]:
        for hour in day["hours"]:
            hour_data = {}

            # Check if date is within timeSpan
            datetime = f"{day['datetime']} {hour['datetime']}"
            if __is_date_suitable(datetime, weather_json['tzoffset'], time_span):
                # Parse data and append
                forecast.append(prepare_weather_data(day, hour))

    return forecast

def get_weather_forecast_df(time_span: int):
    # Creates dataframe with forecast data for given number of hours
    df = []
    # Read regions DF
    df_regions = pd.read_csv(REGIONS_DATASET, sep=',')

    for index, row in df_regions.iterrows():
        # Load forecast for each region
        try:
            weather = load_weather(row['center_city_en'])
        except:
            weather = load_weather(row['center_city_en'] + '(UA)')

        # Parse forecast for next time_span hours
        weather = __prepare_if_sutable(weather, time_span)
        df_city = pd.DataFrame(weather)

        # Add nessesary metadata
        df_city['region_id'] = float(row['region_id'])
        df_city['region'] = row['region']

        df.append(df_city)
    # Unite all loadl DFs in one DF
    return pd.concat(df, axis=0, ignore_index=True)

def get_weather_log_df(date_time: dt.datetime):
    # Creates dataframe with forecast data for given number of hours
    df = []
    # Read regions DF
    df_regions = pd.read_csv(REGIONS_DATASET, sep=',')
    for index, row in df_regions.iterrows():
        # Load forecast for each region
        try:
            weather = load_weather_at(row['center_city_en'], date_time)
        except:
            weather = load_weather_at(row['center_city_en'] + '(UA)', date_time)

        # Parse forecast for next time_span hours
        weather = prepare_weather_data(weather['days'][0], weather['currentConditions'])
        df_city = pd.DataFrame([weather])

        # Add nessesary metadata
        df_city['region_id'] = float(row['region_id'])
        df_city['region'] = row['region']

        df.append(df_city)
    # Unite all loadl DFs in one DF
    return pd.concat(df, axis=0, ignore_index=True)

def get_report_tfidf_vector(report_file_path):
    # Creates TF-IDF vector for given html of ISW report
    lemm = __get_report_lemm(report_file_path)
    # Load TF-IDF
    tfidf = pickle.load(open(f"{MODEL_FOLDER}/{tfidf_transformer_model}_{tfidf_transformer_version}.pkl", "rb"))
    cv = pickle.load(open(f"{MODEL_FOLDER}/{count_vectorizer_model}_{count_vectorizer_version}.pkl", "rb"))
    # Calculate TD-IDF matrix
    word_count_vector = cv.transform([lemm])
    tfidf_vector = tfidf.transform(word_count_vector)

    # transform tfidf matrix to sparse dataframe
    df_tfidf_vector = pd.DataFrame.sparse.from_spmatrix(tfidf_vector, columns=cv.get_feature_names_out())

    return df_tfidf_vector