"""
This module provides methods for data preprocessing, including

- get_report_lemm(report_file_path)

- get_report_tfidf_vector(lemm)
"""

import re
import pickle

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from num2words import num2words

from bs4 import BeautifulSoup

MAX_DF = 0.98
MIN_DF = 0.25

REPORTS_FOLDER = "Reports"
REPORTS_DATA_FOLDER = "data/2_isw_preprocessed"
REPORTS_DATA_FILE = "all_days.csv"

TFIDF_NUMBER = 100

EVENTS_DATA_FOLDER = "data/1_events"
EVENTS_DATA_FILE = "alarms.csv"

WEATHER_DATA_FOLDER = "data/1_weather"
WEATHER_DATA_FILE = "all_weather_by_hour_v2.csv"

REGIONS_DATA_FOLDER = "data/0_meta"
REGIONS_DATA_FILE = "regions.csv"

MODEL_FOLDER = "model"

OUTPUT_FOLDER = "data/4_all_data_preprocessed"
ISW_OUTPUT_DATA_FILE = "all_isw.csv"
WEATHER_EVENTS_OUTPUT_DATA_FILE = "all_hourly_weather_events_v2.csv"
WEATHER_EVENTS_KEYWORDS_OUTPUT_DATA_FILE = "all_hourly_weather_events_isw_v2.csv"

tfidf_transformer_model = "tfidf_transformer"
count_vectorizer_model = "count_vectorizer"

tfidf_transformer_version = "v4"
count_vectorizer_version = "v4"

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

def __create_bag_of_words(text):
    """
    Creates a bag of words (a dictionary of word frequencies) from a string.
    """
    # Split the text into words
    words = text.split()

    # Initialize an empty dictionary
    bag_of_words = {}

    # Loop over each word and count its frequency
    for word in words:
        if word in bag_of_words:
            bag_of_words[word] += 1
        else:
            bag_of_words[word] = 1

    return bag_of_words

def __calculate_term_frequency(bag_of_words):

    # Calculate the total number of words in the bag
    total_words = sum(bag_of_words.values())

    # Initialize an empty dictionary
    term_frequency = {}

    # Loop over each word in the bag and calculate its term frequency
    for word, frequency in bag_of_words.items():
        term_frequency[word] = frequency / total_words

    return term_frequency

def get_report_lemm(report_file_path):
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

def get_report_tfidf_vector(lemm):
    # Load TF-IDF
    tfidf = pickle.load(open(f"{MODEL_FOLDER}/{tfidf_transformer_model}_{tfidf_transformer_version}.pkl", "rb"))
    cv = pickle.load(open(f"{MODEL_FOLDER}/{count_vectorizer_model}_{count_vectorizer_version}.pkl", "rb"))

    word_count_vector = cv.transform([lemm])
    tfidf_vector = tfidf.transform(word_count_vector)

    # transform tfidf matrix to sparse dataframe
    df_tfidf_vector = pd.DataFrame.sparse.from_spmatrix(tfidf_vector, columns=cv.get_feature_names_out())

    return df_tfidf_vector
