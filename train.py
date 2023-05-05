import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from datetime import timedelta

from sklearn.model_selection import TimeSeriesSplit


PROCESSED_DATA_FOLDER = "data/4_all_data_preprocessed"
DATASET_FILE = "all_features"

# Load dataset
dataset = pickle.load(open(f"{PROCESSED_DATA_FOLDER}/{DATASET_FILE}.pkl", "rb"))

# Sort dataset by date
dataset['day_datetime'] = pd.to_datetime(dataset['day_datetime'])
dataset = dataset.sort_values(by='day_datetime')


# Extract relevant date features
dataset['year'] = dataset['day_datetime'].dt.year
dataset['month'] = dataset['day_datetime'].dt.month
dataset['day'] = dataset['day_datetime'].dt.day
dataset['day_of_week'] = dataset['day_datetime'].dt.dayofweek

dataset['season'] = (dataset['day_datetime'].dt.month % 12 // 3).replace({0: 'winter', 1: 'spring', 2: 'summer', 3: 'fall'})

dataset = pd.get_dummies(dataset, columns=['season'])

# Translate city names
dataset.rename(columns = {'city_resolvedAddress':'city'}, inplace=True)


region_dict = {
    "Київ, Україна" : 'Kyiv', 
    "Миколаїв, Україна" : 'Mykolaiv',
    "Дніпро, Україна": 'Dnipro',
    "Харків, Україна" : 'Kharkiv', 
    "Житомир, Україна": 'Zhytomyr',
    "Кропивницький, Україна": 'Kropyvnytskyi',
    "Запоріжжя, Україна": 'Zaporizhzhia',
    "Полтава, Україна": 'Poltava',
    "Чернігів, Україна": 'Chernihiv',
    "Одеса, Україна": 'Odesa',
    "Хмельницька область, Україна": 'Khmelnytskyi',
    "Черкаси, Україна": 'Cherkasy',
    "Суми, Україна": 'Sumy',
    "Вінниця, Україна": 'Vinnytsia',
    "Херсон, Україна": 'Kherson',
    "Львів, Україна": 'Lviv',
    "Луцьк, Луцький район, Україна": 'Lutsk',
    "Рівне, Україна": 'Rivne',
    "Івано-Франківськ, Україна": 'Ivano-Frankivsk',
    "Тернопіль, Україна": 'Ternopil',
    "Чернівці, Україна": 'Chernivtsi',
    "Ужгород, Ужгородський район, Україна": 'Uzhhorod',
    "Донецьк, Україна": 'Donetsk'
}

dataset = dataset.replace({"city": region_dict})

# Define target variable
dataset['target'] = dataset['event_indicator']
dataset = dataset.drop(columns=['event_indicator', 'isw_date_tomorrow_datetime'])

# Convert target to int
dataset['target'] = dataset['target'].astype(int)

# Create id column for each unique city
cities = dataset['city'].unique()
city_dict = {cities[i]: i+1 for i in range(len(cities))}
dataset['city_id'] = dataset['city'].map(city_dict)
dataset.drop('city', axis=1, inplace=True)

# Split the dataset into features (X) and target (y)
min_date = dataset['day_datetime'].min()
max_date = dataset['day_datetime'].max()
X = dataset.drop(columns=['target'])
y = dataset['target']

train_percent = .75
time_between = max_date - min_date
train_cutoff = min_date + train_percent*time_between


# XGBoost
from xgboost import XGBClassifier
xgb_clf = XGBClassifier(min_child_weight=1,
                           max_depth=12,
                           learning_rate=0.2,
                           gamma=0.3,
                           colsample_bytree=0.5,
                           objective='binary:logistic',
                           random_state=5, 
                           verbosity=3)

xgb_clf.fit(X_train, y_train)

predict_proba = xgb_clf.predict_proba(X_test)[:, 1]