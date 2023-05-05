import pickle
import pandas as pd
import pytz
import json
import datetime as dt

from sklearn.model_selection import TimeSeriesSplit

from config import DATASET_FILE, STATE_FILE, MODEL_FOLDER, model_file_name

tz = pytz.timezone('Europe/Kyiv')

# Load dataset
dataset = pickle.load(open(f"{DATASET_FILE}.pkl", "rb"))

# Sort dataset by date
dataset['day_datetime'] = pd.to_datetime(dataset['day_datetime'])
dataset = dataset.sort_values(by='day_datetime')


# Extract relevant date features
dataset['day_of_week'] = dataset['day_datetime'].dt.dayofweek

# dataset['season'] = (dataset['day_datetime'].dt.month % 12 // 3).replace({0: 'winter', 1: 'spring', 2: 'summer', 3: 'fall'})

# dataset = pd.get_dummies(dataset, columns=['season'])

# Remove city names
dataset.drop(columns=['city_resolvedAddress'], inplace=True)

# Define target variable
dataset['target'] = dataset['event_indicator']
dataset = dataset.drop(columns=['day_datetime', 'event_indicator', 'isw_date_tomorrow_datetime'])

# Convert target to int
dataset['target'] = dataset['target'].astype(int)

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
# Train
X = dataset.drop(columns=['target'])
y = dataset['target']

xgb_clf.fit(X, y)

#Save model
pickle.dump(xgb_clf, open(f"./{MODEL_FOLDER}/{model_file_name}.pkl", 'wb'))

# Write metadata
status = {}
try:
    with open(STATE_FILE, 'r') as handle:
        status = json.load(handle)
except:
    status = {}
with open(STATE_FILE, 'w') as handle:
    status["last_model_train_time"] = str(dt.datetime.now(tz))
    json.dump(status, handle)