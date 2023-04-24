import datetime
import pandas as pd
import numpy as np

import weather_collection
import data_collection
import pickle
from sklearn.preprocessing import StandardScaler


weather = weather_collection.get_weather('Kyiv')
df = pd.DataFrame(weather["forecast"])

# Convet time to float
df.day_sunset = df.day_sunset.apply(lambda x:
    (datetime.datetime.strptime(x, "%H:%M:%S") - datetime.datetime.strptime("00:00:00", "%H:%M:%S")).total_seconds()
)
df.day_sunrise = df.day_sunrise.apply(lambda x:
    (datetime.datetime.strptime(x, "%H:%M:%S") - datetime.datetime.strptime("00:00:00", "%H:%M:%S")).total_seconds()
)
df.hour_datetime = df.hour_datetime.apply(lambda x:
    (datetime.datetime.strptime(x, "%H:%M:%S") - datetime.datetime.strptime("00:00:00", "%H:%M:%S")).total_seconds()//3600
)

# Encode categorical values
df['hour_preciptype'].fillna(value=np.nan, inplace=True)
le = pickle.load(open('./model/preciptype_encoder_v1.pkl', 'rb'))
df['hour_preciptype'] = le.transform(df['hour_preciptype']).astype(float)

le = le = pickle.load(open('./model/conditions_encoder_v1.pkl', 'rb'))
df['hour_conditions'] = le.transform(df['hour_conditions']).astype(float)

# Fillna
df.fillna(0, inplace=True)

# Scale
scaler = pickle.load(open('model/scaler_v1.pkl', 'rb'))
df = scaler.transform(df[scaler.get_feature_names_out()])

print(df.shape)
print(df.dtypes)

