ALERTS_API_KEY = ""
WEATHER_API_KEY = ""

MY_API_TOKEN=""

forecast_hours = 12


STATE_FILE = "./data/state.json"
DATASET_FILE = "./data/4_all_data_preprocessed/all_features"
PREDICTIONS_FILE = "./data/alarms_forecast.csv"
HOLIDAY_DATASET = './data/1_holidays/holidays.csv'
REGIONS_DATASET = './data/0_meta/regions.csv'

MODEL_FOLDER = "model"

model_file_name = "6__XGBoost__v1"

tfidf_transformer_model = "tfidf_transformer"
count_vectorizer_model = "count_vectorizer"
conditions_encoder_model = "conditions_encoder"
preciptype_encoder_model = "preciptype_encoder"
scaler_model = "scaler"

tfidf_transformer_version = "v4"
count_vectorizer_version = "v4"
conditions_encoder_version = "v1"
preciptype_encoder_verssion = "v1"
scaler_version = "v1"