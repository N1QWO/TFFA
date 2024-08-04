import os

# Базовый путь к проекту
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Пути к другим часто используемым директориям в проекте
DATA_COLLECTION = os.path.join(BASE_DIR, 'data_collection')
DATA_TRAIN = os.path.join(BASE_DIR, 'data_train')
DATA_REAL = os.path.join(BASE_DIR, 'data_real')
DATA_PREPARATION = os.path.join(DATA_TRAIN, 'data_preparation')
LEARN_MODEL = os.path.join(BASE_DIR, 'learn_model')
MODEL_REPORT = os.path.join(LEARN_MODEL, 'model_report')
MODEL_WEIGHT = os.path.join(LEARN_MODEL, 'model_weights')
MARKET_SIGNAL = os.path.join(BASE_DIR, 'market_signal')
MODEL_WEIGHT_ELIT = os.path.join(MARKET_SIGNAL, 'model_weights')
PREPARATION_DATA = os.path.join(BASE_DIR, 'preparation_data')

# Ваш токен и chat_id
TOKEN = ''
CHAT_ID = ''
URL = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
