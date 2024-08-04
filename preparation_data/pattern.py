import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from DataPreparation import DataPreparation

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:  
    from config.settings import DATA_TRAIN
except ImportError as e:
    print("ImportError:", e)
# Пример загрузки данных
coin = "BTCUSD"
file_path = os.path.join(DATA_TRAIN,f'{coin}.npy')

dp = DataPreparation()
df = dp.open_file_data(file_path)
df = pd.DataFrame(df, columns=['open', 'high', 'low', 'close', 'volume', 'turnover', 'average_price', 
                               'SMA_3', 'SMA_7', 'SMA_28', 'std_dev_7', 'Upper_Band', 'Lower_Band', 
                               'RSI_7', 'RSI_14', 'EMA_5', 'EMA_20', 'MACD', 'Signal_line', 
                               'high_low', 'high_prev_close', 'low_prev_close', 'TR', 'SMA_TR', 
                               'VWAP', 'part_h_max', 'part_m_max', 'part_y_max', 'part_all_max'])

# Предобработка данных (удаление пропусков, вычисление вспомогательных параметров)
df = df.dropna()  # Удаление строк с пропущенными значениями

# Пример создания меток (наличие/отсутствие паттерна)
df['Pattern'] = np.where(df['close'] > df['close'].shift(-1), 1, 0)
df = df.dropna()  # Удаление строк с NaN, которые могут появиться после shift

# Выбор признаков и целевой переменной
X = df[['open', 'high', 'low', 'close', 'volume']]
y = df['Pattern']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели (например, случайный лес)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Оценка модели
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

# Прогнозирование на новых данных для поиска паттернов
# new_data = pd.read_csv('new_data_to_predict.csv')
# Предобработка новых данных

# predictions = model.predict(new_data[['Open', 'High', 'Low', 'Close']])
# print(predictions)
