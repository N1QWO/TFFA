import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
from DataPreparation import DataPreparation

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:  
    from config.settings import DATA_TRAIN
except ImportError as e:
    print("ImportError:", e)

# Функция для поиска "Головы и Плеч"
def head_and_shoulders(df):
    shoulders_tolerance = 0.03  # Допустимое отклонение в процентах
    
    highs = df['high']
    lows = df['low']
    highs = highs.reset_index( drop=True)
    lows = lows.reset_index(drop=True)
    # print(highs)
    shoulders_positions = []
    row,col = df.shape
    for i in range(2, row - 2):
        # Проверка условий для плеч и головы
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            if highs[i-1] < highs[i-2] and highs[i+1] < highs[i+2]:
                left_shoulder = highs[i-2]
                head = highs[i]
                right_shoulder = highs[i+2]
                
                left_valley = lows[i-1]
                right_valley = lows[i+1]
                
                # Проверка уровней плеч и впадин
                if abs(left_shoulder - right_shoulder) / head < shoulders_tolerance and abs(left_valley - right_valley) / head < shoulders_tolerance:
                    shoulders_positions.append(i)
    # print(shoulders_positions)
    df['Head_and_Shoulders'] = 0
    df.loc[df.index[shoulders_positions], 'Head_and_Shoulders'] = 1
    
    return df

# Функция для поиска паттерна "Двойная вершина"
def double_top(df):
    peak = df['close'].rolling(window=10).max()
    # print(peak[1000:1010])
    dt_pattern = (df['close'] > peak.shift(1)) & (df['close'] > peak.shift(-1))
    df['Double_Top'] = dt_pattern.astype(int)
    return df

# Функция для поиска паттерна "Флаг"
def flag_pattern(df):
    df['Flag'] = ((df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))).astype(int)
    return df

# Функция для поиска паттерна "Двойное дно"
def double_bottom(df):
    trough = df['close'].rolling(window=10).min()
    db_pattern = (df['close'] < trough.shift(1)) & (df['close'] < trough.shift(-1))
    df['Double_Bottom'] = db_pattern.astype(int)
    return df

def bullish_flag(df):
    bf_pattern = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2)) & (df['close'].shift(2) > df['close'].shift(3))
    df['Bullish_Flag'] = bf_pattern.astype(int)
    return df
def ascending_triangle(df):
    # Находим вершины треугольника (высокие точки)
    peak1 = df['close'].rolling(window=10).max()
    peak2 = df['close'].rolling(window=10).max().shift(-10)
    peak3 = df['close'].rolling(window=10).max().shift(-20)
    
    # Находим дно треугольника (низкие точки)
    trough1 = df['close'].rolling(window=10).min()
    trough2 = df['close'].rolling(window=10).min().shift(-10)
    trough3 = df['close'].rolling(window=10).min().shift(-20)
    
    # Условия для восходящего треугольника
    ascending_triangle_pattern = (df['close'] > trough1.shift(1)) & \
                                 (df['close'] > trough2.shift(1)) & \
                                 (df['close'] > trough3.shift(1)) & \
                                 (df['close'] <= peak1.shift(1)) & \
                                 (df['close'] <= peak2.shift(1)) & \
                                 (df['close'] <= peak3.shift(1))
    
    df['Ascending_Triangle'] = ascending_triangle_pattern.astype(int)
    return df
def falling_wedge(df):
    peak1 = df['close'].rolling(window=10).max()
    peak2 = df['close'].rolling(window=10).max().shift(-10)
    peak3 = df['close'].rolling(window=10).max().shift(-20)
    
    trough1 = df['close'].rolling(window=10).min()
    trough2 = df['close'].rolling(window=10).min().shift(-10)
    trough3 = df['close'].rolling(window=10).min().shift(-20)
    
    falling_wedge_pattern = (df['close'] < peak1.shift(1)) & \
                            (df['close'] < peak2.shift(1)) & \
                            (df['close'] < peak3.shift(1)) & \
                            (df['close'] > trough1.shift(1)) & \
                            (df['close'] > trough2.shift(1)) & \
                            (df['close'] > trough3.shift(1))
    
    df['Falling_Wedge'] = falling_wedge_pattern.astype(int)
    return df

# Функция для определения паттерна "Молот" (Hammer)
def add_candlestick_patterns(df):
    def is_hammer(row):
        body = abs(row['close'] - row['open'])
        lower_shadow = min(row['open'], row['close']) - row['low']
        upper_shadow = row['high'] - max(row['open'], row['close'])
        hammer = lower_shadow > 2 * body and upper_shadow < body*0.5
        # print(lower_shadow > 2 * body,upper_shadow,body)
        return hammer

    # Функция для определения паттерна "Повешенный" (Hanging Man)
    def is_hanging_man(row):
        return is_hammer(row) and row['close'] < row['open']
    
        # Добавление столбцов с паттернами
    df['Hammer'] = df.apply(is_hammer, axis=1).astype(int)
    df['HangingMan'] = df.apply(is_hanging_man, axis=1).astype(int)

    return df

# Пример загрузки данных
dp = DataPreparation()

coin = "BTCUSD"
file_path = os.path.join(DATA_TRAIN, f'{coin}.npy')
df = dp.open_file_data(file_path)
df = pd.DataFrame(df, columns=['open', 'high', 'low', 'close', 'volume', 'turnover', 'average_price', 
                               'SMA_3', 'SMA_7', 'SMA_28', 'std_dev_7', 'Upper_Band', 'Lower_Band', 
                               'RSI_7', 'RSI_14', 'EMA_5', 'EMA_20', 'MACD', 'Signal_line', 
                               'high_low', 'high_prev_close', 'low_prev_close', 'TR', 'SMA_TR', 
                               'VWAP', 'part_h_max', 'part_m_max', 'part_y_max', 'part_all_max'])

# Предобработка данных (удаление пропусков, вычисление вспомогательных параметров)
df = df.dropna()  # Удаление строк с пропущенными значениями

# Предобработка данных
df['date'] = pd.to_datetime(df.index)
df.set_index('date', inplace=True)

# Применение функций для поиска паттернов
df = head_and_shoulders(df)
df = double_top(df)
df = flag_pattern(df)
df = double_bottom(df)
df = bullish_flag(df)
df = ascending_triangle(df)
df = falling_wedge(df)
df = add_candlestick_patterns(df)

# print(df[1000:1010])
# # Вывод найденных паттернов
print(df[['Hammer']].sum())
print(df[['Head_and_Shoulders']].sum())
print(df[['Double_Top']].sum())
print(df[['Flag']].sum())
print(df[['Double_Bottom']].sum())
print(df[['Bullish_Flag']].sum())
print(df[['Ascending_Triangle']].sum())
print(df[['HangingMan']].sum())
print(df[['Falling_Wedge']].sum())
# Визуализация найденных паттернов
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close'], label='Close Price')

# Добавляем паттерны на график
plt.scatter(df.index[df['Head_and_Shoulders']==1], df['close'][df['Head_and_Shoulders']==1], marker='^', color='r', label='Head_and_Shoulders', s=100)

# plt.scatter(df.index[df['Double_Top']==1], df['close'][df['Double_Top']==1], marker='x', color='g', label='Double Top', s=100)
# plt.scatter(df.index[df['Flag'] == 1], df['close'][df['Flag'] == 1], marker='s', color='b', label='Flag', s=100)
# plt.scatter(df.index[df['Double_Bottom'] == 1], df['close'][df['Double_Bottom'] == 1], marker='d', color='m', label='Double Bottom', s=100)
# plt.scatter(df.index[df['Bullish_Flag'] == 1], df['close'][df['Bullish_Flag'] == 1], marker='o', color='r', label='Bullish_Flag', s=100)
# plt.scatter(df.index[df['Ascending_Triangle'] == 1], 
#             df['close'][df['Ascending_Triangle'] == 1], 
# #             marker='^', color='orange', label='Ascending Triangle', s=100)
# plt.scatter(df.index[df['Hammer'] == 1], 
#             df['close'][df['Hammer'] == 1], 
#             marker='^', color='g', label='Hammer', s=100)
# plt.scatter(df.index[df['HangingMan'] == 1], 
#             df['close'][df['HangingMan'] == 1], 
#             marker='v', color='r', label='Hanging Man', s=100)
# plt.scatter(df.index[df['Falling_Wedge']], df['close'][df['Falling_Wedge']], marker='h', color='purple', label='Falling Wedge', s=100)

plt.legend()
plt.show()
