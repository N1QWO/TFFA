import numpy as np
import pandas as pd
import pytz
from scipy.signal import argrelextrema

class DataProcessor:
    """
    
    calculation of indicators :
    sma,emas,rsi,tr,vwap,std,piece_of_max

    Return:
    Datatframe with indicate 

    """
    def __init__(self,start_ind = 0,refill = False):
        self.refill=refill
        
        self.short_window_sma = 3
        self.long_window_sma = 7
        self.long_long_window_sma = 28

        self.short_window_rsi = 7
        self.long_window_rsi = 14

        self.short_window_ema = 5
        self.long_window_ema = 20

        self.short_window_macd = self.short_window_ema
        self.long_window_macd = self.long_window_ema
        self.signal_window = 9

        self.window_VWAP = 3

        self.start_ind = start_ind
    
    def forward_data(self, df):
        if self.start_ind==0:
            df = self.process_dataframe(df)
        df = self.calculate_smas(df)
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_rsi(df)
        df = self.calculate_emas(df)
        df = self.calculate_macd(df)
        df = self.calculate_tr(df)
        df = self.calculate_vwap(df)
        df = self.calculate_piece_of(df, h = 24, name = 'part_h_max')  
        df = self.calculate_piece_of(df, h = 24*31, name = 'part_m_max')  
        df = self.calculate_piece_of(df, h = 24*31*12, name = 'part_y_max')  
        df = self.calculate_piece_of(df, h = len(df), name = 'part_all_max')  
        # print(df.columns.to_list())
        return df
    
    # def refill_data(self,df):
        

    def process_dataframe(self, df):
        ind = df[:,0]
        df = pd.DataFrame(df,columns=['timestamp','open', 'high', 'low', 'close', 'volume', 'turnover'])
        
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        # Преобразование 'timestamp' в datetime с часовым поясом UTC
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC')

        # Локализация времени в часовой пояс Europe/Moscow
        local_tz = pytz.timezone('Europe/Moscow')
        df['timestamp'] = df['timestamp'].dt.tz_convert(local_tz)

        df.set_index('timestamp', inplace=True)
        
        df = df.astype(float)

        return df
    
    def calculate_smas(self, df):
        df['average_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df.loc[df.index[self.start_ind]:,f'SMA_{self.short_window_sma}'] = df['close'].rolling(window=self.short_window_sma).mean()
        df.loc[df.index[self.start_ind]:,f'SMA_{self.long_window_sma}'] = df['close'].rolling(window=self.long_window_sma).mean()
        df.loc[df.index[self.start_ind]:,f'SMA_{self.long_long_window_sma}'] = df['close'].rolling(window=self.long_long_window_sma).mean()
        
        return df
    
    def calculate_rsi(self, df):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.short_window_rsi).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.short_window_rsi).mean()
        rs = gain / loss
        df.loc[df.index[self.start_ind]:,f'RSI_{self.short_window_rsi}'] = 100 - (100 / (1 + rs))
        
        gain = (delta.where(delta > 0, 0)).rolling(window=self.long_window_rsi).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.long_window_rsi).mean()
        rs = gain / loss
        df.loc[df.index[self.start_ind]:,f'RSI_{self.long_window_rsi}'] = 100 - (100 / (1 + rs))
        
        return df
    
    def calculate_emas(self, df):
        df.loc[df.index[self.start_ind]:,f'EMA_{self.short_window_ema}'] = df['close'].ewm(span=self.short_window_ema, adjust=False).mean()
        df.loc[df.index[self.start_ind]:,f'EMA_{self.long_window_ema}'] = df['close'].ewm(span=self.long_window_ema, adjust=False).mean()
        return df
    
    def calculate_macd(self, df):
        df.loc[df.index[self.start_ind]:,'MACD'] = df[f'EMA_{self.long_window_macd}'] - df[f'EMA_{self.short_window_macd}']
        df.loc[df.index[self.start_ind]:,'Signal_line'] = df['MACD'].ewm(span=self.signal_window, adjust=False).mean()
        return df
    
    def calculate_tr(self, df):
        window_tr = 7
        df['previous_close'] = df['close'].shift(1)
        df['high_low'] = df['high'] - df['low']
        df['high_prev_close'] = np.abs(df['high'] - df['previous_close'])
        df['low_prev_close'] = np.abs(df['low'] - df['previous_close'])
        df['TR'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
        df.loc[df.index[self.start_ind]:,'SMA_TR'] = df['TR'].rolling(window=window_tr).mean()

        df = df.drop(columns=['previous_close'])
        return df
    
    def calculate_vwap(self, df):
        df.loc[df.index[self.start_ind]:,'VWAP'] = (df['close'] * df['volume']).rolling(window=self.window_VWAP).sum() / df['volume'].rolling(window=self.window_VWAP).sum()
        return df
    
    def calculate_piece_of(self, df,h,name):
        df.loc[df.index[self.start_ind]:,name] = df['close'] / df['high'].rolling(window=h, min_periods=1).max()
        return df

    def calculate_bollinger_bands(self, df):
        df.loc[df.index[self.start_ind]:,f'std_dev_{self.long_window_sma}'] = df['close'].rolling(window=self.long_window_sma).std()
        df.loc[df.index[self.start_ind]:,'Upper_Band'] = df[f'SMA_{self.long_window_sma}'] + (df[f'std_dev_{self.long_window_sma}'] * 2)
        df.loc[df.index[self.start_ind]:,'Lower_Band'] = df[f'SMA_{self.long_window_sma}'] - (df[f'std_dev_{self.long_window_sma}'] * 2)
        return df


class DataPattern:
    """
    find of pattern

    get_pattern: main func with find of pattern ->

    head_and_shoulders
    flag_pattern
    bullish_flag
    ascending_triangle
    falling_wedge
    Hammer
    HangingMan

    Return:
    df with bool values of pattern
    """
    def __init__(self,start_ind = 0,shoulders_tolerance=0.03):
        self.shoulders_tolerance = shoulders_tolerance
        self.start_ind = start_ind
    
    def get_pattern(self,df):
         
        df = self.head_and_shoulders(df)
        df = self.flag_pattern(df)
        df = self.bullish_flag(df)
        df = self.ascending_triangle(df)
        df = self.falling_wedge(df)
        df = self.add_candlestick_patterns(df)
        return df
    
    def head_and_shoulders(self,df):
        
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
                    if abs(left_shoulder - right_shoulder) / head < self.shoulders_tolerance and abs(left_valley - right_valley) / head < self.shoulders_tolerance:
                        shoulders_positions.append(i)
        # print(shoulders_positions)
        df['Head_and_Shoulders'] = 0
        df.loc[df.index[shoulders_positions], 'Head_and_Shoulders'] = 1
        
        return df
    
    # Функция для поиска паттерна "Флаг"
    def flag_pattern(self,df):
        df.loc[df.index[self.start_ind]:,'Flag'] = ((df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))).astype(int)
        return df
    
    def bullish_flag(self,df):
        bf_pattern = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2)) & (df['close'].shift(2) > df['close'].shift(3))
        df.loc[df.index[self.start_ind]:,'Bullish_Flag'] = bf_pattern.astype(int)
        return df
    
    def ascending_triangle(self,df):
        window = 10

        # Функция для нахождения локальных экстремумов
        def find_local_extrema(data, order=window//2):
            local_max = argrelextrema(data.values, np.greater, order=order)[0]
            local_min = argrelextrema(data.values, np.less, order=order)[0]
            return local_max, local_min
        # Нахождение локальных экстремумов
        local_max, local_min = find_local_extrema(df['close'])


        df['local_max'] = np.nan
        df['local_min'] = np.nan

        df.loc[df.index[local_max], 'local_max'] = df['close'].iloc[local_max]
        df.loc[df.index[local_min], 'local_min'] = df['close'].iloc[local_min]

        ascending_triangle_pattern  = np.zeros(len(df), dtype=int)
        for i in range(2*window,len(df)):
            recent_max = df['local_max'].iloc[i-2*window:i].dropna()
            recent_min = df['local_min'].iloc[i-2*window:i].dropna()

            if len(recent_max) >= 2 and len(recent_min) >= 2:
                if np.all(np.isclose(recent_max, recent_max.iloc[0], rtol=0.01)) and \
                np.all(np.diff(recent_min) > 0):
                    ascending_triangle_pattern[i] = 1

        df.loc[df.index[self.start_ind]:,'Ascending_Triangle'] = ascending_triangle_pattern

        df = df.drop(columns=['local_max'])
        df = df.drop(columns=['local_min'])

        return df
    
    def falling_wedge(self,df):
        # peak1 = df['close'].rolling(window=10).max()
        # peak2 = df['close'].rolling(window=10).max().shift(-10)
        # peak3 = df['close'].rolling(window=10).max().shift(-20)
        
        # trough1 = df['close'].rolling(window=10).min()
        # trough2 = df['close'].rolling(window=10).min().shift(-10)
        # trough3 = df['close'].rolling(window=10).min().shift(-20)
        
        # falling_wedge_pattern = (df['close'] < peak1.shift(1)) & \
        #                         (df['close'] < peak2.shift(1)) & \
        #                         (df['close'] < peak3.shift(1)) & \
        #                         (df['close'] > trough1.shift(1)) & \
        #                         (df['close'] > trough2.shift(1)) & \
        #                         (df['close'] > trough3.shift(1))
        
        # df['Falling_Wedge'] = falling_wedge_pattern.astype(int)
        # Параметры
        window = 10

        # Нахождение скользящих максимумов и минимумов
        df['peak1'] = df['close'].rolling(window=window).max()
        df['peak2'] = df['peak1'].shift(-window)
        df['peak3'] = df['peak1'].shift(-2*window)
        df['trough1'] = df['close'].rolling(window=window).min()
        df['trough2'] = df['trough1'].shift(-window)
        df['trough3'] = df['trough1'].shift(-2*window)

        # Условия для падающего клина
        falling_wedge_pattern = (df['close'] < df['peak1'].shift(1)) & \
                                (df['close'] < df['peak2'].shift(1)) & \
                                (df['close'] < df['peak3'].shift(1)) & \
                                (df['close'] > df['trough1'].shift(1)) & \
                                (df['close'] > df['trough2'].shift(1)) & \
                                (df['close'] > df['trough3'].shift(1))

        df.loc[df.index[self.start_ind]:,'Falling_Wedge'] = falling_wedge_pattern.astype(int)
        df = df.drop(columns=['peak1'])
        df = df.drop(columns=['peak2'])
        df = df.drop(columns=['peak3'])
        df = df.drop(columns=['trough1'])
        df = df.drop(columns=['trough2'])
        df = df.drop(columns=['trough3'])
        return df
    
    # Функция для определения паттерна "Молот" (Hammer)
    def add_candlestick_patterns(self,df):
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
        df.loc[df.index[self.start_ind]:,'Hammer'] = df.apply(is_hammer, axis=1).astype(int)
        df.loc[df.index[self.start_ind]:,'HangingMan'] = df.apply(is_hanging_man, axis=1).astype(int)

        return df


