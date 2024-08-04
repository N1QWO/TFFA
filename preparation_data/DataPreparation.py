import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import os
import sys
if __name__=='__main__':
    from scipy.stats import norm

dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if dir not in sys.path:
    sys.path.append(dir)
if __name__=="__main__":
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
    if dir not in sys.path:
        sys.path.append(dir)
try:  
    from DataProcessor import DataProcessor,DataPattern
    from learn_model.NN_model.model_params import get_filter
    from config.settings import PREPARATION_DATA,DATA_PREPARATION,DATA_TRAIN
except ImportError as e:
    print("ImportError:", e)

class signal_Filter_data():
    def __init__(self,df):
        self.df = df
    # def all(self): return True
    
    def scrollmean(self):
        """
        SMA_7 ниже SMA_28 и каждоее следующее SMA_7 выше

        SMA_7<SMA_28

        SMA_7[-1]>SMA_7[-2]

        """
        flag1 = self.df.loc[self.df.index[-1], 'SMA_7'] < self.df.loc[self.df.index[-1], 'SMA_28'] and self.df.loc[self.df.index[-2], 'SMA_7'] < self.df.loc[self.df.index[-2], 'SMA_28'] and self.df.loc[self.df.index[-3], 'SMA_7'] < self.df.loc[self.df.index[-3], 'SMA_28']
        flag2 = self.df.loc[self.df.index[-1], 'SMA_7'] > self.df.loc[self.df.index[-2], 'SMA_7'] and self.df.loc[self.df.index[-2], 'SMA_7'] > self.df.loc[self.df.index[-3], 'SMA_7'] and self.df.loc[self.df.index[-3], 'SMA_7'] > self.df.loc[self.df.index[-4], 'SMA_7']
                    
        return flag1 and flag2 
    def scrollmean_up(self):
        """
        SMA_7 выше SMA_28 и каждоее следующее SMA_7 ниже

        SMA_7>SMA_28

        SMA_7[-1]<SMA_7[-2]

        """
        flag1 = self.df.loc[self.df.index[-1], 'SMA_7'] > self.df.loc[self.df.index[-1], 'SMA_28'] and self.df.loc[self.df.index[-2], 'SMA_7'] > self.df.loc[self.df.index[-2], 'SMA_28'] and self.df.loc[self.df.index[-3], 'SMA_7'] > self.df.loc[self.df.index[-3], 'SMA_28']
        flag2 = self.df.loc[self.df.index[-1], 'SMA_7'] < self.df.loc[self.df.index[-2], 'SMA_7'] and self.df.loc[self.df.index[-2], 'SMA_7'] < self.df.loc[self.df.index[-3], 'SMA_7'] and self.df.loc[self.df.index[-3], 'SMA_7'] < self.df.loc[self.df.index[-4], 'SMA_7']
        return flag1 and flag2       
    def scrollmean_top(self):
        """

        SMA_7 выше SMA_28 и каждоее следующее SMA_7 ниже, SMA_28 выше

        SMA_7 > SMA_28

        SMA_7[-1]<SMA_7[-2]

        SMA_28[-1]>SMA_28[-2]

        """

        flag1 = self.df.loc[self.df.index[-1], 'SMA_7'] > self.df.loc[self.df.index[-1], 'SMA_28'] and self.df.loc[self.df.index[-2], 'SMA_7'] > self.df.loc[self.df.index[-2], 'SMA_28'] and self.df.loc[self.df.index[-3], 'SMA_7'] > self.df.loc[self.df.index[-3], 'SMA_28'] and self.df.loc[self.df.index[-4], 'SMA_7'] > self.df.loc[self.df.index[-4], 'SMA_28']
        flag2 = self.df.loc[self.df.index[-1], 'SMA_7'] < self.df.loc[self.df.index[-2], 'SMA_7'] and self.df.loc[self.df.index[-2], 'SMA_7'] > self.df.loc[self.df.index[-3], 'SMA_7'] and self.df.loc[self.df.index[-3], 'SMA_7'] > self.df.loc[self.df.index[-4], 'SMA_7'] 
        flag3 = self.df.loc[self.df.index[-1], 'SMA_28'] > self.df.loc[self.df.index[-2], 'SMA_28'] and self.df.loc[self.df.index[-2], 'SMA_28'] > self.df.loc[self.df.index[-3], 'SMA_28'] and self.df.loc[self.df.index[-3], 'SMA_28'] > self.df.loc[self.df.index[-4], 'SMA_28'] 
        return flag1 and flag2 and flag3          
    def scrollmean_low(self):
        """
        SMA_7 ниже SMA_28 и каждоее следующее SMA_7 выше, SMA_28 ниже

        SMA_7 < SMA_28

        SMA_7[-1] > SMA_7[-2]

        SMA_28[-1] < SMA_28[-2]

        """
        flag1 = self.df.loc[self.df.index[-1], 'SMA_7'] < self.df.loc[self.df.index[-1], 'SMA_28'] and self.df.loc[self.df.index[-2], 'SMA_7'] < self.df.loc[self.df.index[-2], 'SMA_28'] and self.df.loc[self.df.index[-3], 'SMA_7'] < self.df.loc[self.df.index[-3], 'SMA_28'] and self.df.loc[self.df.index[-4], 'SMA_7'] < self.df.loc[self.df.index[-4], 'SMA_28']
        flag2 = self.df.loc[self.df.index[-1], 'SMA_7'] > self.df.loc[self.df.index[-2], 'SMA_7'] and self.df.loc[self.df.index[-2], 'SMA_7'] < self.df.loc[self.df.index[-3], 'SMA_7'] and self.df.loc[self.df.index[-3], 'SMA_7'] < self.df.loc[self.df.index[-4], 'SMA_7'] 
        flag3 = self.df.loc[self.df.index[-1], 'SMA_28'] < self.df.loc[self.df.index[-2], 'SMA_28'] and self.df.loc[self.df.index[-2], 'SMA_28'] < self.df.loc[self.df.index[-3], 'SMA_28'] and self.df.loc[self.df.index[-3], 'SMA_28'] < self.df.loc[self.df.index[-4], 'SMA_28'] 
        return flag1 and flag2 and flag3         
    def rsi_oversold(self):
        """
        Проверка перепроданности RSI (когда RSI < 30 и все предыдущие RSI были выше 30).
        """
        # Проверяем, что текущее RSI меньше 30
        if self.df.loc[self.df.index[-1], 'RSI_7'] >= 30:
            return False
        
        
        # Проверяем, что все предыдущие RSI были выше или равны 30
        for i in range(2, min(5,len(self.df))):  # Проверяем последние 5 значений RSI
            
            if self.df.loc[self.df.index[-i], 'RSI_7'] < 30:
                return False 
        return True

    def rsi_overbought(self):
        """
        Проверяет наличие сигнала перекупленности по RSI.

        RSI выше 70 (перекуплен).
        """
        # Проверяем, что текущее RSI меньше 30
        
        if self.df.loc[self.df.index[-1], 'RSI_7'] < 70:
            return False
        
        # Проверяем, что все предыдущие RSI были выше или равны 30
        for i in range(2,min(5,len(self.df))):  # Проверяем последние 5 значений RSI
            if self.df.loc[self.df.index[-i], 'RSI_7'] >= 70:
                return False
        
        return True
    def macd_crossover(self):
        """
        Проверяет наличие пересечения MACD и сигнальной линии (MACD crossover).

        MACD (12, 26) выше сигнальной линии (9).
        """
        macd = self.df['MACD']
        signal_line = self.df['Signal_line']
        return macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]    
    def patter_param(self,filter_param):
        """
        
        поиск filter_param паттерна
        
        """

        return self.df.loc[self.df.index[-1], filter_param] == 1

# Определение класса для подготовки данных
class DataPreparation:

    
    def __init__(self, prevHours=5, afterHours=1):
        """Инициализация объекта DataPreparation.

        Args:
        prevHours (int): Количество часов для предшествующего окна данных.
        afterHours (int): Количество часов для следующего окна данных.
        """
        self.prevHours = prevHours
        self.afterHours = afterHours
        

    def open_file_data(self, path,data_columns=True):     
        """Загрузка данных из файла и их преобразование с помощью DataProcessor и DataPattern.

        Args:
        path (str): Путь к файлу с данными.

        Returns:
        pandas.DataFrame: Обработанные данные.
        """
        df = np.load(path)
        # print(df)
        dataProc = DataProcessor()
        df = dataProc.forward_data(df)
        dataPatt = DataPattern(shoulders_tolerance=0.03)
        df = dataPatt.get_pattern(df)
        
        if data_columns:
            colm =df.columns.tolist()
            # print(' '.join(map(str,colm)))
            file_path = os.path.join(PREPARATION_DATA, 'data_columns.txt')
            with open(file_path,'w') as f:
                f.write(' '.join(map(str,colm)))
        return df

    def data_normalize_min_max_log(self,patch):
        """Нормализация данных по минимуму и максимуму с применением логарифмического преобразования.

        Args:
        patch (numpy.ndarray): 3D массив данных для нормализации.

        Returns:
        numpy.ndarray: Нормализованные данные.
        """
        depth,row,col =patch.shape
        columns_to_normalize =  np.array(['open', 'high', 'low', 'close','average_price','SMA_3', 'SMA_7', 'SMA_28','EMA_5', 'EMA_20','Upper_Band','Lower_Band'])
        log_norm  = np.array(["volume",'turnover'])

        columns =  []
        file_path = os.path.join(PREPARATION_DATA, 'data_columns.txt')
        with open(file_path,'r') as f:
            columns = f.readline().split(' ')
        columns = np.array(columns)
        columns_to_normalize = np.intersect1d(columns_to_normalize,columns)
        log_norm = np.intersect1d(log_norm,columns)
        columns = columns.tolist()
        log_norm = log_norm.tolist()
        columns_to_normalize = columns_to_normalize.tolist()

        for i in range(depth):
            df = pd.DataFrame(patch[i,:,:],columns=columns)

            r,c = df[columns_to_normalize].shape
            tm = df[columns_to_normalize].values.reshape(1, r*c)
            mx = tm.max()
            mn = tm.min()
            df[columns_to_normalize]= (df[columns_to_normalize]-mn)/(mx-mn)
            if len(log_norm)!=0:
                for colum in log_norm:
                    df[colum] = df[colum].apply(lambda x: np.log(x + 1) if x > 0 else np.log(1))
            
            patch[i] = np.array(df)
        return patch

    def data_preparation(self, df,percel_deviation=0,percel_deviation_top=100,Target=True):
        """Подготовка данных для обучения модели.

        Args:
        df (pandas.DataFrame): Исходные данные.
        percel_deviation (float): Нижняя граница процентного изменения.
        percel_deviation_top (float): Верхняя граница процентного изменения.

        Returns:
        tuple: Массивы X (предикторы) и Y (метки).
        """
        row,col = df.shape

        prev_win = df.rolling(window=self.prevHours)
        X = [window.values for window in prev_win if len(window) == self.prevHours]
        if Target:
            X = np.array(X)[:-self.afterHours,:,:]
            next_mean = df['close'].shift(-1).rolling(window=self.afterHours).mean()
            condition = next_mean > df['close']
            Y = np.where(condition, 0, 1)  # Создаем массив из нулей той же размерности, что и condition
            Y = self.bin_to_onehot(Y)
            Y = Y[self.prevHours:,:]
            # print(X.shape,Y.shape)
            if percel_deviation > 0 or percel_deviation_top<100:
                res = df['close'] / df['close'].shift(1)
                res=  res.reset_index(drop=True)
                res = res[self.prevHours:len(res)-self.afterHours+1]
                up = np.abs(res - 1) * 100 > percel_deviation 
                down = np.abs(res - 1) * 100 < percel_deviation_top
                filter_indices = np.intersect1d(np.where(up)[0], np.where(down)[0])
                # Обновляем X и Y, используя стандартные индексы
                X = X[filter_indices, :,:]
                Y = Y[filter_indices, :]
            return (X, Y)
        else:
            X = np.array(X)[:,:,:]
            return X

    def data_preparation_rnn(self, df,percel_deviation=0,percel_deviation_top=100):
        """Подготовка данных для рекуррентной нейронной сети.

        Args:
        df (pandas.DataFrame): Исходные данные.
        percel_deviation (float): Нижняя граница процентного изменения.
        percel_deviation_top (float): Верхняя граница процентного изменения.

        Returns:
        tuple: Массивы X (предикторы) и Y (метки).
        """
        row,col = df.shape
        # print("df" ,np.array(df).shape)
        prev_win = df.rolling(window=self.prevHours)
        # print("prev_win" ,np.array(prev_win.mean()).shape)
        X = np.array([window.values for window in prev_win if len(window) == self.prevHours])[:-1,:,:]
        # print("X" ,X.shape)
        next_mean = df['close'].shift(-1).rolling(window=self.afterHours).mean()
        # print("next_mean" ,np.array(next_mean).shape)
        condition = next_mean > df['close']
        # print("condition" ,np.array(condition).shape)
        Y = np.where(condition, 0, 1)  # Создаем массив из нулей той же размерности, что и condition
        
        # print("Y" ,Y.shape)
        Y = self.bin_to_onehot(Y)
        # print("Y" ,Y.shape)
        Y = pd.DataFrame(Y).rolling(window=self.prevHours)
        # print("Y" ,Y.mean())
        # print(Y.mean().shape,condition.shape)
        Y = np.array([window.values for window in Y if len(window) == self.prevHours])[:-self.afterHours,:,:]
    
        
        # print(X.shape,Y.shape,condition.shape)
        if percel_deviation > 0 or percel_deviation_top<100:
            res = df['close'] / df['close'].shift(1)
            res=  res.reset_index(drop=True)
            res = res[self.prevHours:len(res)-self.afterHours+1]
            up = np.abs(res - 1) * 100 > percel_deviation 
            down = np.abs(res - 1) * 100 < percel_deviation_top
            filter_indices = np.intersect1d(np.where(up)[0], np.where(down)[0])
            # Обновляем X и Y, используя стандартные индексы
            X = X[filter_indices, :,:]
            Y = Y[filter_indices, :,:]
        return (X, Y)
    
    def data_statistics(self,path):
        """Вычисление статистики данных.

        Args:
        path (str): Путь к файлу с данными.

        Returns:
        numpy.ndarray: Массив статистик.
        """
        df = self.open_file_data(path)
        res = df['close']/df['close'].shift(1)
        res = np.array(res)
        
        non_nan_indices = ~np.isnan(res)
        res = res[non_nan_indices]
        return res
    
    def data_dist(self,path):
        """Построение гистограммы распределения данных.

        Args:
        path (str): Путь к файлу с данными.
        """
        data = self.data_statistics(path)
        data = (data-1)*100
        df = pd.DataFrame(data, columns=['percel'])
        df.hist(bins=1000)  
        plt.show()

    def bin_to_onehot(self,Y):
        """Преобразование бинарных меток классов в формат "one-hot".

        Args:
        bin_labels (numpy.ndarray): Бинарные метки классов.

        Returns:
        numpy.ndarray: Массив "one-hot" меток.
        """
        Y_res = np.zeros((Y.size, 2))
        Y_res[np.arange(Y.size), Y] = 1
        return Y_res
    def bin_to_onehot_3dim(self,Y):

        num_classes = Y.max() + 1  # Определяем количество классов
        onehot = np.zeros((Y.shape[0], Y.shape[1], num_classes))  # Создаем пустую матрицу нулей с добавлением измерения для классов
        for i in range(Y.shape[0]):
            onehot[i, np.arange(Y.shape[1]), Y[i]] = 1
        return onehot

    def resample(self,X, Y):
        """Сэмплирование данных с использованием RandomUnderSampler.

        Args:
        X (numpy.ndarray): Предикторы.
        Y (numpy.ndarray): Метки.

        Returns:
        tuple: Сэмплированные предикторы и метки.
        """
        rus = RandomUnderSampler(random_state=42)
        return rus.fit_resample(X, Y)

    
    def cleaning_nan(self,X, Y):
        """Удаление строк с NaN в данных.

        Args:
        df (pandas.DataFrame): Исходные данные.

        Returns:
        pandas.DataFrame: Очищенные данные.
        """
        X_np = np.array(X)

        # Находим индексы строк без NaN в X
        non_nan_indices = ~np.isnan(X_np).any(axis=(1,2))

        # Фильтруем X, удаляя строки с NaN
        X_cleaned = X_np[non_nan_indices]

        # Фильтруем Y, оставляя только те элементы, индексы которых совпадают с non_nan_indices
        Y_cleaned = Y[non_nan_indices]
        return (X_cleaned,Y_cleaned)
    
    def get_index_filter_data(self,X,filter_param):
        '''
        X : 3dim tensor 
        filter_param : array of name pattern or my filter_methods

        return np.array of index when find pattern or filter_methods
        '''

        if filter_param=='all':
            ind_filter = [i for i in range(len(X))]
            return np.array(ind_filter)
        
        columns =  []
        file_path = os.path.join(PREPARATION_DATA, 'data_columns.txt')
        with open(file_path,'r') as f:
            columns = f.readline().split(' ')
            
        ind_filter = []
        filter_methods = ['scrollmean', 'scrollmean_up', 'scrollmean_top', 'scrollmean_low','rsi_oversold','rsi_overbought','macd_crossover']
    
        if filter_param in filter_methods:
            filter_method = filter_param
        else:
            filter_method = None

        for i in range(len(X)):
            df = pd.DataFrame(X[i, :, :], columns=columns)
            signal = signal_Filter_data(df)
            
            try:
                if filter_method:
                    if getattr(signal, filter_param)():
                        ind_filter.append(i)
                else:
                    if signal.patter_param(filter_param):
                        ind_filter.append(i)
            except KeyError as err:
                print(f"KeyError: {err} for index {i} and filter_param {filter_param}")
        
        return np.array(ind_filter)
        
    def filter_data(self,X,Y = [],filter_param='',Target = True):
        
        ind_filter = self.get_index_filter_data(X,filter_param)
        if(len(ind_filter)==0):
            print(f"пустой массив filter_data")
            return (None,None)
        if Target:
            X = X[ind_filter]
            Y = Y[ind_filter]
            return (X, Y)
        X = X[ind_filter]  
        return X

    def dp_rnn(self,path,coin,normalize=True):
        """Подготовка данных для рекуррентной нейронной сети.

        Returns:
        numpy.ndarray: Данные для обучения.
        """
        df = self.open_file_data(path)
        
        X, Y = self.data_preparation_rnn(df,percel_deviation=0,percel_deviation_top=1)
        X,Y = self.cleaning_nan(X, Y)
        if normalize:
            X = self.data_normalize_min_max_log(X)
        Y = np.argmax(Y, axis=2)

        file_path = os.path.join(DATA_PREPARATION, f'{coin}_data_preparation_rnn.npz')
        np.savez(file_path, X=X, Y=Y)
        Y_last_dimension = Y[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,stratify=Y_last_dimension, random_state=42)
        

        # train_class_distribution = np.bincount(y_train)
        # test_class_distribution = np.bincount(y_test)
        y_train,y_test = self.bin_to_onehot_3dim(y_train),self.bin_to_onehot_3dim(y_test)
        # print(f"Train class distribution: {train_class_distribution}")
        # print(f"Test class distribution: {test_class_distribution}")

        return (X_train, X_test, y_train, y_test)
    
    def dp(self, path,coin,normalize=True):
        """Подготовка данных для обучения.

        Returns:
        numpy.ndarray: Данные для обучения.
        """
        df = self.open_file_data(path)
        
        X, Y = self.data_preparation(df,percel_deviation=0,percel_deviation_top=1)
        X,Y = self.cleaning_nan(X, Y)
        filter_by = get_filter()
        if filter_by!='all':
            print('filter_by',filter_by)
            X,Y = self.filter_data(X,Y,filter_by)
        if normalize:
            X = self.data_normalize_min_max_log(X)
        a,b,c = X.shape
        X = X.reshape(a,b*c)
        Y = np.argmax(Y, axis=1)
        X, Y = self.resample(X, Y)
        
        file_path = os.path.join(DATA_TRAIN, f'{coin}')
        if not os.path.exists(file_path):
            try:
                # Создаем папку
                os.makedirs(file_path)
            except OSError as e:
                print(f"Ошибка при создании папки '{file_path}': {e}")
        file_path = os.path.join(file_path, f'{get_filter()}.npz')
        np.savez(file_path, X=X, Y=Y)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,stratify=Y, random_state=42)
        

        train_class_distribution = np.bincount(y_train)
        test_class_distribution = np.bincount(y_test)
        y_train,y_test = self.bin_to_onehot(y_train),self.bin_to_onehot(y_test)
        print(f"Train class distribution: {train_class_distribution}")
        print(f"Test class distribution: {test_class_distribution}")



        return (X_train, X_test, y_train, y_test)

    def ready_data(self,coin):
        """
        Загрузка готовых данных для обучения.

        Args:
        coin (str): Название монеты или другой идентификатор.

        Returns:
        tuple: Кортеж с данными для обучения (X_train, X_test, y_train, y_test).
        """  
        file_path = os.path.join(DATA_TRAIN, f'{coin}')
        if not os.path.exists(file_path):
            try:
                os.makedirs(file_path)
            except OSError as e:
                print(f"Ошибка при создании папки '{file_path}': {e}")
                
        file_path = os.path.join(file_path, f'{get_filter()}.npz')
        if not os.path.exists(file_path):
            print(f"Файл '{file_path}' не существует или указан неверный путь.")
            return None
        
        loaded = np.load(file_path)
        X_loaded = loaded['X']
        Y_loaded = loaded['Y']
        X_train, X_test, y_train, y_test = train_test_split(X_loaded, Y_loaded, test_size=0.2,stratify=Y_loaded, random_state=42)
        y_train,y_test = self.bin_to_onehot(y_train),self.bin_to_onehot(y_test)

        return (X_train, X_test, y_train, y_test)

    def raedy_data_rnn(self,coin):
        """Загрузка готовых данных для рекуррентной нейронной сети.

        Returns:
        numpy.ndarray: Данные для обучения.
        """
        file_path = os.path.join(DATA_PREPARATION, f'{coin}_data_preparation_rnn.npz')
        loaded = np.load(file_path)
        X_loaded = loaded['X']
        Y_loaded = loaded['Y']
        Y_last_dimension = Y_loaded[:, 4]
        X_train, X_test, y_train, y_test = train_test_split(X_loaded, Y_loaded, test_size=0.2,stratify=Y_last_dimension, random_state=42)
        y_train,y_test = self.bin_to_onehot_3dim(y_train),self.bin_to_onehot_3dim(y_test)

        return (X_train, X_test, y_train, y_test)

    def call_data_preparation(self, path, method_name,coin,update_data=1):
        """Вызов методов подготовки данных в зависимости от выбранной модели.

        Args:
        model (str): Название модели ('dp', 'dp_rnn').

        Returns:
        numpy.ndarray: Данные для обучения.
        """
        if update_data: 
            function_map = {
                'dp': self.dp,
                'dp_rnn': self.dp_rnn
            }
            if method_name in function_map:
                return function_map[method_name](path,coin)
            else:
                return "Function not found"
        else:
            function_map = {
                'dp': self.ready_data,
                'dp_rnn': self.raedy_data_rnn
            }
            if method_name in function_map:
                return function_map[method_name](coin)
            else:
                return "Function not found"



if __name__=="__main__":
    coin = 'BTCUSD'
    d = DataPreparation()
    # X_train, X_test, y_train, y_test = d.call_data_preparation('D:/pg/python/TFF/data_train/BTCUSD.npy','dp_rnn','BTCUSD',update_data=0)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    path_coin = os.path.join(DATA_TRAIN,f'{coin}.npy')
    df = d.open_file_data(path_coin)
    

    # plt.figure(figsize=(14, 7))
    # plt.plot(df['close'], label='Close Price')
    # plt.plot(df['peak1'], label='Peak 1', linestyle='--', alpha=0.7)
    # plt.plot(df['peak2'], label='Peak 2', linestyle='--', alpha=0.7)
    # plt.plot(df['peak3'], label='Peak 3', linestyle='--', alpha=0.7)
    # plt.plot(df['trough1'], label='Trough 1', linestyle='--', alpha=0.7)
    # plt.plot(df['trough2'], label='Trough 2', linestyle='--', alpha=0.7)
    # plt.plot(df['trough3'], label='Trough 3', linestyle='--', alpha=0.7)

    # # Highlight falling wedge
    # plt.plot(df.index[df['Falling_Wedge'] == 1], df['close'][df['Falling_Wedge'] == 1], 'ro', label='Falling Wedge')

    # plt.legend()
    # plt.title('Falling Wedge Pattern')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.show()
    df = np.array(df)
    row,col = df.shape
    df = df.reshape(row,1,col)
    df = d.data_normalize_min_max_log(df)

    columns =  []
    file_path = os.path.join(PREPARATION_DATA, 'data_columns.txt')
    with open(file_path,'r') as f:
        columns = f.readline().split(' ')
    row,de,col = df.shape
    df = df.reshape(row,de*col)
    df = pd.DataFrame(df,columns = columns)
    df = df[['open','close']]


    # Вычисление разности между ценами открытия и закрытия
    df['diff'] = df['close'] - df['open']

    # Нормализация данных
    mean_diff = df['diff'].mean()
    std_diff = df['diff'].std()
    df['diff_normalized'] = (df['diff'] - mean_diff) / std_diff

    # Построение гистограммы нормализованных данных
    plt.hist(df[['diff_normalized','diff']], bins=200, density=True, alpha=0.6)

    # Нанесение на график нормального распределения
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, 0, 1)

    print(df)
    plt.plot(x, p, 'k', linewidth=2)
    title = "график распределения тела(свечи) close-open : mean = %.2f,  std = %.2f" % (0, 1)
    plt.title(title)
    plt.tight_layout()
    plt.show()

