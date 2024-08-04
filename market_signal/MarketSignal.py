
import pandas as pd
import numpy as np
import os
import sys
import torch
from torch import  FloatTensor
from telegram import Bot
import time 
import requests
import asyncio
from tabulate import tabulate
from telegram.error import BadRequest,TelegramError

dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../'))
if dir not in sys.path:
    sys.path.append(dir)
dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
if dir not in sys.path:
    sys.path.append(dir)
# print(sys.path)
try:
    from learn_model.assessment_charms.main import graf_show  # Подставьте свои импорты здесь
    from learn_model.NN_model.ML import TorchNN  # Подставьте свои импорты здесь
    from data_collection.save_in_file import save_in_file  # Подставьте свои импорты здесь
    from config.settings import DATA_REAL,MODEL_WEIGHT_ELIT,TOKEN,URL,CHAT_ID
    from preparation_data.DataPreparation import DataPreparation
    from learn_model.NN_model.model_params import model_hours_params,get_filters,model_params,torch_optim
except ImportError as e:
    print("ImportError:", e)
    sys.exit(1)


class MarketSignal:
    """
    Класс для получения сигналов рынка на основе обученных моделей.
    """
    #список бумаг, уже обученные модели применяются к каждой бумаге
    #берется с api данные любой валюты за время, которое надо, чтобы модель могла выдать ответ
    #обучение происходило: x 10 часов часов(то-есть 10 показателей) и ответ y был рост или падение(но тут все под вопросом)
    #ахах этоп паренек еще не знал что он вернется сюда через 18 дней и показателей уже 36
    #среднее следующих 3(уже 5,сначало был 1, но там acc максимум 55) часов Y, с ним y [1,0] [0,1]
    

    #анализ по валюте  
    def __init__(self,coin):
        self.coin = coin
        
    def get_signal(self,coin):
        """
        Получение сигналов для указанной монеты.
        
        Args:
            coin (str): Название монеты (например, 'BTCUSD').

        Returns:
            tuple: DataFrame с данными и список предсказаний.
        """
        path = os.path.join(DATA_REAL,f'{coin}.npy')
        prev,after = model_hours_params()
        
        #храним выход из open_file_data соединяем с прошлым потом 

        dp = DataPreparation(prev,after)

        # start = time.time()
        df = dp.open_file_data(path)
        # end = time.time()
        # print('\nopen_file_data',end-start)

        start_index = max(0, len(df) - (prev + after))
        end_index = len(df)

        # start = time.time()
        X =  dp.data_preparation(df,percel_deviation=0,percel_deviation_top=1,Target=False)
        # end = time.time()
        # print('data_preparation',end-start)

        df = df.iloc[start_index:end_index,:]
        X = X[start_index:end_index,:]

        # start = time.time()
        X = dp.data_normalize_min_max_log(X)

        # end = time.time()
        # print('data_normalize_min_max_log',end-start)
        a,b,c = X.shape

        temp_X = X
        
        X = X.reshape(a,b*c)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {device}")

        torch_optimizers = torch_optim()

        X = FloatTensor(X).to(device)


        # start = time.time()
        pred = []
        filters_by = get_filters()
        nn_params = model_params(X.shape)
        for filter_by in filters_by:
            for optim_func, optim_name, optim_params in torch_optimizers:
                nn_model = TorchNN(**nn_params).to(device)  # Создание модели model_weights_{coin}_{optim_name}_{method_preparation}.pth'
                
                # Путь к директории с файлами
                files_path = os.path.join(MODEL_WEIGHT_ELIT, f'weights_{coin}/filter_model_weight/{filter_by}/{optim_name}') 
                if not  os.path.exists(files_path):
                    print(f"Папка не найдена {files_path}")
                    continue  
                # Получение списка файлов в указанной директории
                files_and_folders = os.listdir(files_path)
                # Фильтрация только файлов
                files = [item for item in files_and_folders if os.path.isfile(os.path.join(files_path, item))]

                # Проверка наличия файлов и выбор крайнего из них
                if not files:
                    print(f'нет файлов в {files_path}')
                    continue
                
                first_file = files[-1]
                file_path = os.path.join(files_path, first_file)
                #print(f"Найден файл: {file_path}")
                    
                # Загрузка весов модели
                nn_model.load_model_weights(file_path)

                t_nograd = nn_model.ready_model(X)    

                pred.append({'coin': coin, 'optim_name': optim_name, 'filter_by' : [str(filter_by),dp.get_index_filter_data(temp_X,filter_by)] ,'predicted': t_nograd.cpu().numpy()})
                
        
        # end = time.time()
        # print('cycle',end-start)
            
        return df,pred

    async def show_signal(self,df,signal,sendTG = False):
        """
        Отображение сигналов для каждой монеты.

        Args:
            df (DataFrame): DataFrame с данными.
            signal (list): Список предсказаний.
        """
        df_dict = {}

        for data in signal:
            coin, optim_name, filter_by, predicted = data['coin'], data['optim_name'], data['filter_by'], data['predicted']
            try:
                # Проверка формата данных predicted
                if isinstance(predicted, np.ndarray) and predicted.ndim == 2 and predicted.shape[1] == 2:
                    
                    if len(filter_by[1])==0:
                        continue
                    # Создание DataFrame для текущего элемента
                    df_temp = pd.DataFrame(predicted, columns=[f'{filter_by[0]} up', f'down'],index=df.index[-len(predicted):])
                    mask = ~df_temp.index.isin(df_temp.index[filter_by[1]])
                    
                    # print(mask)
                    df_temp.loc[mask, :] = 0  
                    
                    # Добавление DataFrame в словарь
                    if optim_name in df_dict:
                        df_dict[optim_name].append(df_temp)
                    else:
                        df_dict[optim_name] = [df_temp]
                else:
                    raise ValueError("predicted должен быть двумерным массивом с двумя колонками.")
            except ValueError as e:
                print(f"Ошибка при создании DataFrame для {coin} и {optim_name}: {e}")
                        
        for optim_name in df_dict:
            df_dict[optim_name] = pd.concat(df_dict[optim_name], axis=1)

        if sendTG: await self.send_to_telegram(f'coin: {self.coin}')
        # Проход по словарю df_dict и вывод информации
        for optim_name, df_list in df_dict.items():
            print(f'optim_name: {optim_name}')
            
            if sendTG: await self.send_to_telegram(f'optim_name: {optim_name}')
            # table = tabulate(df_list, headers='keys', tablefmt='github', showindex=False)
            print(df_list)
            if sendTG: await self.send_to_telegram(df_list.to_markdown())
    
        await graf_show(df,predict=signal,Target=False)

    async def send_to_telegram(self, message):     
        CHAT_id = CHAT_ID

        bot = Bot(token=TOKEN)
        try:
            await bot.send_message(chat_id=CHAT_id, text="\n```"+message+"\n```", parse_mode='Markdown')
        except BadRequest as e:
            print(f"Ошибка BadRequest при отправке сообщения в Telegram: {e}")
        except TelegramError as e:
            print(f"Произошла ошибка Telegram при отправке сообщения: {e}")

    async def send_table_to_telegram(self,table):
        bot = Bot(token=TOKEN)
        try:
            await bot.send_message(chat_id=CHAT_ID, text=f"```\n{table}\n```", parse_mode='Markdown')
        except BadRequest as e:
            print(f"Ошибка BadRequest при отправке сообщения в Telegram: {e}")
        except TelegramError as e:
            print(f"Произошла ошибка Telegram при отправке сообщения: {e}")

    def massage_signal_to_tg(self):
        '''
        receiving data prediction and send massage console and tg

        '''
        #в лучшем случае эта функция должна так
        #она запрашивает резуьтаты моделей из базы(которая обновляется заведомо каждый час)
        #в моем случае я каждый раз провожу все преобразования над данными и это занимает много времени
        #то есть загрузка сети дожна быть в :59 каждого часа по всем имеющимся валютам и следом сохранение в базу данных 
        #MissingBybitData и get_signal вызываются в :59
        #show_signal в любое время (исключание: загрузка сети :59 попросту неактульная(неполная) информация)
        
        start = time.time()

        sf = save_in_file()
        sf.MissingBybitData(self.coin, self.coin)

        end = time.time()
        print('\nMissingBybitData',end-start)

        df, signal = self.get_signal(self.coin)
        
        start = time.time()
        asyncio.run(self.show_signal(df, signal,sendTG=True))
        end = time.time()
        print('show_signal',end-start)


if __name__=='__main__':
    start = time.time()

    coin = 'BTCUSD'
    ms = MarketSignal(coin)
    ms.massage_signal_to_tg()

    end = time.time()
    print('massage_signal_to_tg',end-start)