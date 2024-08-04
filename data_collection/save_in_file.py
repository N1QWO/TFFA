# main.py
import os
import sys
import numpy as np

dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if dir not in sys.path:
    sys.path.append(dir)
dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
if dir not in sys.path:
    sys.path.append(dir)

try:
    from data_collection.BybitDataFetcher import BybitDataFetcher
    from config.settings import BASE_DIR, DATA_TRAIN,DATA_REAL
except ImportError as e:
    print("ImportError:", e)

class save_in_file:
    """
    Класс для сохранения данных с биржи Bybit в файлы.

    Методы:
    - SaveBybitData: Сохраняет исторические данные с биржи Bybit за указанный период.
    - MissingBybitData: Дополняет существующие данные новыми, отсутствующими данными.
    """

    def __init__(self):
        pass
    
    def SaveBybitData(self, path, symbol, day, interval, path_TR ='data_train'):
        """
        Сохраняет исторические данные с биржи Bybit за указанный период в файл.

        Аргументы:
        - path (str): Путь к файлу для сохранения данных.
        - symbol (str): Символ валютной пары (например, 'BTCUSD').
        - day (int): Количество дней, за которые необходимо получить данные.
        - interval (int): Интервал времени для данных (в минутах).
        - path_TR (str): Директория для сохранения файла с данными.
        """

        # Путь к файлу для сохранения данных
        
        output_dir = os.path.join(DATA_TRAIN, symbol)
        os.makedirs(output_dir, exist_ok=True)

        file_path = os.path.join(BASE_DIR, path)
        

        # Создание экземпляра класса для получения данных
        fetcher = BybitDataFetcher(symbol=symbol, interval=str(interval))

        data = fetcher.get_last_days_historical_data(days=day)
        data = np.array(data).astype(float)
        # print(data)
        # Запись данных в файл
        np.save(file_path, data)

        print(f'\nData has been written to {file_path}')

    def MissingBybitData(self, path, symbol, interval=60, path_TR='data_real'):
        """
        Дополняет существующие данные новыми, отсутствующими данными.

        Аргументы:
        - path (str): Путь к файлу для сохранения данных.
        - symbol (str): Символ валютной пары (например, 'BTCUSD').
        - interval (int): Интервал времени для данных (в минутах, по умолчанию 60).
        - path_TR (str): Директория для сохранения файла с данными (по умолчанию 'data_real').
        """

        file_path = os.path.join(DATA_REAL, f'{path}.npy')
        # path_load = os.path.join(DATA_TRAIN, f'{symbol}.npy')

        data = np.load(file_path)

        
        bdf = BybitDataFetcher(symbol=symbol, interval=str(interval))
        new_data = bdf.get_missing_data()
        
        if len(new_data)>1:
            if data[-1,0]!=new_data[0,0]:
                new_data = np.vstack((data, new_data[:-1,:]))
            else:
                new_data = np.vstack((data, new_data[1:-1,:]))
            # print(data.shape, new_data.shape)
            np.save(file_path, new_data)
        else:
            np.save(file_path, data)
        # print(f"\nData has been written to {file_path}")

if __name__ == '__main__':
    sf = save_in_file()
    sf.MissingBybitData('BTCUSD', 'BTCUSD')


        # data = pd.DataFrame(data, columns=col)



        # bdf = BybitDataFetcher(symbol=symbol, interval=str(interval))
        # new_data = bdf.get_missing_data()

        # ln_new_data = len(new_data)
        # if ln_new_data>1:
        #     new_data = new_data[:-1,:]
        #     ld_col = ['timestamp','open', 'high', 'low', 'close', 'volume', 'turnover']

        #     new_data= pd.DataFrame(new_data,columns=ld_col)
        #     dp = DataProcessor()
        #     new_data = dp.process_dataframe(new_data)

        #     df = pd.DataFrame(np.zeros((len(new_data), len(col))), columns=col)
        #     df[ld_col] = new_data