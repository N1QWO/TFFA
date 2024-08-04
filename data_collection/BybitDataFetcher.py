
import sys
import requests
import time
import os
import numpy as np
from datetime import datetime, timedelta
import pytz
import time

dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../'))
if dir not in sys.path:
    sys.path.append(dir)
dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
if dir not in sys.path:
    sys.path.append(dir)
try:

    from config.settings import BASE_DIR,DATA_REAL
    from preparation_data.DataPreparation import DataPreparation
    from learn_model.NN_model.model_params import model_hours_params
except ImportError as e:
    print("ImportError:", e)

class BybitDataFetcher:
    

    def __init__(self, symbol, interval=60):
        self.symbol = symbol
        self.interval = interval
        self.base_url = 'https://api.bybit.com/v5'

    def get_historical_data(self, from_timestamp, limit=1):
        endpoint = '/market/kline'
        params = {
            'category': 'linear',
            'symbol': self.symbol,
            'interval': self.interval,
            'start': from_timestamp * 1000,
            'limit': limit
        }
        
        while True:
            try:
                response = requests.get(self.base_url + endpoint, params=params)
                response.raise_for_status()
                data = response.json()
                res = data['result']['list']
                #{'ret_code': 0, 'ret_msg': 'OK', 'result': 
                # [{'symbol': 'TONUSDT', 'bid_price': '7.5973', 'ask_price': '7.5974', 'last_price': '7.5959', 
                # 'last_tick_direction': 'ZeroPlusTick', 'prev_price_24h': '7.1848', 'price_24h_pcnt': '0.057218', 
                # 'high_price_24h': '7.6498', 'low_price_24h': '7.1192', 'prev_price_1h': '7.6131', 'mark_price': '7.5986', 
                # 'index_price': '7.6018', 'open_interest': 19829620.2, 'countdown_hour': 0, 'turnover_24h': '58414828.9542', 
                # 'volume_24h': 7928522.4, 'funding_rate': '-0.00020033', 'predicted_funding_rate': '', 
                # 'next_funding_time': '2024-06-23T08:00:00Z', 
                # 'predicted_delivery_price': '', 
                # 'total_turnover': '', 'total_volume': 0, 
                # 'delivery_fee_rate': '', 'delivery_time': '', 'price_1h_pcnt': '', 'open_value': ''}], 'ext_code': '', 'ext_info': '', 'time_now': '1719123506.739364'}
                
                if 'result' in data:
                    return data['result']['list']
                else:
                    print(f"Unexpected response structure: {data}")
                    return []
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    print("Rate limit exceeded. Waiting for 1 minute before retrying...")
                    time.sleep(60)
                else:
                    print(f"HTTP error fetching data for {self.symbol}: {e}")
                    return []
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for {self.symbol}: {e}")
                return []
            except ValueError as e:
                print(f"Error decoding JSON: {e}")
                return []

    @staticmethod
    def date_to_timestamp(date):
        return int(date.timestamp())
    

    def download_line(self,load_list,pos,current_position):
        while pos < current_position/2:
                load_list[pos] = '/'  # Замена точки на символ /
                pos+=1
        sys.stdout.write('\r' + ''.join(load_list))  # Обновление строки в консоли
        sys.stdout.flush()    

        return (load_list,pos)

    

    def get_last_days_historical_data(self, days=1,date=None):

        # Установка временной зоны для Москвы (+03:00)
        local_timezone = pytz.timezone('Europe/Moscow')

        # Получаем текущее локальное время
        local_now = datetime.now()

        # Делаем объект datetime осведомленным о временной зоне
        local_now = local_timezone.localize(local_now)

        # Преобразуем локальное время в UTC
        utc_now = local_now.astimezone(pytz.utc)
        now = local_now
        print("Локальное время в Москве:", local_now)
        print("UTC время:", utc_now)

        if  date==None:
            from_date = now - timedelta(days=days)
        else:
            from_date = date
            
        from_timestamp = self.date_to_timestamp(from_date)
        to_timestamp = self.date_to_timestamp(now)
        # print(from_timestamp,to_timestamp)
        all_data = []
        current_from = from_timestamp
        
        delta = now-from_date
        hours_delta = delta.total_seconds() // 3600
        if hours_delta!=0:
            count_hours = 0
            pos = 0
            load_string = '..................................................'
            load_list = list(load_string)
            sys.stdout.write(f"\r{self.symbol} data download \n")

            while current_from < to_timestamp:
                
                current_position = int(count_hours*100//hours_delta)      
                load_string,pos = self.download_line(load_list,pos,current_position)

                count_hours+=1


                data = self.get_historical_data(current_from)
                if not data:
                    break
                all_data.extend(data)
                current_from = int(data[-1][0]) // 1000 + (int(self.interval) * 60)
        
        # current_position = int(count_hours*100//hours_delta)      
        # load_string,pos = self.download_line(load_list,pos,current_position)
        # sys.stdout.write(f"\r\n")

        return all_data

    def get_missing_data(self):

        
        path_coin = os.path.join(DATA_REAL,f'{self.symbol}.npy')

        all_data_np = np.load(path_coin) 
        from_date = int(all_data_np[-1,0])
        from_date = datetime.fromtimestamp(from_date / 1000)
        local_tz = pytz.timezone('Europe/Moscow')
        from_date = local_tz.localize(from_date)
        # print(from_date)

        # from_date = datetime.utcfromtimestamp(from_date)

        missing_data = self.get_last_days_historical_data(date = from_date)
        missing_data = np.array(missing_data)

        return missing_data
    
if __name__=='__main__':
    bdf = BybitDataFetcher('BTCUSD')
    missing_data = bdf.get_missing_data()
    
    print(missing_data)
