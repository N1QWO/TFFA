import time 
from market_signal.MarketSignal import MarketSignal 

#main 


if __name__=='__main__':
    start = time.time()

    coin = 'TONUSDT'
    ms = MarketSignal(coin)
    ms.massage_signal_to_tg()

    end = time.time()
    print('massage_signal_to_tg',end-start)

