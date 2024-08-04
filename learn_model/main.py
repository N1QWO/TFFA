
import sys
import os
import asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from NN_model.main import NN_LEARN
from RNN_model.main import RNN_LEARN

async def main():
    coin = 'BTCUSD'
    await NN_LEARN(
        coin = coin,
        new_filter=True,
        show_graf_separation=True,
        show_graf_show=True,
        update_data=0,
        learn_model = True,
        show_loss_line_inLearn = 1
    )

if __name__=="__main__": 
    asyncio.run(main())
    # я давно не раскомменчивал и там стоит скорей всего доработать с переодресацией данных
    # показатели на rnn плохие, но в тоже время обучение сводится к 0 loss
    # RNN_LEARN(coin = coin,
    #           new_filter=False,
    #           show_graf_separation=True,
    #           show_graf_show=False,
    #           update_data=0,
    #           learn_model = True,
    #           show_loss_line_inLearn = 1
    #           )

 
