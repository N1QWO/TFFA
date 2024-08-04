
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import sys

dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if dir not in sys.path:
    sys.path.append(dir)
from learn_test import learnNN
from prediction_model import test_data
from assessment_charms.main import best_param_model,filter_pred,graf_separation,graf_show


async def NN_LEARN(coin,new_filter=True,
             show_graf_separation=True,
             show_graf_show=True,
             update_data=1,
             learn_model = True,
             show_loss_line_inLearn=0):

    r, x_test, y_test = learnNN(coin, 'dp',
                                 update_data=update_data,
                                 show_loss_line=show_loss_line_inLearn,
                                ) if learn_model else test_data(coin, 
                                                                'dp',
                                                                update_data=update_data)


    print(f"coin : {r[0]['coin']} ")

    optim_name_bs = -1
    max_accuracy = 0.0
    max_a_precision = 0.0
    max_a_recall = 0.0
    real = np.argmax(y_test, axis=1)

    for j in range(len(r)):
        pred_model = r[j]['predicted']

        one_h = np.argmax(pred_model, axis=1)
        accuracy = accuracy_score(real, one_h)
        precision = precision_score(real, one_h)
        recall = recall_score(real, one_h)
        if max_accuracy < accuracy:
            max_accuracy = accuracy
            max_a_precision = precision
            max_a_recall = recall
            best_pred_model = pred_model
            optim_name_bs = j
        print("--------------------------------------------------------")
        print(f'accuracy на тестовом наборе {r[j]["optim_name"]}: {accuracy:.4f}')
        print(f'precision на тестовом наборе {r[j]["optim_name"]}: {precision:.4f}')
        print(f'recall на тестовом наборе {r[j]["optim_name"]}: {recall:.4f}')
        # if save_filter_weight:
        #     new_pred,new_real = filter_pred(pred_model,real,threshold=0.67)
        #     acc2 = accuracy_score(new_real,new_pred )
        #     accuracy = str(accuracy)[:6]
        #     acc2 = str(acc2)[:6]
        #     func_save_filter_weight(coin,r[j]["optim_name"],'dp',accuracy,acc2)

    print("--------------------------------------------------------")
    print(f'max_accuracy на тестовом наборе {r[optim_name_bs]["optim_name"]}: {max_accuracy:.4f} ; на {len(x_test)} примерах')
    
    best_param_model({r[optim_name_bs]["optim_name"]}, max_accuracy, max_a_precision, max_a_recall, x_test.shape)
    
    if new_filter:
        new_pred,new_real = filter_pred(best_pred_model,real,threshold=0.67) 
        print(f'new accuracy : {accuracy_score(new_pred,new_real ):.4f} на {len(new_pred)} примерах')
    if show_graf_separation:
        graf_separation(best_pred_model, real)
    if show_graf_show:
        await graf_show(x_test, y_test, r[optim_name_bs]['predicted'], r[optim_name_bs]["optim_name"],check=40)
    
    

if __name__=="__main__": 
    coin = 'BTCUSD'
    NN_LEARN(coin = coin, new_filter=False,show_graf_separation=True,show_graf_show=False,update_data=0,learn_model = True,show_loss_line_inLearn = 1)
    
