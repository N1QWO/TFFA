
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import sys

dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if dir not in sys.path:
    sys.path.append(dir)
from learn import learn
from prediction import test_data
from assessment_charms.main import best_param_model,filter_pred,graf_separation,graf_show


def RNN_LEARN(coin,new_filter=False,show_graf_separation=True,show_graf_show=True,update_data=1,learn_model = True,show_loss_line_inLearn=0):


    
    r, x_test, y_test = learn(coin, 'dp_rnn',update_data=update_data,show_loss_line=show_loss_line_inLearn) if \
                        learn_model else \
                        test_data(coin, 'dp_rnn',update_data=update_data)

    print(f"coin : {r[0]['coin']} ")

    optim_name_bs = -1
    max_accuracy = 0.0
    max_a_precision = 0.0
    max_a_recall = 0.0
    real = np.argmax(y_test[:,-1,:], axis=1)

    for j in range(len(r)):
        pred_model = r[j]['predicted']
        # print('pred_model',pred_model,pred_model.shape)
        # print('real',real,real.shape)
        one_h = np.argmax(pred_model, axis=1)
        
        # print('one_h',one_h,one_h.shape)
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

    print("--------------------------------------------------------")
    print(f'max_accuracy на тестовом наборе {r[optim_name_bs]["optim_name"]}: {max_accuracy:.4f} ; на {len(x_test)} примерах')
    
    best_param_model({r[optim_name_bs]["optim_name"]}, max_accuracy, max_a_precision, max_a_recall, x_test.shape)
    
    if new_filter:
        new_pred,new_real = filter_pred(best_pred_model,real) 
        print(f'new accuracy : {accuracy_score(new_pred,new_real ):.4f} на {len(new_pred)} примерах')
    if show_graf_separation:
        graf_separation(best_pred_model, real)
    if show_graf_show:
        graf_show(x_test, y_test, r[optim_name_bs]['predicted'], r[optim_name_bs]["optim_name"])

