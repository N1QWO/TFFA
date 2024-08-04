import torch
from torch import FloatTensor
import numpy as np
import os
import sys

dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if dir not in sys.path:
    sys.path.append(dir)
try:
    from preparation_data.DataPreparation import DataPreparation
    from model_params_rnn import torch_optim,model_params_rnn,model_hours_params
    from RNN import LSTM, evaluate_model
    from config.settings import MODEL_WEIGHT,BASE_DIR
except ImportError as e:
    print("ImportError:", e)


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_data(coin,method_preparation,update_data=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    # Список различных оптимизаторов и их параметры
    torch_optimizers = torch_optim()

    # Путь к файлу с данными по монете
    path_coin = os.path.join(BASE_DIR,f'data_real/{coin}.npy')

    # Подготовка данных для обучения и тестирования
    prevHours,afterHours = model_hours_params()
    DP = DataPreparation(prevHours,afterHours)
    X_train, X_test, y_train, y_test = [],[],[],[]
    X_train, X_test, y_train, y_test = DP.call_data_preparation(path_coin,method_preparation,coin,update_data=update_data)
    X_train, X_test = FloatTensor(np.array(X_train)).to(device), FloatTensor(np.array(X_test)).to(device)
    y_train, y_test = FloatTensor(np.array(y_train)).to(device), FloatTensor(np.array(y_test)).to(device)

    # Параметры для нейронной сети
    nn_params = model_params_rnn(X_train.shape)

    # Список для хранения предсказаний
    pred = []
    
    # Обучение и проверка модели для каждого оптимизатора
    for optim_func, optim_name, optim_params in torch_optimizers:
        nn_model = LSTM(**nn_params).to(device)  # Создание модели
        path_weights = os.path.join(MODEL_WEIGHT,f'weights_rnn_{coin}\model_weights_rnn_{coin}_{optim_name}_{method_preparation}.pth') # Путь к весам модели
        model_weights = torch.load(path_weights)  # Загрузка весов модели
        nn_model.load_state_dict(model_weights)  # Применение весов к модели


        t_nograd,lossname = evaluate_model(nn_model, optim_name, X_test, y_test)
        _, predicted = torch.max(t_nograd, 1)  # Получение предсказаний

        

        pred.append({'coin': coin, 'optim_name': optim_name, 'predicted': t_nograd.cpu().numpy()})

    # data_model = {'pred':pred,'X_train':X_train,'y_train':y_train,'method_preparation':method_preparation}

    flag = True    
    if flag:
        return (pred,X_test.cpu().numpy(), y_test.cpu().numpy())
    return pred

