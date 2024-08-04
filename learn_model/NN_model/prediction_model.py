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
    from model_params import torch_optim,model_params,model_hours_params
    from ML import TorchNN
    from config.settings import MODEL_WEIGHT,DATA_REAL
    from data_collection.save_in_file import save_in_file
except ImportError as e:
    print("ImportError:", e)


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_data(coin,
              method_preparation,
              update_data=1
              ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    # Список различных оптимизаторов и их параметры
    torch_optimizers = torch_optim()


    # Путь к файлу с данными по монете
    path_coin = os.path.join(DATA_REAL,f'{coin}.npy')

    # Подготовка данных для обучения и тестирования
    prevHours,afterHours = model_hours_params()
    DP = DataPreparation(prevHours,afterHours)
    X_train, X_test, y_train, y_test = [],[],[],[]
    X_train, X_test, y_train, y_test = DP.call_data_preparation(path_coin,method_preparation,coin,update_data=update_data)
    X_train, X_test = FloatTensor(np.array(X_train)).to(device), FloatTensor(np.array(X_test)).to(device)
    y_train, y_test = FloatTensor(np.array(y_train)).to(device), FloatTensor(np.array(y_test)).to(device)

    # Параметры для нейронной сети
    nn_params = model_params(X_train.shape)

    # Список для хранения предсказаний
    pred = []
    
    # Обучение и проверка модели для каждого оптимизатора
    for optim_func, optim_name, optim_params in torch_optimizers:
        nn_model = TorchNN(**nn_params).to(device)  # Создание модели

        path_weights = os.path.join(MODEL_WEIGHT,f'weights_{coin}\model_weights_{coin}_{optim_name}_{method_preparation}.pth') # Путь к весам модели
        nn_model.load_model_weights(path_weights)

        t_nograd,lossname = nn_model.evaluate_model(X_test, y_test,optim_name)


        pred.append({'coin': coin, 'optim_name': optim_name, 'predicted': t_nograd.cpu().numpy()})

    flag = True    
    if flag:
        return (pred,X_test.cpu().numpy(), y_test.cpu().numpy())
    return pred

