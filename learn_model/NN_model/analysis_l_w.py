import torch
import torch.nn as nn
import numpy as np
from torch import FloatTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import os

dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if dir not in sys.path:
    sys.path.append(dir)
dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../'))
if dir not in sys.path:
    sys.path.append(dir)
dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
if dir not in sys.path:
    sys.path.append(dir)
try:
    from preparation_data.DataPreparation import DataPreparation
    from model_params import model_params,model_hours_params
    from ML import TorchNN
    from config.settings import MODEL_WEIGHT,BASE_DIR
    from assessment_charms.main import graf_show
except ImportError as e:
    print("ImportError:", e)


def vizualize_neural(coin,optim_name='Adamax',method_preparation='dp',update_data=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Путь к файлу с данными по монете
    path_coin = os.path.join(BASE_DIR,f'data_real/{coin}.npy')

    # Подготовка данных для обучения и тестирования
    prevHours,afterHours = model_hours_params()
    DP = DataPreparation(prevHours,afterHours)
    X_train, X_test, y_train, y_test = [],[],[],[]
    X_train, X_test, y_train, y_test = DP.call_data_preparation(path_coin,method_preparation,coin,update_data=update_data)
    X_train, X_test = FloatTensor(np.array(X_train)).to(device), FloatTensor(np.array(X_test)).to(device)
    y_train, y_test = FloatTensor(np.array(y_train)).to(device), FloatTensor(np.array(y_test)).to(device)

    random_index = torch.randint(0, len(X_test), (1,)).item()
    input_data = X_test[random_index]
    input_data_y = y_test[random_index]

    path_weights = os.path.join(MODEL_WEIGHT,f'weights_{coin}\model_weights_{coin}_{optim_name}_{method_preparation}.pth') # Путь к весам модели
    
    nn_params=model_params(X_train.shape) 
    model = TorchNN(**nn_params).to(device)
    model.load_state_dict(torch.load(path_weights))

    # Включаем режим оценки (evaluation mode) для модели (важно для Dropout и BatchNorm)
    model.eval()
    pred = []
    # Прямой проход через модель для получения активаций
    with torch.no_grad():
        activations = []
        x = input_data
        for layer in model.network:
            x = layer(x)
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
                activations.append(x.squeeze().cpu().numpy())
        pred =x
        print("pred",x)  
    print("real",input_data_y)    

    num_layers = len(activations)
    num_rows = 2
    num_cols = (num_layers + 1) // num_rows 
    
    

    # Визуализируем активации нейронов для каждого слоя
    plt.figure(figsize=(12, 8))
    for i, activation in enumerate(activations):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.hist(activation, bins=50)
        plt.title(f'Layer {i + 1} Activations')
        plt.xlabel('Activation Value')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    
    input_data = np.array(input_data.cpu())
    input_data = input_data.reshape(1,input_data.shape[0])
    input_data_y = np.array(input_data_y.cpu())
    input_data_y = input_data_y.reshape(1,input_data_y.shape[0])
    pred = np.array(pred.cpu())
    pred = pred.reshape(1,pred.shape[0])

    return (input_data, input_data_y, pred)


def vizualize_weight(coin,method_preparation='dp',optim_name='Adamax'):

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_weights = os.path.join(MODEL_WEIGHT,f'weights_{coin}\model_weights_{coin}_{optim_name}_{method_preparation}.pth') # Путь к весам модели 
    #не хочу вызывать X_train(долго) пока просто напишу shape = [1,390]
    prevHours, afterHours = model_hours_params()
    shape = [1,prevHours*36]
    nn_params=model_params(shape) 
    model = TorchNN(**nn_params).to(device)
    model.load_state_dict(torch.load(path_weights))
    model.eval()

    # Пример кода для анализа весов и смещений каждого слоя
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f'Layer: {name} - Mean: {param.mean().item()}, Std: {param.std().item()}, Min: {param.min().item()}, Max: {param.max().item()}')
        elif 'bias' in name:
            print(f'Layer: {name} - Mean: {param.mean().item()}, Std: {param.std().item()}, Min: {param.min().item()}, Max: {param.max().item()}')


if __name__=="__main__":
    coin='BTCUSD'
    optim_name = 'Adamax'
    vizualize_weight(coin)
    x,y,pred = vizualize_neural(coin)
    graf_show(x,y,pred,optim_name)