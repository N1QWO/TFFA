import torch
from torch import  FloatTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys

dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if dir not in sys.path:
    sys.path.append(dir)
try:  
    from ML import TorchNN # Подставьте свои импорты здесь
    from preparation_data.DataPreparation import DataPreparation  # Подставьте свои импорты здесь
    from model_params import model_params,model_hours_params,torch_optim,get_epoch,get_batch_size,get_filter
    from model_report.model_report import get_report
    from config.settings import DATA_TRAIN,MODEL_WEIGHT
except ImportError as e:
    print("ImportError:", e)

def learnNN(coin, 
             method_preparation,
             update_data= 0,
             write_report= 1,
             show_loss_line= 0,
             ):
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    torch_optimizers = torch_optim()

    plt.figure(figsize=(14, 6))


    #проверка на существование файла для обработки  
    path_coin = os.path.join(DATA_TRAIN,f'{coin}')
    if not os.path.exists(path_coin ):
        os.makedirs(path_coin, exist_ok=True)

    preparation_path = os.path.join(path_coin,f'{get_filter}')

    path_coin = os.path.join(path_coin,f'{coin}.npy')
    if not os.path.exists(path_coin ):
        print(f'{coin} not in database')
        return None
    
    #проверка на существование файла для тренировки 
    if not os.path.exists(preparation_path):
        update_data = 1

    prevHours,afterHours = model_hours_params()
    DP = DataPreparation(prevHours,afterHours)

    X_train, X_test, y_train, y_test = DP.call_data_preparation(path_coin, method_preparation,coin,update_data=update_data)
    X_train, X_test, y_train, y_test = FloatTensor(X_train).to(device), FloatTensor(X_test).to(device), FloatTensor(y_train).to(device), FloatTensor(y_test).to(device)
    shape = X_train.shape
    # Преобразование данных в FloatTensor и LongTensor для меток
    # Преобразование в индексы классов
    # train_dataset = TensorDataset(X_train, y_train)
    # test_dataset = TensorDataset(X_test, y_test)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    nn_params = model_params(X_train.shape)
    pred = []
    for optim_cls, optim_name, optim_params in torch_optimizers:
        nn_model = TorchNN(**nn_params).to(device)

        t_losses = nn_model.torch_losses(X_train, y_train,X_test,y_test,
                                optim_cls, optim_params,optim_name,
                                batch_size =get_batch_size(), 
                                num_epochs=get_epoch(),
                                info_learn = True,
                                coin = coin, 
                                method_preparation = method_preparation)
        t_nograd,loss_name = nn_model.evaluate_model(X_test, y_test,optim_name)

        if write_report:
            data_report = {'t_losses':t_losses,'t_nograd':t_nograd,'method_preparation':method_preparation,'optim_name':optim_name,'loss_name':loss_name,'shape' : shape , 'coin' : coin}
            get_report(data_report)
        pred.append({'coin': coin, 'optim_name': optim_name, 'predicted': t_nograd.cpu().numpy()})

        if show_loss_line:
            plt.plot(range(len(t_losses)), t_losses, label=f'{optim_name}')
            
        file_path = os.path.join(MODEL_WEIGHT,f'weights_{coin}')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path_coin_weight = os.path.join(file_path,f'model_weights_{coin}_{optim_name}_{method_preparation}.pth')
        
        nn_model.save_model_weights(file_path_coin_weight)
    
    if show_loss_line:
        plt.title('Pytorch optimizers comparison on train data')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    return (pred,X_test.cpu().numpy(), y_test.cpu().numpy())


# Пример вызова функции
# nnWeight('BTCUSD', 'dp')
