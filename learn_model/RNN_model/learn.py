import torch
from torch import  FloatTensor
import matplotlib.pyplot as plt
import os
import sys

dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if dir not in sys.path:
    sys.path.append(dir)
try:  
    from RNN import LSTM, torch_losses, evaluate_model 
    from preparation_data.DataPreparation import DataPreparation  
    from model_params_rnn import model_params_rnn,model_hours_params,torch_optim,get_epoch
    from model_report.model_report import get_report
    from config.settings import DATA_TRAIN,MODEL_WEIGHT
except ImportError as e:
    print("ImportError:", e)

def learn(coin, method_preparation,update_data=1,write_report=0,show_loss_line=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    torch_optimizers = torch_optim()

    plt.figure(figsize=(14, 6))

    path_coin = os.path.join(DATA_TRAIN,f'{coin}.npy')
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

    nn_params = model_params_rnn(X_train.shape)
    pred = []
    for optim_cls, optim_name, optim_params in torch_optimizers:
        nn_model = LSTM(**nn_params).to(device)

        # t_losses = torch_losses((X_train, y_train), nn_model, optim_cls, optim_params,num_epochs=1000)
        t_losses = torch_losses(X_train, y_train,X_test,y_test, nn_model, optim_cls, optim_params,optim_name,num_epochs=get_epoch())
        t_nograd,loss_name = evaluate_model(nn_model, optim_name, X_test, y_test)

        if write_report:
            data_report = {'t_losses':t_losses,'t_nograd':t_nograd,'method_preparation':method_preparation,'optim_name':optim_name,'loss_name':loss_name,'shape' : shape , 'coin' : coin}
            get_report(data_report)
        pred.append({'coin': coin, 'optim_name': optim_name, 'predicted': t_nograd.cpu().numpy()})
        if show_loss_line:
            plt.plot(range(len(t_losses)), t_losses, label=f'{optim_name}')
        file_path = os.path.join(MODEL_WEIGHT,f'weights_rnn_{coin}')
        if not os.path.exists(file_path):
            # Если папка не существует, создаем её
            os.makedirs(file_path)
        file_path_coin_weight = os.path.join(file_path,f'model_weights_rnn_{coin}_{optim_name}_{method_preparation}.pth')
        torch.save(nn_model.state_dict(),file_path_coin_weight)
    if show_loss_line:
        plt.title('Pytorch optimizers comparison on train data')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        
    return (pred,X_test.cpu().numpy(), y_test.cpu().numpy())