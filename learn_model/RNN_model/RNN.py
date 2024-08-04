import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from learn_model.assessment_charms.main import filter_pred
from torch.utils.data import DataLoader, TensorDataset

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        # Инициализация скрытого состояния LSTM
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size).to(input_seq.device)
        
        # Передача входных данных через LSTM
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        
        # Применение линейного слоя к последнему временному шагу LSTM для получения предсказаний
        predictions = self.fc(lstm_out[:, -1, :])
        
        return predictions

    

def torch_losses(X_train, y_train,X_test,y_test, model, optim_cls, optim_params,optim_name, num_epochs=100,batch_size=128):


    criterion = nn.MSELoss()
    optimizer = optim_cls(model.parameters(), **optim_params)
    losses = []
    
     # Создание загрузчиков данных для батчей
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    
    real = np.argmax(y_test[:, -1, :].cpu().numpy(), axis=1)
    for epoch in tqdm(range(num_epochs), desc=f"Learning {optim_name}",leave=False):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_X)

            batch_y_last = batch_y[:, -1, :] 

            loss = criterion(y_pred, batch_y_last)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(train_loader))

        if epoch%50==0:
            pred_model,loss_name = evaluate_model(model,optim_name,X_test,y_test)
            pred_model = pred_model.cpu().numpy()
            
            one_h = np.argmax(pred_model, axis=1)
            acc1= accuracy_score(real, one_h)

            new_pred,new_real = filter_pred(pred_model,real,threshold=0.67)
            acc2=accuracy_score(new_real,new_pred)
            print(f'{epoch} epoch accuracy_score : {acc1:.5f} ; new accuracy_score : {acc2:.5f}')

    return losses

# Функция для тестирования модели
def evaluate_model(model,nameModel, X_test, y_test):
    model.eval()  # Переключение модели в режим оценки
    with torch.no_grad():  # Отключение градиентов
        y_pred = model(X_test)


        y_test_last = y_test[:, -1, :]  # Выбираем последний временной шаг

        loss_name = 'MSEloss'
        criterion = nn.MSELoss()
        loss = criterion(y_pred, y_test_last)
        print(f'Test Loss: {loss.item()} {nameModel}')
    return (y_pred,loss_name)