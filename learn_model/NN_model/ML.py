import torch
import torch.nn as nn
from tqdm import tqdm
from learn_model.NN_model.model_params import get_negative_slope, get_dropout
from sklearn.metrics import accuracy_score
import numpy as np
import os
import glob
import shutil

try:
    from learn_model.assessment_charms.main import filter_pred
    from config.settings import MODEL_WEIGHT, MODEL_WEIGHT_ELIT
    from learn_model.NN_model.model_params import get_filter

except ImportError as e:
    print("ImportError:", e)



class TorchNN(nn.Module):
    def __init__(self, in_features, h, out):
        super(TorchNN, self).__init__()

        Layers = []
        Layers.append(nn.Linear(in_features, h[0]))
        Layers.append(nn.BatchNorm1d(h[0]))
        Layers.append(nn.LeakyReLU(negative_slope=get_negative_slope()))

        for i in range(1, len(h)):
            Layers.append(nn.Linear(h[i - 1], h[i]))
            Layers.append(nn.BatchNorm1d(h[i]))
            Layers.append(nn.LeakyReLU(negative_slope=get_negative_slope()))
            if i % 2 == 0:
                Layers.append(nn.Dropout(get_dropout()))

        Layers.append(nn.Linear(h[-1], out))
        Layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*Layers)

    def forward(self, x):
        pred = self.network(x)
        return pred

    def torch_losses(self, X_train, y_train, X_test, y_test,
                     optim_cls, optim_params, optim_name,
                     batch_size=64,
                     num_epochs=100,
                     info_learn=False, coin='',
                     method_preparation='',
                     manage_weight=True):

        criterion = nn.MSELoss()
        optimizer = optim_cls(self.parameters(), **optim_params)
        losses = []

        num_samples = len(X_train)
        num_batches = (num_samples + batch_size - 1) // batch_size  # округление вверх

        real = np.argmax(y_test.cpu().numpy(), axis=1)
        file_path = os.path.join(MODEL_WEIGHT, f'weights_{coin}', 'filter_model_weight', get_filter(), optim_name)
        os.makedirs(file_path, exist_ok=True)

        for epoch in tqdm(range(num_epochs), desc=f"Learning {optim_name}"):
            epoch_loss = 0.0
            self.train()

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)

                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]

                y_pred = self.forward(X_batch)
                loss = criterion(y_pred, y_batch)

                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss /= num_batches

            if info_learn:
                pred_model, loss_name = self.evaluate_model(X_test, y_test,optim_name)
                pred_model = pred_model.cpu().numpy()

                one_h = np.argmax(pred_model, axis=1)
                acc1 = accuracy_score(real, one_h)

                new_pred, new_real = filter_pred(pred_model, real, threshold=0.65)
                acc2 = 0
                if len(new_pred):
                    acc2 = accuracy_score(new_real, new_pred)
                
                # print(f'{epoch} epoch accuracy_score : {acc1:.5f} ; new accuracy_score : {acc2:.5f}')

                destination_file = os.path.join(file_path,
                                                f'model_weights_{coin}_{optim_name}_{method_preparation}_{acc1:.5f}_{acc2:.5f}.pth')
                
                self.save_model_weights(destination_file)

            losses.append(epoch_loss)

        # Удаление старых файлов
        files_list = glob.glob(os.path.join(file_path, '*'))
        files_list = sorted(files_list)
        for i in range(len(files_list) - 10):
            file = files_list[i]
            try:
                os.remove(file)
            except Exception as e:
                print(f'Error deleting file {file}: {e}')

        # Копирование весов, если управление весами включено
        if manage_weight:
            source_file = files_list[-1]
            destination_folder = os.path.join(MODEL_WEIGHT_ELIT, f'weights_{coin}', 'filter_model_weight', get_filter(),
                                              optim_name)
            os.makedirs(destination_folder, exist_ok=True)
            destination_file_path = os.path.join(destination_folder, os.path.basename(source_file))

            try:
                shutil.copy(source_file, destination_file_path)

            except Exception as e:
                print(f'Error copying file: {e}')

        return losses

    def evaluate_model(self, X_test, y_test,optim_name):
        criterion = torch.nn.MSELoss()
        self.eval()  # Переводим модель в режим оценки
        with torch.no_grad():  # Отключаем расчет градиентов во время оценки
            y_pred = self(X_test)
            loss = criterion(y_pred, y_test).item()
        # print(f'Test Loss: {loss} {optim_name}')
        return y_pred, "MSE"


    def ready_model(self, X):
        self.eval()  # Переключение модели в режим оценки
        with torch.no_grad():  # Отключение градиентов
            y_pred = self(X)
        return y_pred
    
    def save_model_weights(self,file_path_coin_weight):
        torch.save(self.state_dict(),file_path_coin_weight)

    def load_model_weights(self, load_path):
        self.load_state_dict(torch.load(load_path))