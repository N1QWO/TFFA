
from learn_model.NN_model.model_params import model_params
from datetime import datetime
import sys
import os

try:   
    from config.settings import MODEL_REPORT
except ImportError as e:
    print("ImportError:", e)

def get_report(data_report):
    """
    Сохраняет детальный отчет об обучении модели, включающий метод подготовки, оптимизацию и ошибки.
    
    Параметры:
    - data_report: Словарь, содержащий информацию о модели и процессе обучения.
        - method_preparation: Метод подготовки данных.
        - optim_name: Название метода оптимизации.
        - loss_name: Название функции потерь.
        - t_losses: Список значений потерь на тренировочных данных.
        - t_nograd: Список значений потерь без градиента.
        - shape: Форма модели.
        - coin: Название монеты (финансового инструмента).
    
    Функция генерирует строку отчета и сохраняет её в текстовый файл.
    """
    method_preparation = data_report['method_preparation']
    optim_name = data_report['optim_name']
    loss_name = data_report['loss_name']
    t_losses = ''.join(map(str,data_report['t_losses']))
    t_nograd = ''.join(map(str,data_report['t_nograd']))
    shape = data_report['shape']
    coin = data_report['coin']

    mp = model_params(shape)
    in_features = mp['in_features']
    h = ''.join(map(str,mp['h']))
    out = mp['out']
    now = f'time : {datetime.utcnow()}'
    str_rep  = f'optim_name: {optim_name} , coin : {coin} , method_preparation:{method_preparation} , loss_name: {loss_name} , in_features : {in_features} , layer : {h} , out : {out}'
    loss = f't_losses : {t_losses} , t_nograd : {t_nograd}'
    name = '_'.join(map(str,now.split(' '))).replace(":", "-")
    output_dir = os.path.join(MODEL_REPORT,f"text_report/{optim_name}_{name}.txt")
    all = [now,str_rep,loss]
    with open(output_dir, 'w') as f:
        for entry in all:
            f.write(entry + '\n')
