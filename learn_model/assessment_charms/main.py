import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
import os
import sys
import shutil
from telegram import Bot

if __name__=="__main__":
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
    if dir not in sys.path:
        sys.path.append(dir)

from learn_model.NN_model.model_params import model_hours_params,model_params, get_epoch, get_negative_slope, get_dropout,get_filter

try:
    from config.settings import PREPARATION_DATA, MODEL_WEIGHT,TOKEN,CHAT_ID
except ImportError as e:
    print("ImportError:", e)



async def send_plot_to_telegram(plot_file):
    bot = Bot(token=TOKEN)
    with open(plot_file, 'rb') as photo:
        await bot.send_photo(chat_id=CHAT_ID, photo=photo)

async def graf_show(data, data_y=[], predict=[], optim_name='',check=10,Target=True):
    """
    Отображает графики финансовых данных с опциональными предсказанными и реальными значениями.
    
    Параметры:
    - data: Финансовые данные для отображения.
    - data_y: Реальные целевые значения.
    - predict: Предсказанные значения.
    - optim_name: Название использованного метода оптимизации.
    - check: Количество графиков для отображения.
    - Target: Флаг, указывающий, содержат ли данные целевые значения.
    
    Отображает финансовые данные с дополнительными графиками индикаторов, таких как 
    SMA, RSI, MACD и полосы Боллинджера.
    """    
    
    if len(data_y)==0:
        data_y = [[0,0] for i in range(len(predict))]
    columns =  []
    file_path = os.path.join(PREPARATION_DATA,'data_columns.txt')
    with open(file_path,'r') as f:
        columns = f.readline().split(' ')
    
    if Target:
        data = np.array(data)
        row, col = data.shape
        prev, after = model_hours_params() 

        data = data.reshape(row, prev, col // prev)
        for i in range(check if len(data) > check else len(data)):
            df = pd.DataFrame(data[i, :, :], columns=columns)
            df.index = pd.date_range(start='2022-01-01', periods=len(df), freq='h')
            add_plots = [
                mpf.make_addplot(df['SMA_3'], color='blue', linestyle='-', label='SMA 3'),
                mpf.make_addplot(df['SMA_7'], color='orange', linestyle='-', label='SMA 7'),
                mpf.make_addplot(df['SMA_28'], color='green', linestyle='-', label='SMA 28'),
                mpf.make_addplot(df['Upper_Band'], color='cyan', linestyle='--', label='Upper Band'),
                mpf.make_addplot(df['Lower_Band'], color='cyan', linestyle='--', label='Lower Band'),
                mpf.make_addplot(df['RSI_7'], panel=1, color='purple', secondary_y=False, label='RSI 7'),
                mpf.make_addplot(df['RSI_14'], panel=1, color='brown', secondary_y=False, label='RSI 14'),
                mpf.make_addplot(df['MACD'], panel=2, color='green', secondary_y=False, label='MACD'),
                mpf.make_addplot(df['Signal_line'], panel=2, color='orange', secondary_y=False, label='Signal Line')
            ]

            arm = np.argmax(predict[i])
            fig, axlist = mpf.plot(
                df,
                type='candle',
                style='charles',
                title=f'optim_name : {optim_name} {i} , model predict : {"up {:.4f}".format(predict[i][arm]) if arm == 0 else "down {:.4f}".format(predict[i][arm])} ; real : {"up " if np.argmax(data_y[i]) == 0 else "down"}',
                ylabel='Price',
                volume=True,
                addplot=add_plots,
                figsize=(10, 15),
                returnfig=True,
            )
            for ax in axlist[::2]:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles=handles, labels=labels)

            plt.show()
            plt.close(fig)
    else:
        df = data
        df.index = pd.date_range(start=df.index[0], periods=len(df), freq='h')
        add_plots = [
            mpf.make_addplot(df['SMA_3'], color='blue', linestyle='-', label='SMA 3'),
            mpf.make_addplot(df['SMA_7'], color='orange', linestyle='-', label='SMA 7'),
            mpf.make_addplot(df['SMA_28'], color='green', linestyle='-', label='SMA 28'),
            mpf.make_addplot(df['Upper_Band'], color='cyan', linestyle='--', label='Upper Band'),
            mpf.make_addplot(df['Lower_Band'], color='cyan', linestyle='--', label='Lower Band'),
            mpf.make_addplot(df['RSI_7'], panel=1, color='purple', secondary_y=False, label='RSI 7'),
            mpf.make_addplot(df['RSI_14'], panel=1, color='brown', secondary_y=False, label='RSI 14'),
            mpf.make_addplot(df['MACD'], panel=2, color='green', secondary_y=False, label='MACD'),
            mpf.make_addplot(df['Signal_line'], panel=2, color='orange', secondary_y=False, label='Signal Line')
        ]

        fig, axlist = mpf.plot(
            df,
            type='candle',
            style='charles',
            title=f'',
            ylabel='Price',
            volume=True,
            addplot=add_plots,
            figsize=(20, 15),
            returnfig=True,
        )
        for ax in axlist[::2]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles=handles, labels=labels)

        if not Target:
            # Сохранение графика в файл
            plot_file = f'plot_fragment.png'
            plt.savefig(plot_file)
            plt.close(fig)

            # Отправка файла через Telegram
            await send_plot_to_telegram(plot_file)


            # Удаление временного файла
            os.remove(plot_file)
        else:
            plt.show()
            plt.close(fig)



def best_param_model(name, acc, pre, rec, shape):
    """
    Сохраняет параметры лучшей модели в файл.
    
    Параметры:
    - name: Название модели.
    - acc: Точность модели.
    - pre: Точность предсказаний модели.
    - rec: Полнота предсказаний модели.
    - shape: слои модели.
    
    Сохраняет параметры модели, включая количество эпох, отрицательный наклон(коэффициент LeakyReLU) и дропаут.
    """
    path = f'best/{" ".join(map(str, name))}_{acc:.8f}'
    file_path = os.path.join(MODEL_WEIGHT,path)
    stri = f'{" ".join(map(str, name))}\naccuracy: {acc:.8f}\nprecision: {pre:.8f}\nrecall: {rec:.8f}'
    res = model_params(shape)
    a, b, c = res['in_features'], res['h'], res['out']
    stri2 = f'in_features: {a} h: {" ".join(map(str, b))} out: {c}'
    stri3 = f'epoch : {get_epoch()}\nnegative_slope : {get_negative_slope()}\ndropout : {get_dropout()}'
    all_str = stri + '\n' + stri2 + '\n' + stri3 + '\n'
    with open(file_path, 'w') as f:
        f.write(all_str)

def graf_separation(pred, real):
    """
    Отображает диаграмму, показывающую предсказания модели и их соответствие реальным значениям.
    
    Параметры:
    - pred: Предсказанные значения.
    - real: Реальные значения.
    
    Создает диаграмму с цветными точками: зеленый для правильных предсказаний, красный для неправильных.
    """
    # pred, real = pred[:100], real[:100]
    # Получаем предсказанные метки классов
    one_h = np.argmax(pred, axis=1)
    
    # # Создаем новый массив one-hot на основе порога 0.5
    # new_one_hot_pred = np.where(pred[np.arange(len(pred)), one_h] <= 0.5, (one_h - 1) % 2, one_h)
    
    # Создаем цвета для точек: зеленый для правильных предсказаний, красный для неправильных
    correct = one_h == real
    colors = np.where(correct, 'green', 'red')
    
    plt.figure(figsize=(10, 6))
    
    # Отображение точек для класса 0
    class_0_mask = one_h == 0
    plt.scatter(pred[class_0_mask, 0], np.zeros_like(pred[class_0_mask, 0]), c=colors[class_0_mask], label='Class up', marker='o')
    
    # Отображение точек для класса 1
    class_1_mask = one_h == 1
    plt.scatter(pred[class_1_mask, 1], np.ones_like(pred[class_1_mask, 1]), c=colors[class_1_mask], label='Class 1', marker='^')
    
    plt.axhline(y=0.5, color='grey', linestyle='--')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Class')
    plt.title('Predictions vs Actual Classes')
    plt.yticks([0, 1], ['Class 0', 'Class 1'])
    plt.legend()
    plt.grid(True)
    plt.show()

def pred_prep(pred):
    """
    one-hot performance target of class 
    """
    one_h = np.argmax(pred, axis=1)
    new_one_hot_pred = np.where(pred[np.arange(len(pred)), one_h] <= 0.49, (one_h - 1) % 2, one_h)
    return new_one_hot_pred

def filter_pred(pred,real,threshold=0.75):
    """
    Фильтрует предсказания и реальные значения на основе заданного порога.
    
    Параметры:
    - pred: Предсказанные значения.
    - real: Реальные значения.
    - threshold: Порог для фильтрации.
    
    Возвращает:
    - new_pred: Новые предсказанные значения, отфильтрованные по порогу.
    - new_real: Новые реальные значения, соответствующие отфильтрованным предсказаниям.
    """
    mx_pred = np.max(pred, axis=1)
    indices  = mx_pred>threshold
    new_pred= np.argmax(pred[indices], axis=1)
    new_real = real[indices]
    return (new_pred,new_real)


def func_save_filter_weight(coin,optim_name,method_preparation,acc1,acc2):
    """
    Сохраняет веса отфильтрованной модели в указанный путь.
    
    Параметры:
    - coin: Название монеты.
    - optim_name: Название метода оптимизации.
    - method_preparation: Метод подготовки данных.
    - acc1: Первая метрика точности.
    - acc2: Вторая метрика точности.
    
    Копирует файл весов модели в новое место с добавлением информации о фильтре.
    """
    file_path = os.path.join(MODEL_WEIGHT,f'weights_{coin}/filter_model_weight/{get_filter()}')
    if not os.path.exists(file_path):
            # Если папка не существует, создаем её
        os.makedirs(file_path)
    destination_file  = os.path.join(file_path,f'model_weights_{coin}_{optim_name}_{method_preparation}_{acc1}_{acc2}.pth')
    
    source_file = os.path.join(MODEL_WEIGHT,f'weights_{coin}/model_weights_{coin}_{optim_name}_{method_preparation}.pth')

     
    shutil.copy2(source_file,destination_file)