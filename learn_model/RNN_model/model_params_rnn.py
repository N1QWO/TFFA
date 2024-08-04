from torch import optim
     

def model_params_rnn(shape):
    # 64,128,32,128,32,64,64,128,32,128,32,16
    #29, 64, 256, 8, 400, 64, 128, 32, 128, 32, 64,2
    #29, 64, 256, 8, 400, 64, 128, 32, 128, 32, 64
    # input_size  Размер входного вектора (например, количество признаков)
    # hidden_size = 50   Размер скрытого состояния
    # output_size = 1    Размер выходного вектора
    # num_layers = 2     Количество слоев LSTM
    res = {'input_size': shape[2], 'hidden_size': 64 , 'output_size': 2,'num_layers':shape[1]}
    return res

def model_hours_params():
    return (10,5)

def torch_optim():
    return [
        # (optim.NAdam, 'NAdam', {'lr': 0.01,'weight_decay': 0}),
        # (optim.SGD, 'SGD', {'lr': 0.001,'weight_decay': 0}),
        # (optim.SGD, 'Momentum', {'lr': 0.001, 'momentum': 0.9,'weight_decay': 0}),
        # (optim.SGD, 'Nesterov Momentum', {'lr': 0.001, 'momentum': 0.9, 'nesterov': True,'weight_decay': 0}),
        # (optim.Adagrad, 'Adagrad', {'lr': 0.01,'weight_decay': 0}),
        # (optim.RMSprop, 'RMSprop', {'lr': 0.01,'weight_decay': 0}),
        # (optim.Adam, 'Adam', {'lr': 0.01,'weight_decay': 0}),
        (optim.Adamax, 'Adamax', {'lr': 0.01,'weight_decay': 0}),
        # (optim.AdamW, 'AdamW', {'lr': 0.01,'weight_decay': 0})
        
    ]

def get_epoch():
    return 1500

def get_negative_slope():
    return 0.1

def get_dropout():
    return 0.5

