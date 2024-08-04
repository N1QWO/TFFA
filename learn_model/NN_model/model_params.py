from torch import optim
     

def model_params(shape):
    # res = {'in_features': shape[1], 'h1': shape[1]*2, 'h2': 512,'h3': 32, 'out': 2}
    # 64,128,32,128,32,64,64,128,32,128,32,128,64,128,32,128,32,64,64,128,32,128,32,64,64,128,32,128,32,64,64,128,32,128,32,128,64,128,32,128,32,64,64,128,32,128,32,16
    # 64,128,32,128,32,64,64,128,32,128,32,128,64,128,32,128,32,64,64,128,32,128,32,16
    # 64,128,32,128,32,64,64,128,32,128,32,16
    #29, 64, 256, 8, 400, 64, 128, 32, 128, 32, 64,2
    #29, 64, 256, 8, 400, 64, 128, 32, 128, 32, 64
    #32, 64, 256, 8, 400, 64, 128, 32, 128, 32, 64
    res = {'in_features': shape[1], 'h': [32, 64, 32, 64, 128, 64, 128, 32, 128, 32, 64] , 'out': 2}
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
    return 30000
def get_batch_size():
    return 256
def get_negative_slope():
    return 0.1

def get_dropout():
    return 0.8

def get_filter():
    return 'all'
# 200-300
def get_filters():
    return ['all','scrollmean', 'scrollmean_up', 'scrollmean_top', 'scrollmean_low','rsi_oversold','rsi_overbought','macd_crossover' 'Head_and_Shoulders', 'Flag', 'Bullish_Flag', 'Ascending_Triangle','Falling_Wedge']