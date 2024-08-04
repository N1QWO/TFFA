from save_in_file import save_in_file



val = [
       {'path':'BTCUSD','symbol':'BTCUSD','day':1,'interval':60,'path_for':'data_train'}
       ]
# {'path':'BTCUSD','symbol':'BTCUSD','day':31*3,'interval':60,'path_for':'data_train'}
# {'path':'TONUSDT','symbol':'TONUSDT','day':31*12,'interval':60,'path_for':'data_train'}
put_in = save_in_file()
for i in range(len(val)):
    param = val[i]
    put_in.SaveBybitData(param['path'],param['symbol'],param['day'],param['interval'],param['path_for'])