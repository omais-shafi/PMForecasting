import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from gcn_ltsm import *
import numpy as np
import pandas as pd
import tensorflow as tf
print("# GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))

def data_preparation(datafile):
    lat_range = {'min': 28.486, 'max': 28.72}
    long_range = {'min': 77.1, 'max': 77.32}
    n_lat_grid = 25
    n_long_grid = 25
    
    #Put the file location
    df = pd.read_csv(datafile)
    #type casting
    df.pm1_0 = df.pm1_0.astype(float)
    df.pm2_5 = df.pm2_5.astype(float)
    df.pm10 = df.pm10.astype(float)
    df.lat = round(round(5*df.lat.astype(float),2)/5.0,3)
    df.long= round(round(5*df.long.astype(float),2)/5.0,3)

    # Ensuring Delhi region and removing outliers from data
    df = df[(df.lat.astype(int) == 28) &(df.long.astype(int) == 77)]
    df = df[(df.pm1_0<=1500) & (df.pm2_5<=1500) & (df.pm10<=1500) & (df.pm1_0>=20) & (df.pm2_5>=30) & (df.pm10>=30)]
    #df = df[(df.humidity<=60)&(df.humidity>=7)]

    df['lat_grid'] = df.apply(lambda row: int((n_lat_grid-1)*(row.lat-lat_range['min'])/(lat_range['max']-lat_range['min'])), axis=1 )
    df['long_grid'] = df.apply(lambda row: int((n_long_grid-1)*(row.long-long_range['min'])/(long_range['max']-long_range['min'])), axis=1 )
    df['lat_grid'] = df['lat_grid'].astype(float).astype(int)
    df['long_grid'] = df['long_grid'].astype(float).astype(int)

    # rounding @180min
    df.dateTime = pd.to_datetime(df.dateTime)
    df.dateTime = df.dateTime.dt.round('180min')
    # use time as a feature as well
    df.dateTime = df.dateTime.dt.hour*60 + df.dateTime.dt.minute
    #taking only data from 6Am-12midnight
    df = df[(df.dateTime>=360)] 
    df = df.pivot_table(index=['lat_grid','long_grid'], columns='dateTime', aggfunc='mean')['pm2_5']
    df = df.fillna(0)
    return df

def main_forecast(df, gc_sizes, gc_activations, lstm_sizes, lstm_activations, lr, test_date, history_days):
    model = define_model(gc_sizes, gc_activations, lstm_sizes, lstm_activations, lr)
    history_path = 'results3/scratch/' + test_date
    trainX, trainY, testX, testY, train_data = prepare_train_data(df)
    out = train_model(model, 100, 19, trainX, trainY, testX, testY, train_data, history_path)
    out['GCN sizes'] = [gc_sizes]
    out['LSTM sizes'] = [lstm_sizes]
    out['History (in days)'] = [history_days]
    out['Test Date'] = [test_date]
    # display(out)
    # test_output, test_true = eval_model(model, testX, testY)
    # plot_predictions(test_output, test_true)
    return out



def main_forecast2(model, df, gc_sizes, gc_activations, lstm_sizes, lstm_activations, lr, test_date, history_days):
    # model = define_model(gc_sizes, gc_activations, lstm_sizes, lstm_activations, lr)
    history_path = 'results3/fine-tune_lr_004/' + test_date
    trainX, trainY, testX, testY, train_data = prepare_train_data(df)
    out = train_model(model, 100, 19, trainX, trainY, testX, testY, train_data, history_path)
    # out['GCN sizes'] = [gc_sizes]
    # out['LSTM sizes'] = [lstm_sizes]
    # out['History (in days)'] = [history_days]
    out['lr'] = lr
    out['Test Date'] = [test_date]
    # display(out)
    # test_output, test_true = eval_model(model, testX, testY)
    # plot_predictions(test_output, test_true)
    return out


#fine-tuning with lr =0.004
df = pd.read_csv('Data/Dummy_data_1week_7Dec.csv')
df = df.set_index(['lat_grid','long_grid'])
out_results = pd.DataFrame
model = define_model([4, 4], ["relu", "relu"], [4], ["tanh"], 0.01)
out_results = main_forecast2(model, df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Dec', 6)

lr = 0.004
df1 = data_preparation('PM Datasets/2020-12-08_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
opt = tf.optimizers.Adam(learning_rate = lr)
model.compile(optimizer=opt, loss=my_loss)
out_results = pd.concat([out_results, main_forecast2(model, df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '8 Dec', 6)])

df1 = data_preparation('PM Datasets/2020-12-09_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
opt = tf.optimizers.Adam(learning_rate = lr)
model.compile(optimizer=opt, loss=my_loss)
out_results = pd.concat([out_results, main_forecast2(model, df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '9 Dec', 6)])

df1 = data_preparation('PM Datasets/2020-12-10_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
opt = tf.optimizers.Adam(learning_rate = lr)
model.compile(optimizer=opt, loss=my_loss)
out_results = pd.concat([out_results, main_forecast2(model, df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '10 Dec', 6)])

df1 = data_preparation('PM Datasets/2020-12-11_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
opt = tf.optimizers.Adam(learning_rate = lr)
model.compile(optimizer=opt, loss=my_loss)
out_results = pd.concat([out_results, main_forecast2(model, df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '11 Dec', 6)])

df1 = data_preparation('PM Datasets/2020-12-12_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
opt = tf.optimizers.Adam(learning_rate = lr)
model.compile(optimizer=opt, loss=my_loss)
out_results = pd.concat([out_results, main_forecast2(model, df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '12 Dec', 6)])

df1 = data_preparation('PM Datasets/2020-12-13_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
opt = tf.optimizers.Adam(learning_rate = lr)
model.compile(optimizer=opt, loss=my_loss)
out_results = pd.concat([out_results, main_forecast2(model, df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '13 Dec', 6)])

df1 = data_preparation('PM Datasets/2020-12-14_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
opt = tf.optimizers.Adam(learning_rate = lr)
model.compile(optimizer=opt, loss=my_loss)
out_results = pd.concat([out_results, main_forecast2(model, df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '14 Dec', 6)])

df1 = data_preparation('PM Datasets/2020-12-15_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
opt = tf.optimizers.Adam(learning_rate = lr)
model.compile(optimizer=opt, loss=my_loss)
out_results = pd.concat([out_results, main_forecast2(model, df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Dec', 6)])

df1 = data_preparation('PM Datasets/2020-12-16_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
opt = tf.optimizers.Adam(learning_rate = lr)
model.compile(optimizer=opt, loss=my_loss)
out_results = pd.concat([out_results, main_forecast2(model, df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '16 Dec', 6)])

out_results.to_csv('results3/fine-tune_lr'+str(lr)+'.csv')




#training from scratch
df = pd.read_csv('Data/Dummy_data_1week_7Dec.csv')
df = df.set_index(['lat_grid','long_grid'])
    
out_results = pd.DataFrame
out_results = main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, 0.86, '7 Dec', 6)

df1 = data_preparation('PM Datasets/2020-12-08_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
out_results = pd.concat([out_results, main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '8 Dec', 6)])

df1 = data_preparation('PM Datasets/2020-12-09_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
out_results = pd.concat([out_results, main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '9 Dec', 6)])

df1 = data_preparation('PM Datasets/2020-12-10_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
out_results = pd.concat([out_results, main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '10 Dec', 6)])

df1 = data_preparation('PM Datasets/2020-12-11_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
out_results = pd.concat([out_results, main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '11 Dec', 6)])

df1 = data_preparation('PM Datasets/2020-12-12_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
out_results = pd.concat([out_results, main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '12 Dec', 6)])

df1 = data_preparation('PM Datasets/2020-12-13_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
out_results = pd.concat([out_results, main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '13 Dec', 6)])

df1 = data_preparation('PM Datasets/2020-12-14_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
out_results = pd.concat([out_results, main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '14 Dec', 6)])

df1 = data_preparation('PM Datasets/2020-12-15_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
out_results = pd.concat([out_results, main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Dec', 6)])

df1 = data_preparation('PM Datasets/2020-12-16_all.csv')
df = pd.concat([df,df1], axis=1).iloc[:, 6:]
df = df.fillna(0)
out_results = pd.concat([out_results, main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '16 Dec', 6)])

out_results.to_csv('results3/scratch.csv')