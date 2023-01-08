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
    # history_path = 'results3/scratch/' + test_date
    history_path = None
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


def exp_grid_size(out_results, df, day):
    out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, day, 6)) 
    out_results = out_results.append(main_forecast(df, [8, 8], ["relu", "relu"], [4], ["tanh"], 0.01, day, 6)) 
    out_results = out_results.append(main_forecast(df, [16, 16], ["relu", "relu"], [4], ["tanh"], 0.01, day, 6)) 
    out_results = out_results.append(main_forecast(df, [32, 32], ["relu", "relu"], [4], ["tanh"], 0.01, day, 6)) 
    out_results = out_results.append(main_forecast(df, [4, 4, 4], ["relu", "relu", "relu"], [4], ["tanh"], 0.01, day, 6)) 
    out_results = out_results.append(main_forecast(df, [4, 8, 16], ["relu", "relu", "relu"], [4], ["tanh"], 0.01, day, 6)) 
    out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [8], ["tanh"], 0.01, day, 6)) 
    out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [16], ["tanh"], 0.01, day, 6)) 
    out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [32], ["tanh"], 0.01, day, 6)) 
    out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4, 4], ["tanh", "tanh"], 0.01, day, 6))
    out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4, 8], ["tanh", "tanh"], 0.01, day, 6))
    return out_results
# out_results = pd.read_csv('out_grid_sizes.csv')
out_results = pd.DataFrame()

df = data_preparation('PM Datasets/2020-11-09_all.csv')
for i in range(10, 16):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '15 Nov') 

df = data_preparation('PM Datasets/2020-11-10_all.csv')
for i in range(11, 17):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '16 Nov') 

df = data_preparation('PM Datasets/2020-11-11_all.csv')
for i in range(12, 18):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '17 Nov') 

df = data_preparation('PM Datasets/2020-11-12_all.csv')
for i in range(13, 19):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '18 Nov')  

df = data_preparation('PM Datasets/2020-11-13_all.csv')
for i in range(14, 20):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '19 Nov')  

df = data_preparation('PM Datasets/2020-11-14_all.csv')
for i in range(15, 21):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '20 Nov')  

df = data_preparation('PM Datasets/2020-11-15_all.csv')
for i in range(16, 22):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '21 Nov')

df = data_preparation('PM Datasets/2020-11-16_all.csv')
for i in range(17, 23):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '22 Nov')

df = data_preparation('PM Datasets/2020-11-17_all.csv')
for i in range(18, 24):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '23 Nov')

df = data_preparation('PM Datasets/2020-11-18_all.csv')
for i in range(19, 25):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '24 Nov')

df = data_preparation('PM Datasets/2020-11-19_all.csv')
for i in range(20, 26):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '25 Nov')

df = data_preparation('PM Datasets/2020-11-20_all.csv')
for i in range(21, 27):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '26 Nov')

df = data_preparation('PM Datasets/2020-11-21_all.csv')
for i in range(22, 28):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '27 Nov')

df = data_preparation('PM Datasets/2020-11-22_all.csv')
for i in range(23, 29):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '28 Nov')

df = data_preparation('PM Datasets/2020-11-23_all.csv')
for i in range(24, 30):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '29 Nov')


df = data_preparation('PM Datasets/2020-12-01_all.csv')
for i in range(2, 8):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '7 Dec')


df = data_preparation('PM Datasets/2020-12-04_all.csv')
for i in range(5, 11):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '10 Dec')  

df = data_preparation('PM Datasets/2020-12-09_all.csv')
for i in range(10, 16):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '15 Dec')  

df = data_preparation('PM Datasets/2020-12-12_all.csv')
for i in range(13, 19):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '18 Dec')  

df = data_preparation('PM Datasets/2020-12-18_all.csv')
for i in range(19, 25):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '24 Dec') 

df = data_preparation('PM Datasets/2020-12-18_all.csv')
for i in range(19, 31):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '30 Dec') 

df = data_preparation('PM Datasets/2021-01-01_all.csv')
for i in range(2, 8):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '7 Jan') 

df = data_preparation('PM Datasets/2021-01-04_all.csv')
for i in range(5, 11):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '10 Jan')  

df = data_preparation('PM Datasets/2021-01-09_all.csv')
for i in range(10, 16):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '15 Jan')  

df = data_preparation('PM Datasets/2021-01-12_all.csv')
for i in range(13, 19):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '18 Jan')  

df = data_preparation('PM Datasets/2021-01-18_all.csv')
for i in range(19, 25):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '24 Jan') 

df = data_preparation('PM Datasets/2021-01-18_all.csv')
for i in range(19, 31):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = exp_grid_size(out_results, df, '30 Jan') 
out_results.to_csv('results3/dump_grid_sizes.csv')


