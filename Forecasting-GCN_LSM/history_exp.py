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





out_results = pd.DataFrame()

df = data_preparation('PM Datasets/2020-11-13_all.csv')
for i in range(14, 16):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Nov', 2))
df = data_preparation('PM Datasets/2021-01-21_all.csv')
for i in range(22, 25):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '24 Jan', 3))
df = data_preparation('PM Datasets/2021-01-20_all.csv')
for i in range(21, 25):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '24 Jan', 4))
df = data_preparation('PM Datasets/2021-01-19_all.csv')
for i in range(20, 25):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '24 Jan', 5))
df = data_preparation('PM Datasets/2020-11-09_all.csv')
for i in range(10, 16):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Nov', 6))
df = data_preparation('PM Datasets/2020-11-08_all.csv')
for i in range(9, 16):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Nov', 7))
df = data_preparation('PM Datasets/2020-11-07_all.csv')
for i in range(8, 16):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Nov', 8))
df = data_preparation('PM Datasets/2020-11-06_all.csv')
for i in range(7, 16):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Nov', 9))
df = data_preparation('PM Datasets/2020-11-05_all.csv')
for i in range(6, 16):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Nov', 10))
df = data_preparation('PM Datasets/2020-11-04_all.csv')
for i in range(5, 16):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Nov', 11))
df = data_preparation('PM Datasets/2020-11-03_all.csv')
for i in range(4, 16):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Nov', 12))
# df = data_preparation('PM Datasets/2020-11-01_all.csv')
# for i in range(2, 16):
#     datafile = 'PM Datasets/2020-11-'
#     datafile +=  str(0) + str(i) if i<10 else str(i)
#     datafile +=  '_all.csv'
#     df1 = data_preparation(datafile)
#     df = pd.concat([df,df1], axis=1)
#     df = df.fillna(0)
# out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Nov', 14))


df = data_preparation('PM Datasets/2020-11-18_all.csv')
for i in range(19, 21):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '20 Nov', 2))
df = data_preparation('PM Datasets/2020-11-17_all.csv')
for i in range(18, 21):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '20 Nov', 3))
df = data_preparation('PM Datasets/2020-11-16_all.csv')
for i in range(17, 21):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '20 Nov', 4))
df = data_preparation('PM Datasets/2020-11-15_all.csv')
for i in range(16, 21):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '20 Nov', 5))
df = data_preparation('PM Datasets/2020-11-14_all.csv')
for i in range(15, 21):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '20 Nov', 6))
df = data_preparation('PM Datasets/2020-11-13_all.csv')
for i in range(14, 21):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '20 Nov', 7))
df = data_preparation('PM Datasets/2020-11-12_all.csv')
for i in range(13, 21):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '20 Nov', 8))
df = data_preparation('PM Datasets/2020-11-11_all.csv')
for i in range(12, 21):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '20 Nov', 9))
df = data_preparation('PM Datasets/2020-11-10_all.csv')
for i in range(11, 21):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '20 Nov', 10))
df = data_preparation('PM Datasets/2020-11-09_all.csv')
for i in range(10, 21):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '20 Nov', 11))
df = data_preparation('PM Datasets/2020-11-08_all.csv')
for i in range(9, 21):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '20 Nov', 12))
# df = data_preparation('PM Datasets/2020-11-06_all.csv')
# for i in range(7, 21):
#     datafile = 'PM Datasets/2020-11-'
#     datafile +=  str(0) + str(i) if i<10 else str(i)
#     datafile +=  '_all.csv'
#     df1 = data_preparation(datafile)
#     df = pd.concat([df,df1], axis=1)
#     df = df.fillna(0)
# out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '20 Nov', 14))


# df = data_preparation('PM Datasets/2020-11-27_all.csv')
# for i in range(28, 29):
#     datafile = 'PM Datasets/2020-11-'
#     datafile +=  str(0) + str(i) if i<10 else str(i)
#     datafile +=  '_all.csv'
#     df1 = data_preparation(datafile)
#     df = pd.concat([df,df1], axis=1)
#     df = df.fillna(0)
# out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '28 Nov', 1))
df = data_preparation('PM Datasets/2020-11-26_all.csv')
for i in range(27, 29):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '28 Nov', 2))
df = data_preparation('PM Datasets/2020-11-25_all.csv')
for i in range(26, 29):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '28 Nov', 3))
df = data_preparation('PM Datasets/2020-11-24_all.csv')
for i in range(25, 29):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '28 Nov', 4))
df = data_preparation('PM Datasets/2020-11-23_all.csv')
for i in range(24, 29):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '28 Nov', 5))
df = data_preparation('PM Datasets/2020-11-22_all.csv')
for i in range(23, 29):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '28 Nov', 6))
df = data_preparation('PM Datasets/2020-11-21_all.csv')
for i in range(22, 29):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '28 Nov', 7))
df = data_preparation('PM Datasets/2020-11-20_all.csv')
for i in range(21, 29):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '28 Nov', 8))
df = data_preparation('PM Datasets/2020-11-19_all.csv')
for i in range(20, 29):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '28 Nov', 9))
df = data_preparation('PM Datasets/2020-11-18_all.csv')
for i in range(19, 29):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '28 Nov', 10))
df = data_preparation('PM Datasets/2020-11-17_all.csv')
for i in range(18, 29):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '28 Nov', 11))
df = data_preparation('PM Datasets/2020-11-16_all.csv')
for i in range(17, 29):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '28 Nov', 12))
# df = data_preparation('PM Datasets/2020-11-14_all.csv')
# for i in range(15, 29):
#     datafile = 'PM Datasets/2020-11-'
#     datafile +=  str(0) + str(i) if i<10 else str(i)
#     datafile +=  '_all.csv'
#     df1 = data_preparation(datafile)
#     df = pd.concat([df,df1], axis=1)
#     df = df.fillna(0)
# out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '28 Nov', 14))


df = data_preparation('PM Datasets/2020-12-05_all.csv')
for i in range(6, 8):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Dec', 2)) 
df = data_preparation('PM Datasets/2020-12-04_all.csv')
for i in range(5, 8):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Dec', 3)) 
df = data_preparation('PM Datasets/2020-12-03_all.csv')
for i in range(4, 8):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Dec', 4)) 
df = data_preparation('PM Datasets/2020-12-02_all.csv')
for i in range(3, 8):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Dec', 5)) 
df = data_preparation('PM Datasets/2020-12-01_all.csv')
for i in range(2, 8):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Dec', 6)) 
df = data_preparation('PM Datasets/2020-11-30_all.csv')
for i in range(1, 8):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Dec', 7)) 
df = data_preparation('PM Datasets/2020-11-29_all.csv')
for i in range(30, 31):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
for i in range(1, 8):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Dec', 8)) 
df = data_preparation('PM Datasets/2020-11-28_all.csv')
for i in range(29, 31):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
for i in range(1, 8):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Dec', 9)) 
df = data_preparation('PM Datasets/2020-11-27_all.csv')
for i in range(28, 31):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
for i in range(1, 8):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Dec', 10)) 
df = data_preparation('PM Datasets/2020-11-26_all.csv')
for i in range(27, 31):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
for i in range(1, 8):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Dec', 11)) 
df = data_preparation('PM Datasets/2020-11-25_all.csv')
for i in range(26, 31):
    datafile = 'PM Datasets/2020-11-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
for i in range(1, 8):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Dec', 12)) 
# df = data_preparation('PM Datasets/2020-11-23_all.csv')
# for i in range(24, 31):
#     datafile = 'PM Datasets/2020-11-'
#     datafile +=  str(0) + str(i) if i<10 else str(i)
#     datafile +=  '_all.csv'
#     df1 = data_preparation(datafile)
#     df = pd.concat([df,df1], axis=1)
# for i in range(1, 8):
#     datafile = 'PM Datasets/2020-12-'
#     datafile +=  str(0) + str(i) if i<10 else str(i)
#     datafile +=  '_all.csv'
#     df1 = data_preparation(datafile)
#     df = pd.concat([df,df1], axis=1)
# df = df.fillna(0)
# out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Dec', 14)) 


df = data_preparation('PM Datasets/2020-12-13_all.csv')
for i in range(14, 16):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Dec', 2)) 
df = data_preparation('PM Datasets/2020-12-12_all.csv')
for i in range(13, 16):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Dec', 3)) 
df = data_preparation('PM Datasets/2020-12-11_all.csv')
for i in range(12, 16):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Dec', 4)) 
df = data_preparation('PM Datasets/2020-12-10_all.csv')
for i in range(11, 16):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Dec', 5)) 
df = data_preparation('PM Datasets/2020-12-09_all.csv')
for i in range(10, 16):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Dec', 6)) 
df = data_preparation('PM Datasets/2020-12-08_all.csv')
for i in range(9, 16):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Dec', 7)) 
df = data_preparation('PM Datasets/2020-12-07_all.csv')
for i in range(8, 16):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Dec', 8)) 
df = data_preparation('PM Datasets/2020-12-06_all.csv')
for i in range(7, 16):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Dec', 9)) 
df = data_preparation('PM Datasets/2020-12-05_all.csv')
for i in range(6, 16):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Dec', 10)) 
df = data_preparation('PM Datasets/2020-12-04_all.csv')
for i in range(5, 16):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Dec', 11)) 
df = data_preparation('PM Datasets/2020-12-03_all.csv')
for i in range(4, 16):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Dec', 12)) 
# df = data_preparation('PM Datasets/2020-12-01_all.csv')
# for i in range(2, 16):
#     datafile = 'PM Datasets/2020-12-'
#     datafile +=  str(0) + str(i) if i<10 else str(i)
#     datafile +=  '_all.csv'
#     df1 = data_preparation(datafile)
#     df = pd.concat([df,df1], axis=1)
# df = df.fillna(0)
# out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '15 Dec', 14)) 



df = data_preparation('PM Datasets/2021-01-05_all.csv')
for i in range(6, 8):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Jan', 2))
df = data_preparation('PM Datasets/2021-01-04_all.csv')
for i in range(5, 8):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Jan', 3))
df = data_preparation('PM Datasets/2021-01-03_all.csv')
for i in range(4, 8):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Jan', 4))
df = data_preparation('PM Datasets/2021-01-02_all.csv')
for i in range(3, 8):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Jan', 5))
df = data_preparation('PM Datasets/2021-01-01_all.csv')
for i in range(2, 8):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Jan', 6))
df = data_preparation('PM Datasets/2020-12-31_all.csv')
for i in range(1, 8):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Jan', 7))
df = data_preparation('PM Datasets/2020-12-30_all.csv')
for i in range(31, 32):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
for i in range(1, 8):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Jan', 8))
df = data_preparation('PM Datasets/2020-12-29_all.csv')
for i in range(30, 32):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
for i in range(1, 8):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Jan', 9))
df = data_preparation('PM Datasets/2020-12-28_all.csv')
for i in range(29, 32):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
for i in range(1, 8):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Jan', 10))
df = data_preparation('PM Datasets/2020-12-27_all.csv')
for i in range(28, 32):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
for i in range(1, 8):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Jan', 11))
df = data_preparation('PM Datasets/2020-12-26_all.csv')
for i in range(27, 32):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
for i in range(1, 8):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Jan', 12))
# df = data_preparation('PM Datasets/2020-12-24_all.csv')
# for i in range(25, 32):
#     datafile = 'PM Datasets/2020-12-'
#     datafile +=  str(0) + str(i) if i<10 else str(i)
#     datafile +=  '_all.csv'
#     df1 = data_preparation(datafile)
#     df = pd.concat([df,df1], axis=1)
#     df = df.fillna(0)
# for i in range(1, 8):
#     datafile = 'PM Datasets/2021-01-'
#     datafile +=  str(0) + str(i) if i<10 else str(i)
#     datafile +=  '_all.csv'
#     df1 = data_preparation(datafile)
#     df = pd.concat([df,df1], axis=1)
#     df = df.fillna(0)
# out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '7 Jan', 14))


df = data_preparation('PM Datasets/2021-01-08_all.csv')
for i in range(9, 11):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '10 Jan', 2))
df = data_preparation('PM Datasets/2021-01-07_all.csv')
for i in range(8, 11):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '10 Jan', 3))
df = data_preparation('PM Datasets/2021-01-06_all.csv')
for i in range(7, 11):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '10 Jan', 4))
df = data_preparation('PM Datasets/2021-01-05_all.csv')
for i in range(6, 11):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '10 Jan', 5))
df = data_preparation('PM Datasets/2021-01-04_all.csv')
for i in range(5, 11):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '10 Jan', 6))
df = data_preparation('PM Datasets/2021-01-03_all.csv')
for i in range(4, 11):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '10 Jan', 7))
df = data_preparation('PM Datasets/2021-01-02_all.csv')
for i in range(3, 11):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '10 Jan', 8))
df = data_preparation('PM Datasets/2021-01-01_all.csv')
for i in range(2, 11):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '10 Jan', 9))
df = data_preparation('PM Datasets/2020-12-31_all.csv')
for i in range(1, 11):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '10 Jan', 10))
df = data_preparation('PM Datasets/2020-12-30_all.csv')
for i in range(31, 32):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
for i in range(1, 11):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '10 Jan', 11))
df = data_preparation('PM Datasets/2020-12-29_all.csv')
for i in range(30, 32):
    datafile = 'PM Datasets/2020-12-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
for i in range(1, 11):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '10 Jan', 12))
# df = data_preparation('PM Datasets/2020-12-27_all.csv')
# for i in range(28, 32):
#     datafile = 'PM Datasets/2020-12-'
#     datafile +=  str(0) + str(i) if i<10 else str(i)
#     datafile +=  '_all.csv'
#     df1 = data_preparation(datafile)
#     df = pd.concat([df,df1], axis=1)
#     df = df.fillna(0)
# for i in range(1, 11):
#     datafile = 'PM Datasets/2021-01-'
#     datafile +=  str(0) + str(i) if i<10 else str(i)
#     datafile +=  '_all.csv'
#     df1 = data_preparation(datafile)
#     df = pd.concat([df,df1], axis=1)
#     df = df.fillna(0)
# out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '10 Jan', 14))


df = data_preparation('PM Datasets/2021-01-22_all.csv')
for i in range(23, 25):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '24 Jan', 2))
df = data_preparation('PM Datasets/2021-01-21_all.csv')
for i in range(22, 25):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '24 Jan', 3))
df = data_preparation('PM Datasets/2021-01-20_all.csv')
for i in range(21, 25):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '24 Jan', 4))
df = data_preparation('PM Datasets/2021-01-19_all.csv')
for i in range(20, 25):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '24 Jan', 5))
df = data_preparation('PM Datasets/2021-01-18_all.csv')
for i in range(19, 25):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '24 Jan', 6))
df = data_preparation('PM Datasets/2021-01-17_all.csv')
for i in range(18, 25):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '24 Jan', 7))
df = data_preparation('PM Datasets/2021-01-16_all.csv')
for i in range(17, 25):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '24 Jan', 8))
df = data_preparation('PM Datasets/2021-01-15_all.csv')
for i in range(16, 25):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '24 Jan', 9))
df = data_preparation('PM Datasets/2020-12-14_all.csv')
for i in range(15, 25):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '24 Jan', 10))
df = data_preparation('PM Datasets/2020-12-13_all.csv')
for i in range(14, 25):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '24 Jan', 11))
df = data_preparation('PM Datasets/2020-12-12_all.csv')
for i in range(13, 25):
    datafile = 'PM Datasets/2021-01-'
    datafile +=  str(0) + str(i) if i<10 else str(i)
    datafile +=  '_all.csv'
    df1 = data_preparation(datafile)
    df = pd.concat([df,df1], axis=1)
    df = df.fillna(0)
out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '24 Jan', 12))
# df = data_preparation('PM Datasets/2021-01-10_all.csv')
# for i in range(11, 25):
#     datafile = 'PM Datasets/2021-01-'
#     datafile +=  str(0) + str(i) if i<10 else str(i)
#     datafile +=  '_all.csv'
#     df1 = data_preparation(datafile)
#     df = pd.concat([df,df1], axis=1)
#     df = df.fillna(0)
# out_results = out_results.append(main_forecast(df, [4, 4], ["relu", "relu"], [4], ["tanh"], 0.01, '24 Jan', 14))

out_results.to_csv('results3/dump_history.csv')