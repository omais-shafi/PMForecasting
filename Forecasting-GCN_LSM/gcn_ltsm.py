import stellargraph as sg
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from stellargraph.layer import GCN_LSTM

import datetime
import pickle
import random
random.seed(10)

adj = np.loadtxt('Data/adj_first8.txt', dtype=int)
grid_points = pd.read_csv('Data/grid_points_25.csv')
n_grid_points = grid_points.shape[0] 
grid_points = grid_points.pivot_table(index=['lat_grid', 'long_grid'])
grid_points_indices = grid_points.index

seq_len = 6
pre_len = 1
        
def train_test_split(df):
#     random.seed(10)
    time_len = df.shape[1]
    #testing on 6 windows of last day
    test_size = 6+seq_len
    train_size = time_len-6
    train_data = df.iloc[:, :train_size]
    train_data = train_data.reindex(grid_points_indices,fill_value=0)
    train_data = np.array(train_data)
    test_data = df.iloc[:, -test_size:]
    test_data = test_data.reindex(grid_points_indices,fill_value=0)
    test_data = np.array(test_data)
    return train_data, test_data

def scale_data(train_data, test_data):
    max_pm = train_data.max()
    min_pm = train_data.min()
    train_scaled = (train_data - min_pm) / (max_pm - min_pm)
    test_scaled = (test_data - min_pm) / (max_pm - min_pm)
    return train_scaled, test_scaled

def sequence_data_preparation(seq_len, pre_len, train_data, test_data):
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(train_data.shape[1] - int(seq_len + pre_len - 1)):
        a = train_data[:, i : i + seq_len + pre_len]
        trainX.append(a[:, :seq_len])
        trainY.append(a[:, -1])
    for i in range(test_data.shape[1] - int(seq_len + pre_len - 1)):
        b = test_data[:, i : i + seq_len + pre_len]
        testX.append(b[:, :seq_len])
        testY.append(b[:, -1])
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)
    return trainX, trainY, testX, testY

def my_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()  
    loss = mse(y_pred[y_true>0], y_true[y_true>0] )
    return loss

def prepare_train_data(df):
    train_data, test_data = train_test_split(df)
    # print("Train data: ", train_data.shape)
    # print("Test data: ", test_data.shape)
    train_scaled, test_scaled = scale_data(train_data, test_data)
    trainX, trainY, testX, testY = sequence_data_preparation(
        seq_len, pre_len, train_scaled, test_scaled
    )
    # print('trainX: ', trainX.shape)
    # print('trainY: ', trainY.shape)
    # print('testX: ', testX.shape)
    # print('testY: ', testY.shape)
    return trainX, trainY, testX, testY, train_data
  
from timeit import default_timer as timer
class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)
        
def train_model(model, n_epochs, n_batch, trainX, trainY, testX, testY, train_data, save_path = None):
    cb = TimingCallback()
    net = model
    history = net.fit(
            trainX,
            trainY,
            epochs = n_epochs,
            batch_size = n_batch,
            shuffle=True,
            verbose=0,
            validation_data=(testX, testY),
            callbacks=[cb]
        )
    ## Rescale values
    max_pm = train_data.max()
    min_pm = train_data.min()
    train_scaled = (train_data - min_pm) / (max_pm - min_pm)
    df = pd.DataFrame()
    df['Mean Train loss'] = [sum(tf.sqrt(history.history["loss"]).numpy())*max_pm/len(history.history["loss"])]
    df['Last Train loss'] = [tf.sqrt(history.history["loss"][-1]).numpy()*max_pm]
    df['Mean Test loss'] = [sum(tf.sqrt(history.history["val_loss"])).numpy()*max_pm/len(history.history["val_loss"])]
    df['Last Test loss'] = [tf.sqrt(history.history["val_loss"][-1]).numpy()*max_pm]
    df['Train time callbacks'] = [sum(cb.logs)]
    # display(df)
    if save_path is not None:
        with open(save_path, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    # fig = sg.utils.plot_history(history, return_figure=True)
    # if save_path is not None:
    #     fig.savefig(save_path)
    return df

def eval_model(model, testX, testY, train_data, return_rmse = False):
    net = model
    output = net.predict(testX)
    ## actual train and test values
    ## Rescale values
    max_pm = train_data.max()
    min_pm = train_data.min()
    test_true = np.array((testY * (max_pm - min_pm)) + min_pm)
    ## Rescale model predicted values
    test_output = np.array((output * (max_pm - min_pm)) + min_pm)
    # # Masked predicted values
    mask_test = tf.sign(testY)
    # train_rescpred = train_rescpred*(mask_train)
    test_output = test_output*(mask_test)
    test_mse = my_loss(test_true, test_output)
    test_rmse = tf.sqrt(test_mse)
    # print("Test RMSE: ", test_rmse)
    if return_rmse:
        return test_rmse
    return test_output, test_true

def plot_predictions(test_output, test_true, save_path=None):
    ##all test result visualization
    fig = plt.figure(figsize=(15, 8))
    a_pred = test_output[test_true>0]
    a_true = test_true[test_true>0]
    plt.plot(a_pred, "r-", label="prediction")
    plt.plot(a_true, "b-", label="true")
    plt.xlabel("test points")
    plt.ylabel("PM2.5")
    plt.legend(loc="best", fontsize=10)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
        
def define_model(gc_sizes, gc_activations, lstm_sizes, lstm_activations, lr):
    tf.random.set_seed(45)
    gcn_lstm = GCN_LSTM(
            seq_len=seq_len,
            adj=adj,
            gc_layer_sizes = gc_sizes,
            gc_activations = gc_activations,
            lstm_layer_sizes = lstm_sizes,
            lstm_activations = lstm_activations,
        )
    x_input, x_output = gcn_lstm.in_out_tensors()
    model = Model(inputs=x_input, outputs=x_output)
    opt = tf.optimizers.Adam(learning_rate = lr)
    model.compile(optimizer=opt, loss=my_loss)
    # print(model.summary())
    return model