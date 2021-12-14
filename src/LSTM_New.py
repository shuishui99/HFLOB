# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:43:14 2021

@author: stu
"""
import threading
import _thread
import os

import keras
import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

sd = StandardScaler()
from multiprocessing import pool
from threading import Thread
from multiprocessing import Process

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence as seq

import warnings
warnings.filterwarnings('ignore')

def prepareData(path_data, code,lookback):
    """
    将数据转化为矩阵
    :param path_data: 文件存储的路径 按照股票存储
    :param length: 回看长度
    :return: x, y
    """
    column = ['TradingDay', 'Time', 'Match', 'Volume', 'voi_1', 'voi_2', 'voi_3', 'voi_4', 'voi_5', 'ti_v',
              'relative_spread', 'press', 'Length_imb_1_sum', 'Length_imb_2_sum',
              'Length_imb_3_sum', 'Length_imb_4_sum', 'Length_imb_5_sum', 'label']
    data = pd.read_pickle(path_data).loc[:, column]

    #归一化
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    for col in column[2:-1]:
       data[col] = data[[col]].apply(max_min_scaler)

    x_data = []
    y_data = []

    tradeDays = list(set(data['TradingDay'].values))
    tradeDays.sort()
    #-------------------cross-------------------
    crossTradeDay = tradeDays[:97] + tradeDays[140:]
    tradeDays = crossTradeDay


    for tradeDay in tqdm(tradeDays):
        tmp = data[data['TradingDay'] == tradeDay].sort_values(by=['Time']).drop(columns=['TradingDay', 'Time'])
        for i in range(lookback, tmp.shape[0]):
            if tmp.iloc[i - 1, -1] != 0.5:
                x_data.append(tmp.iloc[i - lookback:i, :-1].values)
                y_data.append(int(tmp.iloc[i - 1, -1]))
    print("x_data")
    return x_data, y_data


def comb_model(code_list, lookback):
    for code in code_list:
        print(code, '---------', code_list.index(code))
        start_time1 = time.time()
        path_data = '/home/yuan/jlw/price_formation/data_pkl/' + code + '.pkl'
        # path_data = '/home/envy/hf/jlw/data_pkl/' + src + '.pkl'
        x_data, y_data = prepareData(path_data, code, lookback)
        print('Data Prepare；', time.time() - start_time1)
        x_data1 = np.array(x_data)
        y_data1 = np.array(y_data)

        a = int(len(x_data1) * 0.7)
        train_x, test_x = x_data1[:a], x_data1[a:]
        train_y, test_y = y_data1[:a], y_data1[a:]

        # LSTM算法
        start_time2 = time.time()
        model = Sequential()
        model.add(LSTM(256, input_shape=(lookback, 15)))
        # model.add(Dense(128))
        # model.add(Dense(64))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        batch_size = 64
        print('Train...')
        model.fit(train_x, train_y, batch_size=batch_size, epochs=20, validation_data=(test_x, test_y))
        print('LSTM:', time.time() - start_time2)
        with open('/home/yuan/jlw/price_formation/result/LSTM_10_scaler_cross.csv', 'a+') as f:
            f.write(code + ',' + str(model.history.history['val_accuracy']))
            f.write('\n')

        # # keras实现LR
        # start_time3 = time.time()
        # LR_test_x = test_x.reshape((-1, lookback * 15))
        # LR_train_x = train_x.reshape((-1, lookback * 15))
        # model_LR = keras.Sequential()
        # from keras import layers
        # model_LR.add(layers.Dense(1, input_dim=(lookback * 15), activation='sigmoid'))
        # model_LR.summary()
        # model_LR.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        # model_LR.fit(LR_train_x, train_y, epochs=20, validation_data=(LR_test_x, test_y))
        # with open('/home/yuan/jlw/price_formation/result/LR_10_scaler.csv', 'a+') as f:
        #     f.write(src + ',' + str(model_LR.history.history['val_acc']))
        #     f.write('\n')
        # f.close()
        # print('LR:', time.time() - start_time3)


if __name__ == "__main__":
    stock_list = pd.read_csv('/home/yuan/jlw/price_formation/300index_stock_list.csv', index_col=0) #经金
    # stock_list = pd.read_csv('/home/envy/hf/jlw/300index_stock_list.csv', index_col=0)   #envy
    code_list = stock_list['stock_code1912'].tolist()  # 1912代表19年12月调整后的成分股,2006、2012与此类似
    code_list = [i[:6] + '.SZ' if i[-1] == 'E' else i[:6] + '.SH' for i in code_list]
    code_re = pd.read_csv('/home/yuan/jlw/price_formation/result/LSTM_10_scaler_cross.csv', header=None).iloc[:, 0].tolist()
    code_list = list(set(code_list) - set(code_re))

    columns = ['WindCode', 'TradingDay', 'Time', 'Match', 'AskPrice1', 'AskPrice2',
               'AskPrice3', 'AskPrice4', 'AskPrice5', 'AskVol1', 'AskVol2', 'AskVol3',
               'AskVol4', 'AskVol5', 'AskVol1', 'AskVol2', 'AskVol3', 'AskVol4',
               'BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5',
               'AskVol5', 'BidVol1', 'BidVol2', 'BidVol3', 'BidVol4', 'BidVol5',
               'Volume', 'HighLimited', 'LowLimited']
    lookback = 10


    # 多线程

    for i in range(21):
        if i == 20:
            t = Process(target=comb_model, args=[code_list[(i+1) * 3:], lookback])
            t.start()
        else:
            t = Process(target=comb_model, args=[code_list[i*3:(1+i)*3], lookback])
            t.start()
        # t = Process(target=comb_model, args=[code_list[i:(1+i)], lookback])
        # t.start()

    # src = '000001.SZ' #个股代码




