# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 21:20:27 2021

@author: stu
"""


import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib 
from tqdm import tqdm
from sklearn import preprocessing

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence as seq

import warnings

warnings.filterwarnings('ignore')


def dataProcess(path, code, columns):
    
    files = os.listdir(path)
    
    data_pck = pd.DataFrame(data=None)
    for file in tqdm(files[:20]):
        data = pd.read_csv(path+file, usecols=columns)
        data = data[data['WindCode']==code]
        data_pck = pd.concat([data, data_pck], ignore_index=True, axis=0)
        
    data_pck = data_pck[data_pck['Time'] > 93000000]
    data_pck = data_pck[data_pck['Time'] < 145700000]
    
    data_pck['label'] = data_pck.groupby('TradingDay')['Match'].apply(lambda x: x.shift(-1) - x).dropna() #label
    
    data_pck['Volume'] = data_pck.groupby('TradingDay')['Volume'].apply(lambda x: x - x.shift(1)).dropna() #Volume
    data_pck = data_pck.dropna()
   
    data_pck['label'] = (np.sign(data_pck['label']) + 1)/2 #定义标签 下一时刻大于当前时刻 定义为1上涨  0下跌
    
    data_pck = data_pck.sort_values(by=['TradingDay', 'Time']).reset_index(drop=True).drop(columns=['WindCode','HighLimited', 'LowLimited'])
    # columns = data_pck.columns
    # maxmin = preprocessing.MinMaxScaler()
    # data_pck = maxmin.fit_transform(data_pck)
    # data_pck = pd.DataFrame(data_pck, columns=columns)
    data_pck.to_pickle('/Users/stu/price formation/data_lstm/'+code+'.pkl')
    

def prepareData(path, code, lookback):
    """
    将数据转化为矩阵
    :param path_data: 文件存储的路径 按照股票存储
    :param length: 回看长度
    :return: x, y
    """
    data = pd.read_pickle(path_data)
    x_data = []
    y_data = []
    
    tradeDays = list(set(data['TradingDay'].values))
    tradeDays.sort()
    for tradeDay in tqdm(tradeDays):
        tmp = data[data['TradingDay']==tradeDay].sort_values(by=['Time']).drop(columns=['TradingDay', 'Time'])
        
        for i in range(lookback, tmp.shape[0]):
            if tmp.iloc[i,-1] != 0.5:
                
                x_data.append(tmp.iloc[i-lookback:i, :-1].values)
                y_data.append(int(tmp.iloc[i-1, -1]))
            
    return x_data, y_data

if __name__ == "__main__":
    
    path = '/Users/stu/price formation/tickData/'
    stock_list = pd.read_csv('/Users/stu/price formation/300index_stock_list.csv',index_col=0)
    code_list = stock_list['stock_code1912'].tolist() #1912代表19年12月调整后的成分股,2006、2012与此类似
    code_list = [i[:6]+'.SZ' if i[-1]=='E' else i[:6]+'.SH' for i in code_list]
    
    columns = ['WindCode', 'TradingDay', 'Time', 'Match', 'AskPrice1', 'AskPrice2',
               'AskPrice3', 'AskPrice4', 'AskPrice5', 'AskVol1', 'AskVol2', 'AskVol3',
               'AskVol4', 'AskVol5', 'AskVol1', 'AskVol2', 'AskVol3', 'AskVol4',
               'BidPrice1','BidPrice2','BidPrice3','BidPrice4','BidPrice5',
               'AskVol5', 'BidVol1', 'BidVol2', 'BidVol3', 'BidVol4', 'BidVol5',
               'Volume', 'HighLimited', 'LowLimited']
    
    
    # 按股票名称存储个股所有时间的tick数据 存储为pkl文件
    for code in tqdm(code_list[:10]):
        print("当前进度...")
        # code = '000001.SZ' #个股代码
        lookback = 20  #定义回看期长度 当取值为1时 就是当前tick数据预测下一个tick涨跌
        dataProcess(path, code, columns)
    
    
    code = '000001.SZ' #个股代码
    
    code = code_list[8]
    path_data = '/Users/stu/price formation/data_lstm/'+code+'.pkl'
    x_data, y_data = prepareData(path_data, code, lookback)
    
    x_data1 = np.array(x_data)
    y_data1 = np.array(y_data)
    
    a = int(len(x_data1)*0.9)
    train_x,test_x = x_data1[:a],x_data1[a:]
    train_y,test_y = y_data1[:a],y_data1[a:]
    
    model = Sequential()
    model.add(LSTM(128,input_shape=(20, 22)))
    model.add(Dense(1, activation='sigmoid'))
    
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    batch_size =64
    print('Train...')
    model.fit(train_x, train_y,batch_size=batch_size, epochs=5, validation_data=(test_x, test_y))
