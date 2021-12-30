from torch.utils import data
import pandas as pd
import numpy as np

class DataSetProcess(data.Dataset):
    '''
    Characterizers a dataset for pytorch
    '''
    def __init__(self, data, num_class, window, horizon):
        self.data = data
        self.num_class = num_class
        self.window = window
        self.horizon = horizon
        self.normalizeData()

    def normalizeData(self):
        '''
        normalize the dataset according the past 4800 ticks
        except columns

         [TradingDay, Time, Match, price]
        :return:
        '''
        self.data['price'] = (self.data['AskPrice1'] + self.data['BidPrice1']) / 2
        columns = self.data.columns.to_list()[3:]
        for column in columns:
            if column != 'price':
                self.data[column] = (self.data[column] - self.data[column].rolling(4800).mean()) / self.data[column].rolling(4800).std()
        self.data = self.data.dropna()

    def prepare_x(self):
        df = np.array(self.data.iloc[:, 3:-1])
        return df

    def get_label(self):
        label = self.data['price']




