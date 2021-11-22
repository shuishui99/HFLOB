import os
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
from sklearn import preprocessing
import torch.utils.data as Data
from time import time
def dataProcess(path, code, columns):
    files = os.listdir(path)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    data_pck = pd.DataFrame(data=None)
    for file in tqdm(files):
        data = pd.read_csv(path+file, usecols=columns)
        data = data[data['WindCode']==code]
        data_pck = pd.concat([data, data_pck], ignore_index=True, axis=0)
    data_pck = data_pck[data_pck['Time'] > 93000000]
    data_pck = data_pck[data_pck['Time'] < 145700000]
    data_pck['Volume'] = data_pck.groupby('TradingDay')['Volume'].apply(lambda x: x - x.shift(1)) #Volume
    data_pck['label'] = data_pck.groupby('TradingDay')['Match'].apply(lambda x: x.shift(-1) - x).dropna() #label
    data_pck = data_pck.dropna()
    data_pck['label'] = np.where(data_pck['label'] >= 0, 1, 0)
    data_pck = data_pck.sort_values(by=['TradingDay', 'Time']).drop(columns=['WindCode', 'HighLimited', 'LowLimited'])
    #暂不归一化
    # columns = data_pck.columns
    # maxmin = preprocessing.MinMaxScaler()
    # data_pck = maxmin.fit_transform(data_pck)
    # data_pck = pd.DataFrame(data_pck, columns=columns)
    data_pck.to_pickle('../data/'+code+'.pkl')

def prepareData(path, length):
    """
    将数据转化为矩阵
    :param path: 文件存储的路径
    :param length: 每次回看的长度
    :return: x: tensor, y:tensor
    """
    data = pd.read_pickle(path)
    x_data = []
    y_data = []
    tradeDays = list(set(data['TradingDay'].values))
    tradeDays.sort()
    for tradeDay in tradeDays:
        tmp = data[data['TradingDay']==tradeDay].sort_values(by=['Time']).drop(columns=['TradingDay', 'Time'])
        for i in range(length, tmp.shape[0]):
            x_data.append(tmp.iloc[i-length:i, :-1].values)
            y_data.append(int(tmp.iloc[i-1, -1]))
    x_data = torch.from_numpy(np.array(x_data)).to(torch.float32)
    y_data = torch.from_numpy(np.array(y_data)).to(torch.float32)
    return x_data, y_data




class Model(torch.nn.Module):
    def __init__(self, input_size, hidden, n_layer, n_class):
        super(Model, self).__init__()
        self.hidden = hidden
        self.n_layer = n_layer
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden, num_layers=1, batch_first=True) #input_size, hidden, laywer
        self.classfier = torch.nn.Linear(hidden, n_class)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        x = h_n[-1, :, :]
        x = torch.sigmoid(self.classfier(x))
        return x


# class DemoDatasetLSTM(Data.Dataset):
#     """
#         Support class for the loading and batching of sequences of samples
#         Args:
#             dataset (Tensor): Tensor containing all the samples
#             sequence_length (int): length of the analyzed sequence by the LSTM
#             transforms (object torchvision.transform): Pytorch's transforms used to process the data
#     """
#
#     ##  Constructor
#     def __init__(self, dataset, sequence_length=1, transforms=None):
#         self.dataset = dataset
#         self.seq_len = sequence_length
#         self.transforms = transforms
#
#     ##  Override total dataset's length getter
#     def __len__(self):
#         return self.dataset.__len__()
#
#     ##  Override single items' getter
#     def __getitem__(self, idx):
#         if idx + self.seq_len > self.__len__():
#             if self.transforms is not None:
#                 item = torch.zeros(self.seq_len, self.dataset[0].__len__())
#                 item[:self.__len__() - idx] = self.transforms(self.dataset[idx:])
#                 return item, item
#             else:
#                 item = []
#                 item[:self.__len__() - idx] = self.dataset[idx:]
#                 return item, item
#         else:
#             if self.transforms is not None:
#                 return self.transforms(self.dataset[idx:idx + self.seq_len]), self.transforms(
#                     self.dataset[idx:idx + self.seq_len])
#             else:
#                 return self.dataset[idx:idx + self.seq_len], self.dataset[idx:idx + self.seq_len]
#


#
# x, y = prepareData('../data/000001.SZ.pkl', 20)
# torch_data = Data.TensorDataset(x, y)
# trainloader = Data.DataLoader(dataset=torch_data, batch_size=6, shuffle=False, num_workers=2, drop_last=True)
# net = Model(22, 128, 1, 1)
# criterion = torch.nn.functional(reduction='mean')
# optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


def train(epoch):
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to('cpu'), targets.to('cpu')
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs.squeeze(-1), targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 471 == 0:
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))



# 相对价差
def get_relative_spread(df, window):
    df['relative_spread'] = (df['a1_p'] - df['b1_p']) / ((df['a1_p'] + df['b1_p']) / 2)
    df['relative_spread_sum'] = df['relative_spread'].rolling(window).sum()
    df = df.dropna()
    return df


# 深度不平衡
def get_Length_imb(df, window):
    df['Length_imb_1_sum'] = ((df['b1_v'] - df['a1_v']) / (df['a1_v'] + df['b1_v'])).rolling(window).sum()
    df['Length_imb_2_sum'] = ((df['b2_v'] - df['a2_v']) / (df['a2_v'] + df['b2_v'])).rolling(window).sum()
    df['Length_imb_3_sum'] = ((df['b3_v'] - df['a3_v']) / (df['a3_v'] + df['b3_v'])).rolling(window).sum()
    df['Length_imb_4_sum'] = ((df['b4_v'] - df['a4_v']) / (df['a4_v'] + df['b4_v'])).rolling(window).sum()
    df['Length_imb_5_sum'] = ((df['b5_v'] - df['a5_v']) / (df['a5_v'] + df['b5_v'])).rolling(window).sum()
    df = df.dropna()
    return df

def get_press(df, window):
    df['mid'] = (df['BidPrice1']+df['AskPrice1']) / 2
    df['BidSigma'] = 0
    df['AskSigma'] = 0
    for i in range(1, 6, 1):
        df['BidSigma'] += df['mid']/(df['mid']-df['BidPrice%d' % i])
        df['AskSigma'] += df['mid'] / (df['mid'] - df['AskPrice%d' % i])
    wb1 = (df['mid'] / (df['mid'] - df['BidPrice1'])) / df['BidSigma']
    wb2 = (df['mid'] / (df['mid'] - df['BidPrice2'])) / df['BidSigma']
    wb3 = (df['mid'] / (df['mid'] - df['BidPrice3'])) / df['BidSigma']
    wb4 = (df['mid'] / (df['mid'] - df['BidPrice4'])) / df['BidSigma']
    wb5 = (df['mid'] / (df['mid'] - df['BidPrice5'])) / df['BidSigma']

    wa1 = (df['mid'] / (df['mid'] - df['AskPrice1'])) / df['AskSigma']
    wa2 = (df['mid'] / (df['mid'] - df['AskPrice2'])) / df['AskSigma']
    wa3 = (df['mid'] / (df['mid'] - df['AskPrice3'])) / df['AskSigma']
    wa4 = (df['mid'] / (df['mid'] - df['AskPrice4'])) / df['AskSigma']
    wa5 = (df['mid'] / (df['mid'] - df['AskPrice5'])) / df['AskSigma']

    df['press'] = np.log(wb1*df['BidVol1'] + wb2*df['BidVol2'] + wb3*df['BidVol3'] + wb4*df['BidVol4'] + wb5*df['BidVol5']) \
                - np.log(wa1*df['AskVol1'] + wa2*df['AskVol2'] + wa3*df['AskVol3'] + wa4*df['AskVol4'] + wa5*df['AskVol5'])
    df = df.drop(columns=['mid','AskSigma','BidSigma'])
    if window < 1:
        raise ("the window wrong!!")
    else:
        df['press'] = df['press'].rolling(window).sum()
        df = df.dropna()
    return df

def sigmoid(x):
    # TODO: Implement sigmoid function
    return 1/(1 + np.exp(-x))

def get_height_imb(df, window):
    for i in range(2, 6, 1):
        df['height_imb_%d'%i] = (sigmoid((df['BidPrice%d'%i] - df['BidPrice%d'% (i-1)]) - (df['AskPrice%d'%i] - df['AskPrice%d'%(i-1)])) /
                                 sigmoid((df['BidPrice%d'%i] - df['BidPrice%d'% (i-1)]) + (df['AskPrice%d'%i] - df['AskPrice%d'%(i-1)]))).rolling(window).sum()
    df = df.dropna()
    return df

def get_Length_imb(df,window):
    for i in range(1, 6, 1):
        df['Length_imb_%d_sum'%i] = ((df['BidVol%d'%i] - df['AskVol%d'%i]) / (df['AskVol%d'%i] + df['BidVol%d'%i])).rolling(window).sum()
    df = df.dropna()
    return df




if __name__ == "__main__":
    # path = '/Users/jam/PycharmProjects/OderFlow/tickData/'
    # columns = ['WindCode', 'TradingDay', 'Time', 'Match', 'AskPrice1', 'AskPrice2',
    #            'AskPrice3', 'AskPrice4', 'AskPrice5', 'AskVol1', 'AskVol2', 'AskVol3',
    #            'AskVol4', 'AskVol5', 'AskVol1', 'AskVol2', 'AskVol3', 'AskVol4',
    #            'BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5',
    #            'AskVol5', 'BidVol1', 'BidVol2', 'BidVol3', 'BidVol4', 'BidVol5',
    #            'Volume', 'HighLimited', 'LowLimited']
    # dataProcess(path, '000001.SZ', columns)
    # start = time()
    # for epoch in tqdm(range(20)):
    #     train(epoch)
    # end = time()
    # print((end - start)/60)
    data = pd.read_pickle('../data/000001.SZ.pkl')
    get_Length_imb(data, 1)



