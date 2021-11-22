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
    y_data = torch.from_numpy(np.array(y_data)).to(torch.long)
    return x_data, y_data




class Model(torch.nn.Module):
    def __init__(self, input_size, hidden, n_layer, n_class):
        super(Model, self).__init__()
        self.hidden = hidden
        self.n_layer = n_layer
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden, num_layers=2, batch_first=True) #input_size, hidden, laywer
        self.classfier = torch.nn.Linear(hidden, n_class)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        x = h_n[-1, :, :]
        x = self.classfier(x)
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



x, y = prepareData('../data/000001.SZ.pkl', 20)
torch_data = Data.TensorDataset(x, y)
trainloader = Data.DataLoader(dataset=torch_data, batch_size=6, shuffle=False, num_workers=2, drop_last=True)
net = Model(22, 64, 2, 2)
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


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

if __name__ == "__main__":
    # path = '/Users/jam/PycharmProjects/OderFlow/tickData/'
    # columns = ['WindCode', 'TradingDay', 'Time', 'Match', 'AskPrice1', 'AskPrice2',
    #            'AskPrice3', 'AskPrice4', 'AskPrice5', 'AskVol1', 'AskVol2', 'AskVol3',
    #            'AskVol4', 'AskVol5', 'AskVol1', 'AskVol2', 'AskVol3', 'AskVol4',
    #            'BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5',
    #            'AskVol5', 'BidVol1', 'BidVol2', 'BidVol3', 'BidVol4', 'BidVol5',
    #            'Volume', 'HighLimited', 'LowLimited']
    # dataProcess(path, '000001.SZ', columns)
    start = time()
    for epoch in tqdm(range(20)):
        train(epoch)
    end = time()
    print((end - start)/60)



