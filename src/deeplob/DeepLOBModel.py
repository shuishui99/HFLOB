import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def prepare_x(data):
    df1 = data[:40, :].T
    return np.array(df1)


def get_label(data):
    lob = data[-5:, :].T
    return lob


def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX, dataY


def torch_data(x, y):
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 1)
    y = torch.from_numpy(y)
    y = F.one_hot(y, num_classes=3)
    return x, y


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data, k, num_classes, T):
        """Initialization"""
        self.k = k
        self.num_classes = num_classes
        self.T = T

        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
        y = y[:, self.k] - 1
        self.length = len(x)

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]

dec_data = np.loadtxt('/home/yuan/jam/data/FI-2010/Train_Dst_NoAuction_DecPre_CF_7.txt')
dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]

batch_size = 64

dataset_train = Dataset(data=dec_train, k=4, num_classes=3, T=100)
train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
tmp_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=1, shuffle=True)



class deepLob(nn.Module):
    def __init__(self, yLen):
        super(deepLob, self).__init__()
        self.yLen = yLen

        #convolution layer
        '''
        为什么conv1用LeakyReLU，而conv2用Tanh？
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.BatchNorm2d(32),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.BatchNorm2d(32),
        )


        # Inception Module

        self.inception1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.BatchNorm2d(64),
        )

        self.inception2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.BatchNorm2d(64),
        )

        self.inception3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.BatchNorm2d(64),
        )

        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.yLen)


    def forward(self, x):

        h0 = torch.zeros(1, x.size(0), 64).to(device)
        c0 = torch.zeros(1, x.size(0), 64).to(device)


        # x.shape = [batch, channel, h, w] = [64, 1, 100, 40]
        # [64, 1, 100,40] -> [64, 32, 94, 20]
        x = self.conv1(x)
        # print("con1", x.shape)
        # [64, 32, 94, 20] -> [64, 32, 88, 10]
        x = self.conv2(x)
        # print("con2", x.shape)
        # [64, 32, 88, 10] -> [64, 32, 82, 1]
        x = self.conv3(x)
        # print("con3", x.shape)

        # [64, 32, 82, 1] -> [64, 64, 82, 1]
        x_incp1 = self.inception1(x)
        x_incp2 = self.inception2(x)
        x_incp3 = self.inception3(x)

        # [batch, channel, h, w] -> [64, 192, 82, 1]
        x = torch.cat((x_incp1, x_incp2, x_incp3), dim=1)

        x = x.permute(0, 2, 1, 3)
        # [64, 82, 192]
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        forecast_y = torch.softmax(x, dim=1)

        print(forecast_y.shape)

        return forecast_y



dec_data = np.loadtxt('/home/yuan/jam/data/FI-2010/Train_Dst_NoAuction_DecPre_CF_7.txt')
dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]

batch_size = 64

dataset_train = Dataset(data=dec_train, k=4, num_classes=3, T=100)
train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
tmp_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=1, shuffle=True)



model = deepLob(dataset_train.num_classes)
model.to(device)

for inputs, targets in train_loader:
    # move data to GPU
    inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
    print("inputs.shape:", inputs.shape)
    print("target.shape:", targets.shape)
    outputs = model(inputs)
    print("--------done!-----------")
    # zero the parameter gradients
    break
