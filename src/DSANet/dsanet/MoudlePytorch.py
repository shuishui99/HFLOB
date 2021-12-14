import torch
import torch.nn as nn
from test_tube import HyperOptArgumentParser

from DSANet.dsanet.Layers import EncoderLayer
import torch.nn.functional as F
import argparse
from DSANet.dataset import MTSFDataset


class singleGlobalSelfAttMoudle(nn.Module):
    def __init__(
            self,
            window, n_multiv, n_kernels, w_kernel,
            d_k, d_v, d_model, d_inner,
            n_layers, n_head, drop_prob=0.1):
        '''

        :param window(int):  输入窗口长度
        :param n_multiv(int): num of univariate time series（特征数）
        :param n_kernels(int): the num of channels
        :param w_kernels(int): 初始通道数，default=1
        :param d_k(int): d_model / n_head
        :param d_v(int): d_model / n_head
        :param d_model(int): 输出维度
        :param d_inner(int): the inner-layer dimension of Position-wise Feed-Forward Networks
        :param n_layers(int): num of layers in Encoder
        :param n_head(int): the num of Multi-head
        :param drop_prob(float): the probability of dropout
        '''

        super(singleGlobalSelfAttMoudle, self).__init__()
        self.window = window
        self.n_multiv = n_multiv
        self.n_kernels = n_kernels
        self.w_kernel = w_kernel
        self.drop_prob = drop_prob
        self.conv2 = nn.Conv2d(1, n_kernels, kernel_size=(window, w_kernel))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])
    def forward(self, x, return_attns=False):
        # [channel, window, n_multiv]
        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x2 = F.relu(self.conv2(x))
        x2 = nn.Dropout(p=self.drop_prob)(x2)
        x = torch.squeeze(x2, 2) #维度压缩，去除dim为1的
        print(x.shape)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)

        enc_slf_attn_list = []
        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,


class singleLocalSelfAttMoudle(nn.Module):
    def __init__(self,
                 window, local, n_multiv, n_kernels, w_kernel,
                 d_k, d_v, d_model, d_inner,
                 n_layers, n_head, drop_prob=0.1):
        '''

        :param local: the length of local kernel
        :param w_kernel: the num of channels

        '''
        super(singleLocalSelfAttMoudle, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(1, n_kernels, (local, w_kernel))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, n_multiv))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)
        ])

    def forward(self, x, return_attns=False):
        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x1 = F.relu(self.conv1(x))
        x1 = self.pooling1(x1)

        x1 = nn.Dropout(p=self.drop_prob)(x1)
        print("befor squeeze: ", x1.shape)
        x = torch.squeeze(x1, 2)
        x = torch.transpose(x, 1, 2)
        print("After transpose: ", x.shape)
        enc_output = self.in_linear(x)

        enc_slf_attn_list = []

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,



class AR(nn.Module):
    def __init__(self, window):
        '''

        :param window: the length of the input window size
        '''
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
        x = torch.squeeze(x, 1)
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x


class DSNetModel(nn.Module):
    def __init__(
            self, window, local, n_kernels, w_kernel, d_model, d_inner, n_layers,
            n_head, d_k, d_v, n_multive=1, batch_size=16, drop_prob=0.1):

        '''
        :param batch_size: default=16
        :param local:the length of local kernel
        :param drop_prob:
        :param window(int):  输入窗口长度
        :param n_multiv(int): num of univariate time series（特征数）
        :param n_kernels(int): the num of channels
        :param w_kernels(int): 初始通道数，default=1
        :param d_k(int): d_model / n_head
        :param d_v(int): d_model / n_head
        :param d_model(int): 输出维度
        :param d_inner(int): the inner-layer dimension of Position-wise Feed-Forward Networks
        :param n_layers(int): num of layers in Encoder
        :param n_head(int): the num of Multi-head
        :param drop_prob(float): the probability of dropout
        '''
        super(DSNetModel, self).__init__()
        self.window = window
        self.local = local
        self.n_kernels = n_kernels
        self.w_kernel = w_kernel
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.n_multive = n_multive
        self.batch_size = batch_size
        self.drop_prob = drop_prob
        self.sgsf = singleGlobalSelfAttMoudle(
            window=self.window, n_multive=self.n_multive, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head,
            drop_prob=self.drop_prob
        )

        self.slsf = singleLocalSelfAttMoudle(
            window=self.window, local=self.local, n_multiv=self.n_multive,n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model, d_inner=self.d_inner,
            n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob
        )

        self.ar = AR(window=self.window)

        self.W_output1 = nn.Linear(2*self.n_kernels, 1)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()


    def forward(self, x):
        '''

        :param x:
        :return:
        '''
        sgsg_output, *_ = self.sgsf(x)
        slsf_output, *_ = self.slsf(x)
        sf_output = torch.cat((sgsg_output, slsf_output), 2)
        sf_output = self.dropout(sf_output)
        sf_output = self.W_output1(sf_output)
        ar_output = self.ar(x)
        output = sf_output+ar_output
        return output



def main():
    '''
    :param batch_size: default=16
    :param local:the length of local kernel
    :param window(int):  输入窗口长度
    :param n_multiv(int): num of univariate time series（特征数）
    :param n_kernels(int): the num of channels
    :param w_kernels(int): 初始通道数，default=1
    :param d_k(int): d_model / n_head
    :param d_v(int): d_model / n_head
    :param d_model(int): 输出维度
    :param d_inner(int): the inner-layer dimension of Position-wise Feed-Forward Networks
    :param n_layers(int): num of layers in Encoder
    :param n_head(int): the num of Multi-head
    :param drop_prob(float): the probability of dropout
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-window', type=int, default=100)
    parser.add_argument('-local', type=int, default=3)
    parser.add_argument('-drop_prob', type=float, default=0.1)
    parser.add_argument('-n_multiv', type=int, default=8)
    parser.add_argument('-n_kernels', type=int, default=32)
    parser.add_argument('-horizon', type=int, default=3)
    parser.add_argument('-w_kernels', type=int, default=64)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner', type=int, default=2048)
    parser.add_argument('-n_head', type=int, default=8)








if __name__ == '__main__':
    globalModel = singleGlobalSelfAttMoudle(100, 8, 64, 1, d_k=64, d_v=64, d_inner=2048, d_model=512, n_layers=6,
                                            n_head=8)
    localModel = singleLocalSelfAttMoudle(100, 3, 8, 32, 1, 64, 64, 512, 2048, 6, 8)

    ARModel = AR(100)

    x = torch.randn([64, 1, 100, 8])
    localModel = localModel(x)
    ARModel = ARModel(x)

