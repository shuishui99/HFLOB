import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):

    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        '''
        torch.bmm:两个矩阵点乘，两个矩阵维度必须为三维，batch1(bxnxm) batch2(bxmxp) -> (bxnxp)
        :param q:
        :param k:
        :param v:
        :return:
        '''

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
