# -*- coding:utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class StatisticsPooling(torch.nn.Module):
    """ An usual mean [+ stddev] poolling layer"""

    def __init__(self, stddev=True):
        super(StatisticsPooling, self).__init__()

        self.stddev = stddev

    def forward(self, x, x_lens):
        """
        x: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        mean = torch.zeros(x.shape[:1])
        # mean = torch.mean(x, dim=-1)
        if self.stddev:
            std = torch.zeros_like(mean)

        for i in range(x.shape[0]):
            mean[i] = x[i][:,:x_lens[i]].mean(dim=-1)
            if self.stddev:
                std[i] = torch.sqrt(torch.var(x[i][:,:x_lens[i]], dim=-1) + 1e-10)
        if self.stddev:
            return torch.cat((mean, std), dim=1)
        else
            return mean
            # std = torch.sqrt(torch.var(x, dim=-1) + 1e-10)

class AttCalcualte(torch.nn.Module):
    def __init__(self, input_dim, affine_layers=2, norm_way='softmax'):
        '''
        input_dim: the second dimension of a three-dimensional input
        norm_way: choose from softmax or sigmoid
        '''
        super(AttCalcualte, self).__init__()

        assert affine_layers >= 2

        self.norm_way = norm_way

        output_dim = 1
        channel_dims = np.linspace(input_dim, output_dim, affine_layers + 1).astype(int)

        self.att_trans = nn.Sequential()
        for i in range(affine_layers-1):
            self.att_trans.add_module('att_' + str(i), nn.Conv1d(channel_dims[i], channel_dims[i+1], 1, 1))
            self.att_trans.add_module('relu_' + str(i), nn.ReLU())
        self.att_trans.add_module('att_' + str(affine_layers-1), nn.Conv1d(channel_dims[affine_layers-1], channel_dims[affine_layers], 1, 1))

    def forward(self, input):

        att_score = self.att_trans(input)
        if self.norm_way == 'softmax':
            att = F.softmax(att_score, dim=-1)
        else:
            att = torch.sigmoid(att_score)
            weight = torch.sum(att, dim=-1, keepdim=True)
            att = att / weight

        return att


class AttentiveStatisticsPooling(torch.nn.Module):
    """ An attentive statistics pooling.
    Reference: Okabe, Koji, Takafumi Koshinaka, and Koichi Shinoda. 2018. "Attentive Statistics Pooling
               for Deep Speaker Embedding." ArXiv Preprint ArXiv:1803.10963.
    """
    def __init__(self, input_dim, affine_layers=2,  stddev=True, norm_way='softmax'):
        super(AttentiveStatisticsPooling, self).__init__()

        self.stddev = stddev
        self.attention = AttCalcualte(input_dim=input_dim, affine_layers=affine_layers, norm_way=norm_way)

    def forward(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:
            input = input.reshape(input.shape[0], input.shape[1]*input.shape[2], input.shape[3])
        assert len(input.shape) == 3

        alpha = self.attention(input)

        mean = torch.sum(alpha * input, dim=2)

        if self.stddev:
            var = torch.sum(alpha * input ** 2, dim=2) - mean ** 2
            std = torch.sqrt(var.clamp(min=1e-10))

            return torch.cat((mean, std), dim=1)
        else:
            return mean



if __name__ == '__main__':

    data = torch.randn(16, 512, 5)
    # model = AttentiveStatisticsPooling(input_dim=512, affine_layers=3, norm_way='softmax',stddev=True)
    model = StatisticsPooling()
    print(model)

    out = model(data)
    print(out.shape)



