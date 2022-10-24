# -*- encoding: utf-8 -*-
'''
@File    :   ConvRNN.py
@Time    :   2022/07/26
@Author  :   zrainj
@Mail    :   rain1709@foxmail.com
@Description:   Based on MindSpore
'''

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.ops as ops
import mindspore.numpy as np

class CLSTM_cell(nn.Cell):
    """
    ConvLSTMCell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.SequentialCell([
            nn.Conv2d(in_channels=self.input_channels + self.num_features, out_channels=4 * self.num_features, kernel_size=self.filter_size, stride=1, pad_mode='pad', padding=self.padding, has_bias=True),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features)])

    def construct(self, inputs=None, hidden_state=None, seq_len=10):
        #  seq_len=10 for moving_mnist
        if hidden_state is None:
            hx = np.zeros((P.Shape()(inputs)[1], self.num_features, self.shape[0],
                             self.shape[1]))
            cx = np.zeros((P.Shape()(inputs)[1], self.num_features, self.shape[0],
                             self.shape[1]))
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = np.zeros((P.Shape()(hx)[0], self.input_channels, self.shape[0],
                                self.shape[1]))
            else:
                x = inputs[index, ...]

            combined = P.Concat(1)((x, hx))
            gates = self.conv(combined)  # gates: S, num_features*4, H, W

            split = ops.Split(1, 4)
            gates = split(gates)
            ingate, forgetgate, cellgate, outgate = gates[0], gates[1], gates[2], gates[3]
            ingate = mindspore.ops.Sigmoid()(ingate)
            forgetgate = mindspore.ops.Sigmoid()(forgetgate)
            cellgate = mindspore.ops.Tanh()(cellgate)
            outgate = mindspore.ops.Sigmoid()(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * mindspore.ops.Tanh()(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return mindspore.ops.Stack()(output_inner), (hy, cy)
