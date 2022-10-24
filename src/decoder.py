# -*- encoding: utf-8 -*-
'''
@File    :   decoder.py
@Time    :   2022/07/26
@Author  :   zrainj
@Mail    :   rain1709@foxmail.com
@Description:   Based on MindSpore
'''

import mindspore
import mindspore.nn as nn
import mindspore.ops as P
from utils.utils import make_layers

class Decoder(nn.Cell):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))

    def forward_by_stage(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, state, seq_len=10)
        seq_number, batch_size, input_channel, height, width = P.Shape()(inputs)
        inputs = mindspore.ops.Reshape()(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = mindspore.ops.Reshape()(inputs, (seq_number, batch_size, P.Shape()(inputs)[1],
                                        P.Shape()(inputs)[2], P.Shape()(inputs)[3]))
        return inputs

        # input: 5D S*B*C*H*W

    def construct(self, hidden_states):
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       getattr(self, 'stage3'),
                                       getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))
        inputs = inputs.transpose(1, 0, 2, 3, 4)  # to B,S,1,64,64
        return inputs
