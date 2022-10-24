# -*- encoding: utf-8 -*-
'''
@File    :   encoder.py
@Time    :   2022/07/26
@Author  :   zrainj
@Mail    :   rain1709@foxmail.com
@Description:   Based on MindSpore
'''

import mindspore
import mindspore.nn as nn
import mindspore.ops as P
from utils.utils import make_layers

class Encoder(nn.Cell):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)
        
        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = P.Shape()(inputs)
        inputs = mindspore.ops.Reshape()(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = mindspore.ops.Reshape()(inputs, (seq_number, batch_size, P.Shape()(inputs)[1],
                                        P.Shape()(inputs)[2], P.Shape()(inputs)[3]))
        outputs_stage, state_stage = rnn(inputs, None)
        return outputs_stage, state_stage

    def construct(self, inputs):
        inputs = inputs.transpose(1, 0, 2, 3, 4)  # to S,B,1,64,64
        hidden_states = []
        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage(
                inputs, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)
    