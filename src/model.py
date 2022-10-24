# -*- encoding: utf-8 -*-
'''
@File    :   encoder.py
@Time    :   2022/07/26
@Author  :   zrainj
@Mail    :   rain1709@foxmail.com
@Description:   Based on MindSpore
'''

import mindspore.nn as nn

class ED(nn.Cell):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, input):
        state = self.encoder(input)
        output = self.decoder(state)
        return output
