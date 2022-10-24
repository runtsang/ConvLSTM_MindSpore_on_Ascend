# -*- encoding: utf-8 -*-
'''
@File    :   encoder.py
@Time    :   2022/07/26
@Author  :   zrainj
@Mail    :   rain1709@foxmail.com
@Description:   Based on MindSpore
'''

from src.encoder import Encoder
from src.decoder import Decoder
from src.model import ED
from src.net_params import convlstm_encoder_params, convlstm_decoder_params
from src.MMDatasets import MovingMNIST
from utils.utils import ssim_preprocess

import os
import numpy as np
import moxing as mox
import argparse
import datetime

import mindspore
import mindspore.nn as nn
from mindspore.dataset import GeneratorDataset
from mindspore import context

parser = argparse.ArgumentParser()
parser.add_argument('--data_url',
                    help='path to training/inference dataset folder',
                    default='./data')

parser.add_argument('--train_url',
                    help='model folder to save/load',
                    default='./model')

parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=['Ascend'],
    help='device where the code will be implemented (default: Ascend)')

parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=24,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=10,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=10,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-checkpoints',
                    default='checkpoint_66_0.000961.ckpt',
                    type=str,
                    help='use which checkpoints')
parser.add_argument('-epochs', default=500, type=int, help='sum of epochs')
args = parser.parse_args()

random_seed = 1989
np.random.seed(random_seed)
mindspore.set_seed(random_seed)

######################## 将数据集从obs拷贝到训练镜像中 （固定写法）########################   
# 在训练环境中定义data_url和train_url，并把数据从obs拷贝到相应的固定路径
obs_data_url = args.data_url
args.data_url = '/home/work/user-job-dir/inputs/data/'
obs_train_url = args.train_url
args.train_url = '/home/work/user-job-dir/outputs/model/'
try:
    mox.file.copy_parallel(obs_data_url, args.data_url)
    print("Successfully Download {} to {}".format(obs_data_url,
                                                    args.data_url))
except Exception as e:
    print('moxing download {} to {} failed: '.format(
        obs_data_url, args.data_url) + str(e))
######################## 将数据集从obs拷贝到训练镜像中 ########################
    
#将dataset_path指向data_url，save_checkpoint_path指向train_url
dataset_path = args.data_url

save_dir = './save_model/'

testGenerator = MovingMNIST(is_train=False,
                          root=dataset_path,
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[2])

testDataset = GeneratorDataset(testGenerator, column_names=['data', 'label']).batch(batch_size=args.batch_size)

encoder_params = convlstm_encoder_params
decoder_params = convlstm_decoder_params

def init_group_params(net):
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': 0.0},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    return group_params

def test():
    encoder = Encoder(encoder_params[0], encoder_params[1])
    decoder = Decoder(decoder_params[0], decoder_params[1])
    net = ED(encoder, decoder)

    context.set_context(device_target='Ascend', evice_id=0)
    context.set_context(mode=context.PYNATIVE_MODE)
    
    load_checkpoint = os.path.join(save_dir, args.checkpoints)
    if os.path.exists(load_checkpoint):
        # load existing model
        print('==> loading existing model')
        model_info = mindspore.load_checkpoint(load_checkpoint)
        mindspore.load_param_into_net(net, model_info)
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    lossfunction = nn.MSELoss()

    # caculate the MAE MSE and SSIM
    total_mse, total_mae,total_ssim = 0,0,0

    # test the model #
    data_iterator = testDataset.create_tuple_iterator()
    for data, label in data_iterator:
        data = mindspore.Tensor(data, dtype=mindspore.float32)
        label = mindspore.Tensor(label, dtype=mindspore.float32)
        output = net(data)

        mse_batch = np.mean((output.asnumpy()-label.asnumpy())**2 , axis=(0,1,2)).sum()
        mae_batch = np.mean(np.abs(output.asnumpy()-label.asnumpy()) ,  axis=(0,1,2)).sum() 
        total_mse += mse_batch
        total_mae += mae_batch
        loss = lossfunction(output, label)
        ssim_cal = nn.SSIM()
        output, label=ssim_preprocess(output, label)
        mean = mindspore.ops.ReduceMean()
        ssim_val = ssim_cal(output, label)
        total_ssim += mean(ssim_val)

    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print_msg = (f'# TIME: {time}' +
                f'ssim: {(total_ssim.asnumpy().item() / testDataset.get_dataset_size()):.6f} '+
                f'mae: {(total_mae / testDataset.get_dataset_size()):.6f} '+
                f'mse: {(total_mse / testDataset.get_dataset_size()):.6f} ')

    print(print_msg)
    return None

if __name__ == "__main__":
    test()