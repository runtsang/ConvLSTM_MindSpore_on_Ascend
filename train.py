# -*- encoding: utf-8 -*-
'''
@File    :   main.py
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
from utils.earlystopping import EarlyStopping
from utils.utils import ssim_preprocess

import os
import json
import numpy as np
import moxing as mox
import argparse
import datetime

import mindspore
import mindspore.nn as nn
from mindspore.dataset import GeneratorDataset
import mindspore.ops as ops
from mindspore import context

# openi setting

### Defines whether the task is a training environment or a debugging environment ###
def WorkEnvironment(environment): 
    if environment == 'train':
        workroot = '/home/work/user-job-dir' 
    elif environment == 'debug':
        workroot = '/home/ma-user/work' 
    print('current work mode:' + environment + ', workroot:' + workroot)
    return workroot

### Copy multiple datasets from obs to training image ###
def MultiObsToEnv(multi_data_url, workroot):
    multi_data_json = json.loads(multi_data_url)  
    for i in range(len(multi_data_json)):
        path = workroot + "/" + multi_data_json[i]["dataset_name"]
        if not os.path.exists(path):
            os.makedirs(path)
        try:
            mox.file.copy_parallel(multi_data_json[i]["dataset_url"], path) 
            print("Successfully Download {} to {}".format(multi_data_json[i]["dataset_url"],
                                                        path))
        except Exception as e:
            print('moxing download {} to {} failed: '.format(
                multi_data_json[i]["dataset_url"], path) + str(e))

# parser setting
parser = argparse.ArgumentParser()
parser.add_argument('--data_url',
                    help='path to training/inference dataset folder',
                    default='./data')

parser.add_argument('--multi_data_url',
                    help='path to multi dataset',
                    default= WorkEnvironment('train'))

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
parser.add_argument('-epochs', default=80, type=int, help='sum of epochs')
args = parser.parse_args()

random_seed = 1989
np.random.seed(random_seed)
mindspore.set_seed(random_seed)

# openi setting

######################## 将数据集从obs拷贝到训练镜像中 （固定写法）########################   
# 在训练环境中定义data_url和train_url，并把数据从obs拷贝到相应的固定路径
obs_data_url = args.data_url
args.data_url = '/home/work/user-job-dir/'
obs_train_url = args.train_url
environment = 'train'
workroot = WorkEnvironment(environment)
args.train_url = workroot + '/model'   
if not os.path.exists(args.train_url):
    os.makedirs(args.train_url)
MultiObsToEnv(args.multi_data_url, workroot)
######################## 将数据集从obs拷贝到训练镜像中 ########################
    
#将dataset_path指向data_url，save_checkpoint_path指向train_url
dataset_path = args.data_url

save_dir = (workroot + "/checkpoint_66_0.000961/")

trainGenerator = MovingMNIST(is_train=True,
                          root=(workroot + "/train-images-idx3-ubyte/"),
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[3])
validGenerator = MovingMNIST(is_train=False,
                          root=(workroot + "/train-images-idx3-ubyte/"),
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[3])

trainDataset = GeneratorDataset(trainGenerator, column_names=['data', 'label'], shuffle=True).batch(batch_size=args.batch_size, drop_remainder=True)

validDataset = GeneratorDataset(validGenerator, column_names=['data', 'label']).batch(batch_size=args.batch_size)

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

def train():
    '''
    main function to run the training
    '''
    encoder = Encoder(encoder_params[0], encoder_params[1])
    decoder = Decoder(decoder_params[0], decoder_params[1])
    net = ED(encoder, decoder)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)
    context.set_context(device_target='Ascend', device_id=0)
    context.set_context(mode=context.PYNATIVE_MODE)
    
    load_checkpoint = os.path.join(save_dir, args.checkpoints)
    if os.path.exists(load_checkpoint):
        # load existing model
        print('==> loading existing model')
        model_info = mindspore.load_checkpoint(load_checkpoint)
        mindspore.load_param_into_net(net, model_info)
        cur_epoch = int(args.checkpoints[11:13])
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0

    lossfunction = nn.MSELoss()
    min_lr = 0.01
    max_lr = 0.1
    decay_steps = 4
    pla_lr_scheduler = nn.CosineDecayLR(min_lr, max_lr, decay_steps)
    group_params = init_group_params(net)
    optimizer = nn.Adam(group_params, learning_rate=pla_lr_scheduler)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf

    # caculate the MAE MSE and SSIM
    total_mse, total_mae,total_ssim = 0,0,0

    ###################
    # train the model #
    ###################
    net_with_loss = nn.WithLossCell(net, lossfunction)
    train_network = nn.TrainOneStepCell(net_with_loss, optimizer)
    train_network.set_train()
    data_iterator = trainDataset.create_tuple_iterator(num_epochs=args.epochs)
    
    for i in range(cur_epoch, args.epochs):
        total_mse, total_mae,total_ssim = 0,0,0
        epoch_len = len(str(args.epochs))
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print_msg = (f'[{i:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'TIME: {time} Start Trainning!' )
        print(print_msg)
        for data, label in data_iterator:
            loss = train_network(data, label)
            loss_aver = loss.asnumpy().item() / args.batch_size
            train_losses.append(loss_aver)

        train_loss = np.average(train_losses)
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print_msg = (f'# TIME: {time} Train End with loss: {train_loss:.6f}!' )
        print(print_msg)

        ######################
        # validate the model #
        ######################
        data_iterator = validDataset.create_tuple_iterator()
        for data, label in data_iterator:
            output = net(data)
            mse_batch = np.mean((output.asnumpy()-label.asnumpy())**2 , axis=(0,1,2)).sum()
            mae_batch = np.mean(np.abs(output.asnumpy()-label.asnumpy()) ,  axis=(0,1,2)).sum() 
            total_mse += mse_batch
            total_mae += mae_batch
            loss = lossfunction(output, label)
            loss_aver = loss.asnumpy().item() / args.batch_size
            valid_losses.append(loss_aver)
            ssim_cal = nn.SSIM()
            output, label=ssim_preprocess(output, label)
            mean = mindspore.ops.ReduceMean()
            ssim_val = ssim_cal(output, label)
            total_ssim += mean(ssim_val)

        # print training/validation statistics
        # calculate average loss over an epoch
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print_msg = (f'# TIME: {time} Validate End with loss: {valid_loss:.6f}!' )
        print(print_msg)

        print_msg = (f'[{i:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f} |   '+
                     f'ssim: {(total_ssim.asnumpy().item() / validDataset.get_dataset_size()):.6f} '+
                     f'mae: {(total_mae / validDataset.get_dataset_size()):.6f} '+
                     f'mse: {(total_mse / validDataset.get_dataset_size()):.6f} ')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        early_stopping(valid_loss.item(), net, i, args.train_url)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)

    ######################
    # evaluate the model #
    ######################
    print('\nEvaluating the model based on Moving MNIST!')
    testGenerator = MovingMNIST(is_train=False,
                          root=(workroot + "/mnist_test_seq/"),
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[2])

    testDataset = GeneratorDataset(testGenerator, column_names=['data', 'label']).batch(batch_size=args.batch_size)
    
    load_checkpoint = os.path.join(save_dir, args.checkpoints)
    if os.path.exists(load_checkpoint):
        # load existing model
        model_info = mindspore.load_checkpoint(load_checkpoint)
        mindspore.load_param_into_net(net, model_info)
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    # to track the test loss as the model trains
    test_losses = []
    # to track the average test loss per epoch as the model trains
    avg_test_losses = []

    lossfunction = nn.MSELoss()

    # caculate the MAE MSE and SSIM
    total_mse, total_mae, total_ssim = 0,0,0

    ######################
    # test the model #
    ######################
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
        loss_aver = loss.asnumpy().item() / args.batch_size
        test_losses.append(loss_aver)
        ssim_cal = nn.SSIM()
        output, label=ssim_preprocess(output, label)
        mean = mindspore.ops.ReduceMean()
        ssim_val = ssim_cal(output, label)
        total_ssim += mean(ssim_val)

    # print test statistics
    # calculate test loss over an epoch
    valid_loss = np.average(test_losses)
    avg_test_losses.append(valid_loss)

    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print_msg = (f'# TIME: {time} ' +
                f'test_loss: {valid_loss:.6f} |   '+
                f'ssim: {(total_ssim.asnumpy().item() / testDataset.get_dataset_size()):.6f} '+
                f'mae: {(total_mae / testDataset.get_dataset_size()):.6f} '+
                f'mse: {(total_mse / testDataset.get_dataset_size()):.6f} ')

    print(print_msg)

    # openi setting
    try:
        mox.file.copy_parallel(args.train_url, obs_train_url)
        print("Successfully Upload {} to {}".format(args.train_url,
                                                    obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(args.train_url,
                                                        obs_train_url) + str(e))

    return None

 

if __name__ == "__main__":
    train()
