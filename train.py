'''
Note:
I've made a babynet for test, including a conv and a relu
But I have to make some changes to let it run
please check them!
I don't know if they are caused by babynet or other error!

1. cross entropy not working!
ValueError: Expected target size (5, 2048), got torch.Size([5, 1024, 2048])
I've checked the shape of pred and batch_y
they are [5, 1024, 2048]
Then I try MSEloss
it works!

2. loss.avg can't be understand
I delete the avg and it works
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import numpy as np
from cityscape_dataset.dataset import CityscapeDataset
from config import DefaultConfig

from SETR import Net, BabyNet

import time

import argparse

def train(epoch_num, batch_size, learning_rate, device, output_path, pretrained_model):
    config = DefaultConfig()
    torch.backends.cudnn.enabled = False

    SETRNet = BabyNet()
    # SETRNet = Net(3).to(device)
    
    train_data_set = CityscapeDataset(config.train_img_root, config.train_target_root, train = True, test = False)
    train_data_loader = Data.DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True)
    val_data_set = CityscapeDataset(config.val_img_root, config.val_target_root, train = False, test = False)
    val_data_loader = Data.DataLoader(dataset=val_data_set, batch_size=batch_size, shuffle=True)
    
    best_epoch = 0
    best_mIoU = 0    
    
    CrossEntropyLoss = nn.CrossEntropyLoss()
    MSEloss = nn.MSELoss()
    opt = optim.Adam(SETRNet.parameters(), lr=learning_rate)
    loss_record = []
    
    tick = time.time()
    for epoch in range(epoch_num):
        tick_e = time.time()
        SETRNet.train()
        train_loss = 0
        for step, (batch_x, batch_y) in enumerate(train_data_loader):
            opt.zero_grad()
            pred = SETRNet(batch_x.to(device))
            pred = pred.squeeze()
            # print(pred.shape)
            # print(batch_y.shape)
            loss = CrossEntropyLoss(batch_y.to(device).float(), pred)
            # loss = MSEloss(batch_y.float(), pred)
        
        '''
        cross entropy not working
        MSEloss works
        '''
            
        
        loss.backward()
        opt.step()
        train_loss += loss.item()
        loss_record.append(loss.item())
        # log = 'Epoch:{0}\tLoss: {loss.avg:.4f}\t'.format(epoch, loss=train_loss)
        '''
        the avg seems not working
        '''
        log = 'Epoch:{0}\tLoss: {loss:.4f}\t'.format(epoch, loss=train_loss)
        print(log + '\n')
    # Net.eval()
    # with torch.no_grad():
    #     ## TO BE DONE: Evaluate the model
    #     mIoU = 0
    #     train_loss /= step + 1
    #
    #
    # print('----------------------------')
    # print("Epoch={:3d}\tTrain_loss={:6.4f}".format(epoch, train_loss))
    # if mIoU > best_mIoU:
    #     best_OA = eval_arr[-1]
    #     best_epoch = epoch
    #     torch.save(Net.state_dict(), output_path, _use_new_zipfile_serialization=False)
    # print('Best Epoch: ', best_epoch, ' Best mIoU: ', best_mIoU)
    # print("Time: {:5.2f}(Total: {:5.2f})".format(time.time() - tick_e, time.time() - tick))

'''
check the change in config.py
instead of enter hyper parameters every time,
construct a config class there

def parser():
    
    p = argparse.ArgumentParser()
    
    p.add_argument('-ep', '--epoch_num', 
                    help='the number of epoch (default: %(default)s)',
                    type=int, default=100)
    p.add_argument('-bs', '--batch_size', 
                    help='The number of batch size (default: %(default)s)',
                    type=int, default=50)
    p.add_argument('-lr', '--learning_rate', 
                    help='the number of learn rate (default: %(default)s)',
                    type=float, default=0.0001)    
    p.add_argument('-o', '--output_path',
                    help='Path to output folder (default: %(default)s)',
                    type=str, default='./pretrained_model/model')
    p.add_argument('-pm', '--pretrained_model',
                    help='the path of pretrained model (Transformer or Streamline) (default: %(default)s)',
                    type=str, default=None)
    
    return p.parse_args()
'''

if __name__ == '__main__':
    
    args = DefaultConfig()
    if args.gpu_index is not None:
        with torch.cuda.device(args.gpu_index):
            train(args.epoch_num, args.batch_size, args.learning_rate, args.gpu_index, args.output_path, args.pretrained_model)
    else: 
        train(args.epoch_num, args.batch_size, args.learning_rate, args.gpu_index, args.output_path, args.pretrained_model)
        
    