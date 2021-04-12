import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import numpy as np

from utils import *
from SETR.model import Segmentor

import time

import argparse


class Dataset(Data.Dataset):
    # TO BE MODIFIED !!!
    def __init__(self, data_tensor, target_tensor):

        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

def train(batch_size, learning_rate, epoch_num, output_path, pretrained_model):
    
    torch.backends.cudnn.enabled = False
    
    device = 'cuda' if torch.cuda.is_available else 'cpu'    
    Net = Segmentor().to(device)
    
    train_data_set = Dataset()
    train_data_loader = Data.DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True)
    val_data_set = Dataset()
    val_data_loader = Data.DataLoader(dataset=val_data_set, batch_size=batch_size, shuffle=True)
    
    best_epoch = 0
    best_mIoU = 0    
    
    CrossEntropyLoss = nn.CrossEntropyLoss()
    opt = optim.Adam(Net.parameters(), lr=learning_rate)
    loss_record = []
    
    tick = time.time()
    for epoch in range(epoch_num):
        tick_e = time.time()
        Net.train()
        train_loss = 0
        for step, (batch_x, batch_y) in enumerate(train_data_loader):
            opt.zero_grad()
            pred = Net(batch_x.to(device))
            loss = CrossEntropyLoss(batch_y.to(device))
        
        loss.backward()
        opt.step()
        train_loss += loss.item()
        loss_record.append(loss.item())
        
    Net.eval()
    with torch.no_grad():
        ## TO BE DONE: Evaluate the model
        mIoU = 0
        train_loss /= step + 1

    
    print('----------------------------')
    print("Epoch={:3d}\tTrain_loss={:6.4f}".format(epoch, train_loss))
    if mIoU > best_mIoU:
        best_OA = eval_arr[-1]
        best_epoch = epoch
        torch.save(Net.state_dict(), output_path, _use_new_zipfile_serialization=False)
    print('Best Epoch: ', best_epoch, ' Best mIoU: ', best_mIoU)
    print("Time: {:5.2f}(Total: {:5.2f})".format(time.time() - tick_e, time.time() - tick))

    
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

if __name__ == '__main__':

    args = parser()
    if args.gpu_index is not None:
        with torch.cuda.device(args.gpu_index):
            train(args.epoch_num, args.batch_size, args.learning_rate, args.gpu_index, args.output_dir, args.pretrained_model)
    else: 
        train(args.epoch_num, args.batch_size, args.learning_rate, args.gpu_index, args.output_dir, args.pretrained_model)
        
    