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
import torchvision

from cityscape_dataset.dataset import CityscapeDataset
from config import DefaultConfig
from evaluate import mean_iou, total_intersect_and_union

from SETR import Net, BabyNet

import time

import argparse

from torch.utils.tensorboard import SummaryWriter

def train(
    train_img_root,
    train_target_root,
    val_img_root,
    val_target_root,
    epoch_num,
    batch_size,
    learning_rate,
    output_path,
    gpu_index,
    pretrained_model,
    writer
    ):
    
    torch.backends.cudnn.enabled = False
    device = gpu_index
    
    SETRNet = Net(128, 19).to(device)
    # SETRNet = BabyNet()
    # if pretrained_model is not None:
    #     SETRNet.load_state_dict(torch.load(pretrained_model))
    
    train_data_set = CityscapeDataset(train_img_root, train_target_root, train=True, test=False)
    train_data_loader = Data.DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True)
    val_data_set = CityscapeDataset(val_img_root, val_target_root, train=False, test=False)
    val_data_loader = Data.DataLoader(dataset=val_data_set, batch_size=batch_size, shuffle=True)
    
    '''
    here is the start of tensorboard
    I load a test image 
    and the net
    '''
    # get some random training images
    dataiter = iter(train_data_loader)
    images, labels = dataiter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # write to tensorboard
    writer.add_image('test_image_show', img_grid)

    writer.add_graph(SETRNet, images)

    '''
    first tensorborad code end here
    '''

    print('Data loaded: {:d} train images and {:d} validation images'.format(len(train_data_set), len(val_data_set)))
    best_epoch = 0
    best_mIoU = 0    
    
    # CrossEntropyLoss = nn.CrossEntropyLoss()
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
            loss = CrossEntropyLoss(pred, batch_y.to(device))
            # loss = MSEloss(pred, batch_y.float())

            loss.backward()
            opt.step()
            loss += train_loss
            loss_record.append(loss.item())

            '''
            here is the tensorborad recording the loss during the training
            '''
            writer.add_scalar('training loss',
                                loss / 1000,
                                epoch * len(train_data_loader) + step)
            
        #log = 'Epoch:{0}\tLoss: {loss.avg:.4f}\t'.format(epoch, loss=train_loss)
        #print(log + '\n')
        train_loss /= step + 1
        
        
        SETRNet.eval()
        with torch.no_grad():
            total_area_intersect, total_area_union, total_area_pred_label, total_area_label = np.zeros(19), np.zeros(19), np.zeros(19), np.zeros(19)
            for step, (batch_x, batch_y) in enumerate(val_data_loader):
                pred = SETRNet(batch_x.to(device))
                area_intersect, area_union, area_pred_label, \
                    area_label = total_intersect_and_union(pred.cpu().numpy().argmax(axis=1), batch_y.numpy(), 19, -1)
                total_area_intersect += area_intersect
                total_area_union += area_union
                total_area_pred_label += area_pred_label
                total_area_label += area_label
            all_acc = total_area_intersect.sum() / total_area_label.sum()
            iou = total_area_intersect[total_area_union > 0] / total_area_union[total_area_union > 0]
            mIoU = iou.sum() / (len(iou) + 1)
        
        print('----------------------------')
        print("Epoch={:3d}\tTrain_loss={:6.4f}\tIoU={:.2f}\tAcc={:.2f}".format(epoch, train_loss, mIoU, all_acc))
        if mIoU > best_mIoU:
            best_IoU = mIoU
            best_epoch = epoch
            # torch.save(SETRNet.state_dict(), output_path)
        print('Best Epoch: ', best_epoch, ' Best mIoU: ', best_mIoU)
        print("Time: {:5.2f}(Total: {:5.2f})".format(time.time() - tick_e, time.time() - tick))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    #return SETRNet

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

    writer = SummaryWriter('D:/PycharmProjects/cv_mid_project/runs/cityscapes_experiment_1')
    args = DefaultConfig()
    if args.gpu_index is not None:
        with torch.cuda.device(args.gpu_index):
            train(
                args.train_img_root,
                args.train_target_root,
                args.val_img_root,
                args.val_target_root,
                args.epoch_num,
                args.batch_size,
                args.learning_rate,
                args.output_path,
                args.gpu_index,
                args.pretrained_model,
                writer,
            )
    else: 
        train(
            args.train_img_root,
            args.train_target_root,
            args.val_img_root,
            args.val_target_root,
            args.epoch_num,
            args.batch_size,
            args.learning_rate,
            args.output_path,
            args.gpu_index,
            args.pretrained_model,
            writer,
        )
        
    
