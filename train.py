import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils.transforms import SegmentationPresetTrain, SegmentationPresetEval
from utils.metrics import *

from config import Config
from model import *

import time


def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    config = Config()
    config.show_config()
    
    batch_size = config.batch_size
    val_batch_size = config.val_batch_size
    num_epoches = config.num_epoches
    num_eval_batch = config.num_eval_batch
    lr = config.lr
    T_max = config.T_max
    last_epoch = config.last_epoch
    save_path = config.save_path
    pretrained = config.pretrained
    model_type = config.model
    
    model_config = config.model_config
    model_config.show_config()

    num_classes = model_config.num_classes
    ignore_index = model_config.ignore_index
    
    if model_type == 'FCN':
        model = FCNet(model_config)
    elif model_type == 'SETR':
        model = SETR(model_config)
    elif model_type == 'DeepLab':
        version = model_config.version
        if model_config == '1':
            model = DeepLabV1(num_classes)
        elif model_config == '2':
            model = DeepLabV2(num_classes)
        elif model_config == '3':
            model = DeepLabV3(num_classes)
        else:
            model = DeepLabV3p(num_classes)
        
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained, map_location='cpu'))
        
    model.to(device)
    
    if model_config.dataset == 'VOC':
        train_dataset = torchvision.datasets.VOCSegmentation('./data',
                                         year='2012',
                                         image_set='train',
                                         download=False,
                                         transforms=SegmentationPresetTrain(640, 512))

        val_dataset = torchvision.datasets.VOCSegmentation('./data',
                                             year='2012',
                                             image_set='val',
                                             download=False,
                                             transforms=SegmentationPresetEval(512))
        
    elif model_config.dataset == 'Cityscapes':
        train_dataset = torchvision.datasets.Cityscapes('./data/Cityscapes',
                                        split='train',
                                        mode='fine',
                                        target_type='semantic',
                                        transforms=SegmentationPresetTrain(1024, 512))

        val_dataset = torchvision.datasets.Cityscapes('./data/Cityscapes',
                                       split='val',
                                       mode='fine',
                                       target_type='semantic',
                                       transforms=SegmentationPresetEval(512))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

    print("{:d} training images in {:d} batches".format(len(train_dataset), len(train_dataloader)))
    print("{:d} validation images in {:d} batches".format(len(val_dataset), len(val_dataloader)))
    

    CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=model_config.ignore_index)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=T_max, last_epoch=last_epoch)

    loss_record = []
    val_loss_record = []
    acc_record = []
    mIoU_record = []

    tick =  time.time()

    best_mIoU = -1
    best_epoch = -1

    for epoch in range(num_epoches):
        print("Epoch: {:3d}".format(epoch + 1))
        tick_e = time.time()
        print("------------------ Training ------------------")
        for step, (X, y) in enumerate(train_dataloader):
            tick_i = time.time()
            
            model.train()
            optim.zero_grad()

            pred = model(X.to(device))
            loss = CrossEntropyLoss(pred, y.to(device))

            loss.backward()
            optim.step()

            loss_record.append(loss.item())

            print("Iteration {:3d} |\tLoss: {:.4f}\tTime:{: .2f}s (Total: {:.2f}s)".format(step,
                                                                 loss.item(),
                                                                 time.time() - tick_i,
                                                                 time.time() - tick))
            
            #print(time.time() - tick_i)
            #if step == 5:
            #    break

        print("------------------ Evaluation ----------------")
        model.eval()

        tick_eval = time.time()
        val_loss = 0
        total_area_intersect, total_area_union, total_area_pred_label, total_area_label = 0, 0, 0, 0

        for step, (X, y) in enumerate(val_dataloader):
            with torch.no_grad():
                pred = model(X.to(device))
                val_loss += CrossEntropyLoss(pred, y.to(device)).item()

                label_pred = pred.argmax(dim=1)
                area_intersect, area_union, area_pred_label, area_label = total_intersect_and_union(label_pred.cpu().numpy(),
                                                                        y.numpy(),
                                                                        num_classes=num_classes,
                                                                        ignore_index=ignore_index)
                total_area_intersect += area_intersect
                total_area_union += area_union
                total_area_pred_label += area_pred_label
                total_area_label += area_label

            if step + 1 == num_eval_batch:
                break

        val_loss /= (step + 1)
        val_loss_record.append(val_loss)

        acc = total_area_intersect.sum() / total_area_label.sum()
        mIoU = (total_area_intersect / total_area_union).mean()

        acc_record.append(acc)
        mIoU_record.append(mIoU)    

        if mIoU > best_mIoU:
            best_mIoU = mIoU
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

        print("Evaluation completed in {:.2f}s (Epoch: {:.2f}s\tTotal: {:.2f}s)".format(time.time() - tick_eval,
                                                              time.time() - tick_e,
                                                              time.time() - tick))
        print("Loss = {:.3f}\tPixelwise accuracy = {:.2f}\tmIoU = {:.2f}".format(val_loss, acc, mIoU))
        print("Best mIoU: {:.2f} in epoch {:3d}".format(best_mIoU, best_epoch + 1))
        print("==============================================")
        scheduler.step()


    '''
    Plot the results    
    '''

    plt.plot(np.arange(1, len(loss_record) + 1) / len(train_dataloader), loss_record)
    plt.plot(val_loss_record)

    plt.xticks(np.arange(1, num_epoches + 1, 10))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.savefig('loss.png', format='png')
    plt.clf()

    plt.plot(acc_record)

    plt.xticks(np.arange(1, num_epoches + 1, 10))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.savefig('accuracy.png', format='png')
    plt.clf()

    plt.plot(mIoU_record)

    plt.xticks(np.arange(1, num_epoches + 1, 10))
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')

    plt.savefig('mIoU.png', format='png')
    plt.clf()

    '''
    Evaluate the whole model    
    '''
    model.eval()

    tick_eval = time.time()
    total_area_intersect, total_area_union, total_area_pred_label, total_area_label = 0, 0, 0, 0
    for step, (X, y) in enumerate(val_dataloader):
        with torch.no_grad():
            label_pred = model(X.to(device)).argmax(dim=1)
            area_intersect, area_union, area_pred_label, area_label = total_intersect_and_union(label_pred.cpu().numpy(),
                                                                    y.numpy(),
                                                                    num_classes=num_classes,
                                                                    ignore_index=ignore_index)
            total_area_intersect += area_intersect
            total_area_union += area_union
            total_area_pred_label += area_pred_label
            total_area_label += area_label

    acc = total_area_intersect.sum() / total_area_label.sum()
    mIoU = (total_area_intersect / total_area_union).mean()
    print("Evaluation completed in {:.2f}s".format(time.time() - tick_eval))
    print("Pixelwise accuracy = {:.2f}\tmIoU = {:.2f}".format(acc, mIoU))

if __name__ == '__main__':
    train()
