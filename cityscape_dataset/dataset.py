import torch
import torch.utils.data as Data 
import os
import torchvision
import numpy as np
from PIL import Image

class CityscapeDataset(Data.Dataset):
    def __init__(self, img_root, target_root, transforms=None, train=True, test = False):
        classes_list =[f'{img_root}/{oneClass}' for oneClass in os.listdir(img_root)]
        self.img = []
        for oneList in classes_list:
            [self.img.append(f'{oneList}/{filename}') for filename in os.listdir(oneList)]
        self.img = self.img[:50]

        classes_list_target =[f'{target_root}/{oneClass}' for oneClass in os.listdir(target_root)]
        self.target = []
        for oneList in classes_list_target:
            for filename in os.listdir(oneList):
                if 'label' in filename:
                    self.target.append(f'{oneList}/{filename}')
        self.target = self.target[:50]

        self.dataset = dict(zip(self.img, self.target))
        
        self.dataset_num = len(self.dataset)

        self.train = train
        self.test = test
        if not transforms:
            normalize = torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                     std = [0.229, 0.224, 0.225])
            self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        normalize])

    def __getitem__(self, index):
        img_sample = self.transforms(Image.open(self.img[index]))
        target_sample = torch.tensor(np.array(Image.open(self.dataset[self.img[index]])))
        return img_sample, target_sample

    def __len__(self):
        return self.dataset_num
