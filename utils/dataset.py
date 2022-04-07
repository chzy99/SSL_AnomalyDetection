'''
Created: 2022-03-09 15:27:15
Author : Ziyi
Email : z1chen@whu.edu.cn
-----
Description: Finetune dataset and anomaly detection dataset
'''

from utils.logger import Logger

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, ConcatDataset

log = Logger().getLogger(module_name='__dataset.py__')

class DataPipeline:
    def __init__(self, params, is_rotate):
        backbone = params['backbone']
        self.img_size = int(params['input_shape'].split(',')[0])
        self.kernel_size = int(0.1 * self.img_size)
        log.info('Dataset img_size: %d, kernel_size: %d' %(self.img_size, self.kernel_size))
        # data aug
        self.is_rotate = is_rotate
        self.n_views = params['n_views']

    def __get_transforms(self, s=1):
        sigma = np.random.uniform(0.1, 2.0)
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        soft_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.img_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=sigma),
                                              transforms.ToTensor()])
        return soft_transforms
        
    def __call__(self, image):
        if not self.is_rotate:
            soft_transform = self.__get_transforms()
            return [soft_transform(image) for i in range(self.n_views)]
        else:
            hard_transform_1 = transforms.Compose([transforms.RandomResizedCrop(size=self.img_size),
                                                  transforms.RandomRotation(degrees=90),
                                                  transforms.ToTensor()])
            hard_transform_2 = transforms.Compose([transforms.RandomResizedCrop(size=self.img_size),
                                                   transforms.RandomRotation(degrees=180),
                                                   transforms.ToTensor()])
            return [hard_transform_1(image), hard_transform_2(image)]

class Dataset:
    def __init__(self, params):
        self.root = '../dataset/'
        self.dataset_map = {
            'cifar10': lambda: [datasets.CIFAR10(self.root, train=True, transform=DataPipeline(params, rot), download=True) for rot in range(2)]
        }
    
    ''' get finetune dataset '''
    def get_ft_dataset(self, name):
        dataset_list = self.dataset_map[name.lower()]()
        concat_dataset = ConcatDataset(dataset_list)
        return concat_dataset

class ADDataset(object):
    def __init__(self, normal_classes: int = 5):
        
        self.root = '../dataset/'
        self.n_classes = 2
        self.normal_classes = tuple([normal_classes])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_classes)
        self.outlier_classes = tuple(self.outlier_classes)

        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        self.train_set = datasets.CIFAR10(root=self.root, train=True, transform=transform, target_transform=target_transform,
                              download=True)
                              
        idxs = (torch.tensor(self.train_set.targets)[..., None] == torch.tensor([normal_classes])).any(-1).nonzero(as_tuple=True)[0]
        self.train_set = Subset(self.train_set, idxs)

        self.test_set = datasets.CIFAR10(root=self.root, train=False, transform=transform, target_transform=target_transform,
                                  download=True)