'''
Created: 2022-03-09 15:27:15
Author : Ziyi
Email : z1chen@whu.edu.cn
-----
Description: Dataset
'''

from utils.logger import Logger

import numpy as np
from torchvision import datasets, transforms

log = Logger().getLogger(module_name='__dataset__')

class DataPipeline:
    def __init__(self, params):
        model_name = params['backbone_arch']
        self.size = int(params['input_shape'].split(',')[0])
        self.kernel_size = int(0.1 * self.size)
        if 'vit' in model_name:
            self.size = 224
            self.kernel_size = 21
        log.info('img_size: %d, kernel_size: %d' %(self.size, self.kernel_size))
        self.n_views = 2

    def __get_transforms(self, s=1):
        sigma = np.random.uniform(0.1, 2.0)
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=sigma),
                                            transforms.ToTensor()])
        return data_transforms
        
    def __call__(self, image):
        transform = self.__get_transforms()
        return [transform(image) for i in range(self.n_views)]

class Dataset:
    def __init__(self, params):
        self.root = '../dataset/'
        self.dataset_map = {
            'cifar10': lambda: datasets.CIFAR10(self.root, train=True, transform=DataPipeline(params=params), download=True),
        }
    
    ''' get finetune dataset '''
    def get_ft_dataset(self, name):
        log.info('Dataset.get_dataset name = %s' % name)
        dataset_fn = self.dataset_map[name.lower()]
        return dataset_fn()
        
    ''' get anomaly detection dataset '''
    def get_ad_dataset(self, normal_classes):
        return ADDataset(root=self.root, normal_classes=normal_classes)

class ADDataset(object):
    def __init__(self, root, normal_classes: int = 5):
        super().__init__(root)

        self.n_classes = 2
        self.normal_classes = tuple([normal_classes])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_classes)
        self.outlier_classes = tuple(self.outlier_classes)

        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        self.train_set = datasets.CIFAR10(root=self.root, train=True, transform=transform, target_transform=target_transform,
                              download=True)

        self.test_set = datasets.CIFAR10(root=self.root, train=False, transform=transform, target_transform=target_transform,
                                  download=True)