'''
Created: 2022-03-24 16:40:02
Author : Ziyi
Email : z1chen@whu.edu.cn
-----
Description: Graduation Design
'''

from utils.dataset import Dataset
from models.simclr import Simclr
from models.ocsvm import OCSVM

from absl import flags
import torch

# data
flags.DEFINE_string(name='dataset', default='cifar10', help='in-distribution dataset')
flags.DEFINE_integer(name='normal_classes', default=5, help='')

# model
flags.DEFINE_float(name='nu', default=0.1, help='backbone architecture of feature extractor')
flags.DEFINE_string(name='kernel', default='linear', help='')

# train
flags.DEFINE_integer(name='num_of_workers', default=8, help='')
flags.DEFINE_integer(name='batch_size', default=128, help='')
flags.DEFINE_integer(name='seed', default=123, help='')
flags.DEFINE_integer(name='gpu_idx', default=0, help='')

FLAGS = flags.FLAGS

def main():
    params = {flag.name: flag.value for flag in FLAGS.get_flags_for_module('__main__')}
    # load model pretrained weights
    model = Simclr('resnet18', is_pretrained=False)
    model.load_state_dict(torch.load('xx'))
    # init OCSVM model
    ocsvm = OCSVM(params, model)

    # prepare dataset for OCSVM(feature extraction, split...)
    dataset = Dataset().get_ad_dataset(params['normal_classes'])

    # train and eval
    ocsvm.train(dataset)

    ocsvm.test(dataset)

if __name__ == '__main__':
    main()