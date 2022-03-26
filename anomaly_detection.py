'''
Created: 2022-03-24 16:40:02
Author : Ziyi
Email : z1chen@whu.edu.cn
-----
Description: Graduation Design
'''

from utils.dataset import ADDataset
from utils.logger import Logger
from models.simclr import Simclr
from models.ocsvm import OCSVM

from absl import flags
import torch

# data
flags.DEFINE_string(name='dataset', default='cifar10', help='in-distribution dataset')
flags.DEFINE_integer(name='num_classes', default=10, help='')
flags.DEFINE_integer(name='normal_class', default=5, help='')

# model
flags.DEFINE_string(name='backbone', default='resnet50', help='')
flags.DEFINE_float(name='nu', default=0.1, help='backbone architecture of feature extractor')
flags.DEFINE_string(name='kernel', default='linear', help='')

# train
flags.DEFINE_integer(name='num_of_workers', default=8, help='')
flags.DEFINE_integer(name='batch_size', default=512, help='')
flags.DEFINE_integer(name='seed', default=123, help='')
flags.DEFINE_integer(name='gpu_idx', default=0, help='')

FLAGS = flags.FLAGS
log = Logger().getLogger("__detection__")

def main():
    params = {flag.name: flag.value for flag in FLAGS.get_flags_for_module('__main__')}

    if torch.cuda.is_available():
        log.info("main: cuda is available")
        params['device'] = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        params['device'] = torch.device('cpu')

    # load model pretrained weights
    model = Simclr(params['backbone'], num_classes=params['num_classes'], is_pretrained=False)
    # print(torch.load('runs/Mar25_14-06-47_admin4/checkpoint_0200.pth.tar').keys())
    model.load_state_dict(torch.load('runs/Mar25_14-06-47_admin4/checkpoint_0200.pth.tar')['state_dict'])
    model.eval()
    # init OCSVM model
    ocsvm = OCSVM(params, model)

    # prepare dataset for OCSVM(feature extraction, split...)
    dataset = ADDataset(root='../dataset/', normal_classes=params['normal_class'])

    # train and eval
    ocsvm.train(dataset)

    # ocsvm.test(dataset)

if __name__ == '__main__':
    main()