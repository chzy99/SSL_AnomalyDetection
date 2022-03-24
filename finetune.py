'''
Created: 2022-03-11 13:57:53
Author : Ziyi
Email : z1chen@whu.edu.cn
-----
Description: Training SimCLR
'''

from absl import flags

import torch
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from utils.logger import Logger
from utils.dataset import Dataset
from models.simclr import Simclr

from trainer import Trainer

log = Logger().getLogger("__finetune__")

# data
flags.DEFINE_string(name='dataset', default='cifar10', help='in-distribution dataset')
flags.DEFINE_integer(name='n_labels', default=10, help='')
flags.DEFINE_string(name='input_shape', default='32,32,3', help='input image shape')

# model
flags.DEFINE_string(name='backbone_arch', default='resnet50', help='backbone architecture of feature extractor')
flags.DEFINE_integer(name='n_views', default=2, help='')

# train
flags.DEFINE_integer(name='epochs', default=200, help='')
flags.DEFINE_integer(name='batch_size', default=256, help='')
flags.DEFINE_float(name='lr', default=3e-4, help='')
flags.DEFINE_float(name='weight_decay', default=1e-4, help='')
flags.DEFINE_float(name='temperature', default=7e-2, help='')
flags.DEFINE_integer(name='log_every_n_steps', default=100, help='')

flags.DEFINE_integer(name='seed', default=None, help='')
flags.DEFINE_integer(name='gpu_idx', default=0, help='')

FLAGS = flags.FLAGS

def get_training_component(hparams, train_loader):
    model = Simclr(hparams['backbone_arch'], is_pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), hparams['lr'], weight_decay=hparams['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss().to(hparams['device'])
    scaler = GradScaler(enabled=True)
    writer = SummaryWriter()

    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'scaler': scaler,
        'writer': writer
    }

def main():
    hparams = {flag.name: flag.value for flag in FLAGS.get_flags_for_module('__main__')}
    log.info("Training hyperparameters: {}".format(hparams))

    # use cuda by default
    if torch.cuda.is_available():
        log.info("main: cuda is available")
        hparams['device'] = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        hparams['device'] = torch.device('cpu')
    
    dataset = Dataset().get_dataset(name=hparams['dataset'], is_train=True)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=hparams['batch_size'], shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True
    )

    component = get_training_component(hparams=hparams, train_loader=train_loader)

    with torch.cuda.device(hparams['gpu_idx']):
        trainer = Trainer(component, hparams)
        trainer.train(train_loader)

if __name__ == '__main__':
    main()