'''
Created: 2022-03-11 16:13:57
Author : Ziyi
Email : z1chen@whu.edu.cn
-----
Description: Trainer
'''

import os
import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from utils.logger import Logger
from utils.train_utils import save_training_params, save_checkpoint, accuracy

log = Logger().getLogger('__trainer__')

class Trainer(object):
    def __init__(self, train_component, hparams):
        self.component = train_component
        self.params = hparams

    def nt_xent_loss(self, features):
        labels = torch.cat([torch.arange(self.params['batch_size']) for i in range(self.params['n_views'])], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.params['device'])

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.params['device'])
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
 
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.params['temperature']
        return logits, labels
    
    def train(self, data_loader):
        n_iter = 0

        component = self.component
        model = component['model']
        optimizer = component['optimizer']
        scheduler = component['scheduler']
        criterion = component['criterion']
        scaler = component['scaler']
        writer = component['writer']

        save_training_params(writer.log_dir, self.params)
        log.info("Trainer.train Start training with gpu %d" % self.params['gpu_idx'])

        for epoch in range(self.params['epochs']):
            for images, _ in tqdm(data_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.params['device'])

                with autocast(enabled=True):
                    features = component.model(images)
                    logits, labels = self.nt_xent_loss(features)
                    loss = component.criterion(logits, labels)

                component.optimizer.zero_grad()

                component.scaler.scale(loss).backward()

                component.scaler.step(component.optimizer)
                component.scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch >= 10:
                component.scheduler.step()
            log.info(f"Epoch: {epoch}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        log.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        log.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
