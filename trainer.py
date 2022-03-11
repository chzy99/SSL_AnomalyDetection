'''
Created: 2022-03-11 16:13:57
Author : Ziyi
Email : z1chen@whu.edu.cn
-----
Description: Trainer
'''
import tqdm

import torch
from torch.cuda.amp import autocast
from utils.logger import Logger

log = Logger().getLogger('__trainer__')

class Trainer(object):
    def __init__(self, train_component, hparams):
        self.component = train_component
        self.params = hparams

    def loss(self):
        pass
    
    def train(self, data_loader):
        n_iter = 0
        component = self.component
        log.info("Trainer.train Start training with gpu %d" % self.params.gpu_idx)

        for epoch in range(self.params.epochs):
            for images, _ in tqdm(data_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.params.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                component.scaler.scale(loss).backward()

                component.scaler.step(self.optimizer)
                component.scaler.update()

                # if n_iter % self.args.log_every_n_steps == 0:
                    # top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    # self.writer.add_scalar('loss', loss, global_step=n_iter)
                    # self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    # self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    # self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch >= 10:
                self.scheduler.step()
            log.info(f"Epoch: {epoch}\tLoss: {loss}\tTop1 accuracy: {}")
        

        

