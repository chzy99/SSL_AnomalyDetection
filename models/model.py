'''
Created: 2022-03-10 12:43:50
Author : Ziyi
Email : z1chen@whu.edu.cn
-----
Description: Graduation Design
'''

from utils.logger import Logger

from torch import nn
import timm

log = Logger().getLogger("__model__")

class Model(nn.Module):
    def __init__(self, model_name, out_classes, is_pretrained):
        super().__init__()
        self.model_map = {
            'resnet18': lambda: timm.create_model('resnet18', pretrained=is_pretrained, num_classes=out_classes),
            'resnet50': lambda: timm.create_model('resnet50', pretrained=is_pretrained, num_classes=out_classes),
            'vit': lambda: timm.create_model('vit_base_patch16_224', pretrained=is_pretrained, num_classes=out_classes)
        }
        self.model = self.__get_encoder(model_name)
        feat = self.model.fc.in_features
        self.mlp = nn.Linear(feat, feat)
        self.model.fc = nn.Sequential(self.mlp, nn.ReLU(), self.model.fc)

    def __get_encoder(self, model_name):
        try:
            model_fn = self.model_map[model_name.lower()]
        except KeyError:
            log.error("Model.get_encoder: invalid model name %s" % model_name)
        return model_fn()

    def forward(self, x):
        return self.model(x)
