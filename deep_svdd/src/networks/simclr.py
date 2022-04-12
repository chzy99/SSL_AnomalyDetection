from torch import nn
import timm

class Simclr(nn.Module):
    def __init__(self, model_name, num_classes, is_pretrained):
        super().__init__()
        self.model_map = {
            'effinet': lambda: timm.create_model('efficientnet_b0', pretrained=is_pretrained, num_classes=num_classes),
            'resnet18': lambda: timm.create_model('resnet18', pretrained=is_pretrained, num_classes=num_classes),
            'resnet50': lambda: timm.create_model('resnet50', pretrained=is_pretrained, num_classes=num_classes),
            'vit': lambda: timm.create_model('vit_base_patch16_224', pretrained=is_pretrained, num_classes=num_classes)
        }
        self.model = self.__get_model(model_name)
        self.__insert_mlp(model_name=model_name)

    def __get_model(self, model_name):
        try:
            model_fn = self.model_map[model_name.lower()]
        except KeyError:
            print("invalid model name %s" % model_name)
        return model_fn()
    
    def __insert_mlp(self, model_name):
        # TODO: hard code
        if 'resnet' in model_name:
            feat = self.model.fc.in_features
            print('model in-feature dim: {}'.format(feat))
            self.model.fc = nn.Sequential(nn.Linear(feat, feat), nn.ReLU(), self.model.fc)
        elif 'vit' in model_name:
            feat = self.model.head.in_features
            print('model in-feature dim: {}'.format(feat))
            self.model.head = nn.Sequential(nn.Linear(feat, feat), nn.ReLU(), self.model.head)
        elif 'eff' in model_name:
            feat = self.model.classifier.in_features
            print('model in-feature dim: {}'.format(feat))
            self.model.classifier = nn.Sequential(nn.Linear(feat, feat), nn.ReLU(), self.model.classifier)

    def extract_features(self, x):
        return self.model.forward_features(x)

    def forward(self, x):
        return self.model(x)