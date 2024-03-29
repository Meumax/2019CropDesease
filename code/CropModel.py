# coding: utf-8
from functools import partial
from torch import nn
import torchvision.models as M
import pretrainedmodels

class InceptionV3Finetune(nn.Module):
    finetune = True

    def __init__(self, num_classes: int):
        super().__init__()
        # self.net = M.inception_v3(pretrained=True)
        self.net = M.inception_v3(pretrained=False)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        if self.net.training:
            x, _aux_logits = self.net(x)
            return x
        else:
            return self.net(x)

class DenseNetFinetune(nn.Module):
    finetune = True
    
    def __init__(self, num_classes, net_cls=M.densenet121):
        super().__init__()
        # self.net = net_cls(pretrained=True)
        self.net = net_cls(pretrained=False)
        self.net.classifier = nn.Linear(self.net.classifier.in_features, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        return self.net(x)

class ResNetFinetune(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls=M.resnet50, dropout=False):
        super().__init__()
        # self.net = net_cls(pretrained=True)
        self.net = net_cls(pretrained=False)
        self.bn=nn.BatchNorm2d(3)
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.avgpool=nn.AdaptiveAvgPool2d(1)
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
      #  x=self.bn(x)        
        return self.net(x)

class FinetunePretrainedmodels(nn.Module):
    finetune = True

    def __init__(self, num_classes: int, net_cls, net_kwards):
        super().__init__()
        self.net = net_cls(**net_kwards)
        self.net.last_linear = nn.Linear(self.net.last_linear.in_features, num_classes)

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)
        

resnet50_finetune = partial(ResNetFinetune, net_cls=M.resnet50)
resnet101_finetune = partial(ResNetFinetune, net_cls=M.resnet101)
resnet152_finetune = partial(ResNetFinetune, net_cls=M.resnet152)       

densenet121_finetune = partial(DenseNetFinetune, net_cls=M.densenet121)
densenet161_finetune = partial(DenseNetFinetune, net_cls=M.densenet161)
densenet201_finetune = partial(DenseNetFinetune, net_cls=M.densenet201)        

# inceptionresnetv2_finetune = partial(FinetunePretrainedmodels,
                                     # net_cls=pretrainedmodels.inceptionresnetv2,
                                     # net_kwards={'pretrained': 'imagenet+background', 'num_classes': 1001})        

inceptionresnetv2_finetune = partial(FinetunePretrainedmodels,
                                     net_cls=pretrainedmodels.inceptionresnetv2,
                                     net_kwards={'pretrained': False, 'num_classes': 1000}) 
                                     
# inceptionv4_finetune = partial(FinetunePretrainedmodels,
                               # net_cls=pretrainedmodels.inceptionv4,
                               # net_kwards={'pretrained': 'imagenet+background', 'num_classes': 1001})     

inceptionv4_finetune = partial(FinetunePretrainedmodels,
                               net_cls=pretrainedmodels.inceptionv4,
                               net_kwards={'pretrained': False, 'num_classes': 1000}) 
                           