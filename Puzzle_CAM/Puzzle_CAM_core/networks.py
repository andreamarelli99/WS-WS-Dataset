import torch.nn as nn
import torch.nn.functional as F
import timm

import torch.utils.model_zoo as model_zoo

from .arch_resnet import resnet
from .abc_modules import ABC_Model


#######################################################################
# Normalization
#######################################################################

class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

def group_norm(features):
    return nn.GroupNorm(4, features)
#######################################################################


class Backbone(nn.Module, ABC_Model):
    def __init__(self, model_name, num_classes=20, mode='fix', segmentation=False):
        super().__init__()

        self.mode = mode

        if self.mode == 'fix': 
            self.norm_fn = FixedBatchNorm
        else:
            self.norm_fn = nn.BatchNorm2d

        if 'efficientnet' in model_name:
            self.model = timm.create_model(model_name, pretrained=True)

            del self.model.global_pool
            del self.model.classifier

            self.stages = []

            for name, module in self.model.named_children():
                self.stages.append(getattr(self.model, name))

            self.num_features = self.model.num_features 
        
        else:
            if 'resnet' in model_name:
                self.model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[model_name], strides=(2, 2, 2, 1), batch_norm_fn=self.norm_fn)

                state_dict = model_zoo.load_url(resnet.urls_dic[model_name])
                state_dict.pop('fc.weight')
                state_dict.pop('fc.bias')
                

                self.model.load_state_dict(state_dict)
            else:
                if segmentation:
                    dilation, dilated = 4, True
                else:
                    dilation, dilated = 2, False

                self.model = eval("resnest." + model_name)(pretrained=True, dilated=dilated, dilation=dilation, norm_layer=self.norm_fn)

                del self.model.avgpool
                del self.model.fc

            self.num_features = 2048
            
            self.stages = nn.ModuleList()
            temp_stage = []
            stage_zero_done = False

            for name, module in self.model.named_children():

                if "layer" in name:
                    if not stage_zero_done:
                        stage_zero_done = True
                        self.stages.append(nn.Sequential(*temp_stage))

                    self.stages.append(nn.Sequential(getattr(self.model, name)))
                else:
                    temp_stage.append(getattr(self.model, name))

class Classifier(Backbone):
    
    def __init__(self, model_name, num_classes=20, mode='fix'):
        super().__init__(model_name, num_classes, mode)

        self.classifier = nn.Conv2d(self.num_features, num_classes, 1, bias=False) #2048
        self.num_classes = num_classes

        self.initialize([self.classifier])
    
    def forward(self, x, with_cam=False):
        
        for i in range(len(self.stages)):
            x = self.stages[i](x)
        
        if with_cam:
            features = self.classifier(x)
            logits = self.global_average_pooling_2d(features)
            return logits, features
        else:
            x = self.global_average_pooling_2d(x, keepdims=True) 
            logits = self.classifier(x).view(-1, self.num_classes)
            return logits
        
