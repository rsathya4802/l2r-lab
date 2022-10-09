import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from src.encoders.base import BaseEncoder
from src.config.yamlize import yamlize
from src.constants import DEVICE

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, eps=1):
        inputs = nn.Flatten()(inputs[:, 0])
        targets = nn.Flatten()(targets)
        dice = 1 - (2*(inputs * targets).sum() + eps)/(inputs.sum() + targets.sum() + eps)  
        return dice


class EfficientNetV2Backbone(nn.Module):
    def __init__(self, loss=None, pretrained=False, *args, **kwargs):
        super().__init__()
        # remove Conv2dNormActivation, AvgPool, classifier
        # Howe doesn't use fused MBConv. V2 should, in theory.
        self.encoder = nn.Sequential(*list(torchvision.models.efficientnet_v2_s().features._modules.values())[:-1])
        self.hiddens = OrderedDict()
        if pretrained == True:
            raise NotImplementedError
            

    def forward(self, x):
        self.hiddens = OrderedDict()
        for i, block in enumerate(self.encoder.children()):
            x = block(x) 
            if i > len(self.encoder) - 5: # number of feature pyramid skips
                self.hiddens[str(i)] = x
        return x



class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.layers(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x

    
class SegmentationBranch(nn.Module):
    # Had to use these num_upsample values to get the right dimensions for efficientnetv2_s
    def __init__(self, channels, n_classes, num_upsamples=[1,2,2,3]):
        super().__init__()
        self.layers = [nn.Sequential(*[UpsampleBlock(channels, channels).to(DEVICE) for _ in range(i)]) for i in num_upsamples]
        self.head = nn.Conv2d(channels, n_classes, 1, padding=0)
        self.softmax = nn.Softmax2d()
        
    
    def forward(self, hidden_list):
        tmp = []
        for layer, x in zip(self.layers, hidden_list):
            tmp.append(layer(x))
        x = torch.stack(tmp, dim=0).sum(dim=0)
        x = self.head(x)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)
        x = self.softmax(x)
        return x
        
@yamlize
class FPNSegmentation(BaseEncoder, nn.Module):
    def __init__(
            self,
            n_classes: int = 2,
            fpn_filters: list = [64, 128, 160, 256],
            out_channels: int = 128,
    ):
        super().__init__()
        self.encoder = EfficientNetV2Backbone()
        self.feature_pyramid =  torchvision.ops.FeaturePyramidNetwork(fpn_filters, out_channels)
        self.segmentation_branch = SegmentationBranch(out_channels, n_classes)
        self.loss = DiceLoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.feature_pyramid(self.encoder.hiddens)
        x = self.segmentation_branch(list(x.values()))
        return x


    def encode(self, x):
        # assume x is RGB image with shape (H, W, 3)
        x = torch.Tensor(x.transpose(2, 0, 1)) / 255
        segm = self.forward(x.unsqueeze(0))
        return segm




    
