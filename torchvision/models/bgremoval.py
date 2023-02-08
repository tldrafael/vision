from collections import OrderedDict
import torch
import torch.nn as nn
from .detection import MaskRCNN_ResNet50_FPN_V2_Weights
from .resnet import resnet50
from ._utils import IntermediateLayerGetter
from .detection.transform import GeneralizedRCNNTransform


__all__ = ['BGRemovalDecoder', 'BGRemoval']


def conv_block(in_channels, out_channels, ks=3, padding='same'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, ks, padding=padding, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
    )


class ExpandBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = conv_block(in_channels + skip_channels, out_channels)
        self.conv2 = conv_block(out_channels, out_channels)

    def forward(self, x_bottom, x_skip):
        x_bottom = self.upsample(x_bottom)
        x = torch.cat([x_skip, x_bottom], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BGRemovalDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList()
        exp2_factor = [5, 4, 3, 2, 0]
        for i in range(len(exp2_factor) - 1):
            in_channels = 64 * 2 ** exp2_factor[i]
            out_channels = 64 * 2 ** exp2_factor[i + 1]
            self.blocks.append(
                ExpandBlock(in_channels, out_channels, out_channels)
            )

        self.conv2 = conv_block(64, 16, 3)
        self.conv3 = nn.Conv2d(16, 1, 1)
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, list_features):
        x = list_features[-1]
        for i, b in enumerate(self.blocks, 2):
            x = b(x, list_features[-i])

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.upsample3(x)
        x = self.sigmoid(x)
        return x


class BGRemoval(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.weights = MaskRCNN_ResNet50_FPN_V2_Weights.verify(weights)

        min_size=800,
        max_size=1333,
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        self.backbone = resnet50(weights=None)
        self.return_layers = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
        self.return_layers = {k:k for k in self.return_layers}
        self.backbone = IntermediateLayerGetter(self.backbone, return_layers=self.return_layers)
        self.decoder = BGRemovalDecoder()

        self.load_pretrained_weights()

    def load_pretrained_weights(self):
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.verify(weights)

        state_dict = weights.get_state_dict(True)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'body' in k:
                new_key = k.replace('body.', '')
                new_state_dict[new_key] = v

        self.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        x = self.backbone(x)
        x = list(x.values())
        x = self.decoder(x)
        return x
