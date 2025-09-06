from argparse import Namespace

import torchvision
from torch import nn


def init_params(layer):
    nn.init.kaiming_normal_(layer.weight, mode="fan_out")
    nn.init.constant_(layer.bias, 0)


def build_model(args: Namespace, pretrained):
    if args.backbone == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, args.n_bits)
        if pretrained:
            init_params(net.fc)
    else:
        raise NotImplementedError(f"not support: {args.backbone}")
    return net.cuda()
