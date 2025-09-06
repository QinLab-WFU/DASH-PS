from argparse import Namespace

import torch
import torch.nn as nn
import torchvision
from torch import Tensor


def build_model(args: Namespace, pretrained=True):
    if args.backbone == "resnet50":
        net = ResNet50(args.n_bits, pretrained, use_LN=args.use_LN, freeze_BN=args.freeze_BN, add_GMP=args.add_GMP)
    else:
        raise NotImplementedError(f"not support: {args.backbone}")
    return net.cuda()


class ResNet50(nn.Module):
    def __init__(self, n_bits, pretrained=True, **kwargs):
        super().__init__()

        self.use_LN = kwargs.pop("use_LN", False)
        self.add_GMP = kwargs.pop("add_GMP", False)
        self.need_feature = kwargs.pop("need_feature", False)

        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = torchvision.models.resnet50(weights=weights)
        self.dim_feature = self.model.fc.in_features #输入特征的维度
        if n_bits != self.model.fc.out_features:
            self.model.fc = nn.Linear(self.dim_feature, n_bits)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_out")
            nn.init.constant_(self.model.fc.bias, 0) # 如果维度不同，换成新的线性层
        else:
            print("use default fc")

        if self.use_LN:
            self.layer_norm = nn.LayerNorm(n_bits, elementwise_affine=False)

        if self.add_GMP:
            self.model.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))

        if kwargs.pop("freeze_BN", False):
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.train = lambda _: None
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def forward(self, x: Tensor):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        if self.add_GMP:
            x = self.model.avgpool(x) + self.model.maxpool2(x)
        else:
            x = self.model.avgpool(x)

        features = torch.flatten(x, 1)
        logits = self.model.fc(features)

        if self.use_LN:
            logits = self.layer_norm(logits)

        if self.need_feature:
            return features, logits

        return logits


class HGNN(nn.Module):
    """
    Hypergraph Neural Networks (AAAI 2019)
    we used two layers of HGNN
    """

    def __init__(self, n_classes, n_bits, dim_hidden):
        super(HGNN, self).__init__()
        # layer1
        self.theta1 = nn.Linear(n_bits, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.lrelu = nn.LeakyReLU(0.1)
        # layer2
        self.theta2 = nn.Linear(dim_hidden, n_classes)

    def compute_G(self, H):
        # the number of hyperedge
        n_edges = H.size(1)
        # the weight of the hyperedge
        we = torch.ones(n_edges, device=H.device)
        # the degree of the node
        Dv = (H * we).sum(dim=1)
        # the degree of the hyperedge
        De = H.sum(dim=0)

        We = torch.diag(we)
        inv_Dv_half = torch.diag(torch.pow(Dv, -0.5))
        inv_De = torch.diag(torch.pow(De, -1))
        H_T = torch.t(H)

        # propagation matrix
        # torch.chain_matmul is deprecated, use torch.linalg.multi_dot
        # G = torch.chain_matmul(inv_Dv_half, H, We, inv_De, H_T, inv_Dv_half)
        G = torch.linalg.multi_dot((inv_Dv_half, H, We, inv_De, H_T, inv_Dv_half))

        return G

    def forward(self, X, H):
        G = self.compute_G(H)

        # 1st layer
        X = G.matmul(self.theta1(X))
        # Function σ(.) denotes a non-linear activation: 2 lines below
        X = self.bn1(X)
        X = self.lrelu(X)

        # 2nd layer
        out = G.matmul(self.theta2(X))

        return out


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    _N = 12

    _args = Namespace(
        backbone="resnet50",
        n_classes=10,
        freeze_BN=True,
        n_bits=32,
        use_LN=True,
        add_GMP=True,
    )

    net = build_model(_args, True)

    _images = torch.randn((_N, 3, 224, 224)).cuda()
    _targets = torch.randint(_args.n_classes, (_N,)).cuda()
    _labels = torch.nn.functional.one_hot(_targets, _args.n_classes).float()

    _logits = net(_images)
    print(_logits.shape)

    from loss import PD4HG

    criterion = PD4HG(n_classes=_args.n_classes, n_bits=_args.n_bits).cuda()

    dist_loss, H = criterion(_logits, _labels)
    print("H.shape", H.shape)  # TODO: H.shape[1] may less than C
    print("dist_loss", dist_loss)

    hgnn = HGNN(n_classes=_args.n_classes, n_bits=_args.n_bits, dim_hidden=512).cuda()
    out = hgnn(_logits, H)
    print(out.shape)

    # net.train()
    # for x in net.modules():
    #     if isinstance(x, nn.BatchNorm2d):
    #         if x.training or x.weight.requires_grad or x.bias.requires_grad:
    #             print("training", x.training)
    #             print("weight.requires_grad", x.weight.requires_grad)
    #             print("bias.requires_grad", x.bias.requires_grad)
