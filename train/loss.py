import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits, labels):
        """
        Parameters
        ----------
        logits: input logits
        labels: multi-hot labels
        """
        print(logits)
        logits = F.normalize(logits)
        x = logits @ logits.T  # cosine similarity 对称矩阵
        y = (labels @ labels.T > 0).float()# 标签向量的内积是否>0判断相似与否
        #print(y)
        # Calculating Probabilities
        # x_sigmoid = torch.sigmoid(x)
        # xs_pos = x_sigmoid
        # xs_neg = 1 - x_sigmoid
        xs_pos = x
        xs_neg = 1 - x

        # Asymmetric Clipping
        # Asymmetric Probability Shifting
        if self.clip is not None and self.clip > 0: # 是否剪切操作
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))  # L+
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))  # L-
        loss = los_pos + los_neg

        # Asymmetric Focusing-非对称焦点调整
        # calc Eq. (7) in a matrix way
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            # 计算不对称的γ w-权重系数
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()
