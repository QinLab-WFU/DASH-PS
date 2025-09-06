import torch
import torch.nn.functional as F
from torch import nn


def pairwise_mahalanobis(X, means, log_vars):
    """
    Computes pairwise squared Mahalanobis distances between X (data points) and a set of distributions
    :param X: [N, F] where N is the batch size and F is the feature dimension
    :param means: [C, F] C is the number of classes
    :param log_vars: [C, F] C is the number of classes, we assume a diagonal covariance matrix
    :return: pairwise squared Mahalanobis distances... [N, C, F] matrix
    i.e., M_ij = (x_i-means_j)\top * inv_cov_j * (x_i - means_j)

    """
    batch_size = X.size(0)
    n_classes = means.size(0)

    new_X = torch.unsqueeze(X, dim=1)  # [N, 1, F]
    new_X = new_X.expand(-1, n_classes, -1)  # [N, C, F]

    new_means = torch.unsqueeze(means, dim=0)  # [1, C, F]
    new_means = new_means.expand(batch_size, -1, -1)  # [N, C, F]

    # pairwise distances
    diff = new_X - new_means

    # convert log_var to covariance
    covs = torch.unsqueeze(torch.exp(log_vars), dim=0)  # [1, C, F]

    # the squared Mahalanobis distances
    M = torch.div(diff.pow(2), covs).sum(dim=-1)  # [N, C]

    return M


class PD4HG(nn.Module):
    """
    Prototypical Distributions for Hypergraph
    """

    def __init__(self, n_classes, n_bits, tau=32, alpha=0.9):
        super().__init__()
        # Parameters (means and covariance)
        self.means = nn.Parameter(torch.Tensor(n_classes, n_bits))
        self.log_vars = nn.Parameter(torch.Tensor(n_classes, n_bits))

        # Initialization
        nn.init.kaiming_normal_(self.means, mode="fan_out")
        nn.init.kaiming_normal_(self.log_vars, mode="fan_out")

        self.tau = tau
        self.alpha = alpha

    def forward(self, X, T):
        """
        X: [B, K] where B is the batch size and K is the feature dimension
        T: labels of each distribution (B x C matrix)
        """
        mu = self.means
        log_vars = F.relu6(self.log_vars)

        # L2 normalize
        X = F.normalize(X, p=2, dim=-1)
        mu = F.normalize(mu, p=2, dim=-1)

        # Compute pairwise mahalanobis distances (B x C matrix)
        D = pairwise_mahalanobis(X, mu, log_vars)

        # Distribution loss: L_D
        P_all = F.softmax(-1 * self.tau * D, dim=1)  # B x C
        P_sum = torch.sum(P_all * T, dim=1)
        loss = -torch.log(P_sum[P_sum != 0]).mean()

        # Constructing Semantic Tuples
        class_within_batch = torch.nonzero(T.sum(dim=0) != 0).squeeze(dim=1)
        exp_term = torch.exp(-1 * self.alpha * D[:, class_within_batch])
        # semantic tuple S: Eq. (4)
        # S: B x C* -> H: |V| x |E|
        S = T[:, class_within_batch] + exp_term * (1 - T[:, class_within_batch])

        return loss, S


if __name__ == "__main__":
    print(torch.log(torch.tensor([1.0, 0, 2.0])).mean())
    exit(-1)
