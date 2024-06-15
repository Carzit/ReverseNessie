import torch
import torch.nn as nn
import torch.nn.functional as F
from DistributionMixture import DistributionMixture

class ScaleNormalize(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, input:torch.Tensor):
        sum_dim = input.sum(dim=self.dim, keepdim=True)
        sum_dim[sum_dim == 0] = 1
        scale_factor = 1.0 / sum_dim
        normalized_tensor = input * scale_factor
        return normalized_tensor

class EncoderLoss(nn.Module):
    def __init__(self, distribution_mixture:DistributionMixture, scale_normal:bool):
        super().__init__()
        self.distribution_mixture:DistributionMixture = distribution_mixture
        self.scale_normal = scale_normal

    def forward(self, w, r, p, x_seq, probe_seq):
        probe_seq = F.relu(probe_seq)
        if self.scale_normal:
            probe_seq = ScaleNormalize(dim=0)(probe_seq)

        nb_distribution = self.distribution_mixture(w, r, p)
        probe_pred = nb_distribution(x_seq)
        probe_true = probe_seq

        loss = probe_true * (torch.log(probe_true) - torch.log(probe_pred))
        loss = loss.sum(dim=-1)#shape=[batch_size]
        loss = loss.mean()

        return loss


class DecoderLoss(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, cme_para_pred:torch.Tensor, cme_para_true:torch.Tensor):
        loss = (cme_para_pred - cme_para_true).norm(self.p, dim=0)
        loss = loss.mean()
        return loss
    
