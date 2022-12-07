import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class Loss(nn.Module):
    def __init__(self, z_dim, opt):
        super().__init__()
        self.prior_z = MultivariateNormal(
            torch.zeros(z_dim).to(opt.device),
            torch.eye(z_dim).to(opt.device)
        )

    def forward(self, z, log_det_jacobian):
        return -1 * torch.mean(self.prior_z.log_prob(z) + log_det_jacobian)
