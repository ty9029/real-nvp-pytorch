import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class CouplingLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, mask):
        super().__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)

        self.s_layers = Linear(in_dim, hidden_dim)

        self.t_layers = Linear(in_dim, hidden_dim)

    def forward(self, x):
        x_ = x * self.mask
        s = torch.tanh(self.s_layers(x_)) * (1 - self.mask)
        t = self.t_layers(x_) * (1 - self.mask)

        x = x_ + (1 - self.mask) * (x * torch.exp(s) + t)

        return x, s.sum(dim=1)

    def inverse(self, x):
        x_ = x * self.mask
        s = torch.tanh(self.s_layers(x_)) * (1 - self.mask)
        t = self.t_layers(x_) * (1 - self.mask)

        x = (1 - self.mask) * (x - t) * torch.exp(-s) + x_

        return x


class RealNVP(nn.Module):
    def __init__(self, n_layers, in_dim, hidden_dim):
        super().__init__()
        masks = torch.tensor([[0, 1], [1, 0]])

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(CouplingLayer(in_dim, hidden_dim, masks[i % 2]))

    def forward(self, x):
        sum_ldj = 0
        for l in self.layers:
            x, ldj = l(x)
            sum_ldj = sum_ldj + ldj

        return x, sum_ldj

    def inverse(self, z):
        for l in reversed(self.layers):
            z = l.inverse(z)

        return z
