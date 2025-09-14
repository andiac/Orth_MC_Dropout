import torch
import torch.nn as nn

class OrthogonalDropout(nn.Module):
    def __init__(self, p_list):
        super(OrthogonalDropout, self).__init__()
        self.p_list = []                      # dropout probability of each x
        self.sum_one_minus_p = 0.
        self.zp_list = []                     # probability of each z (Bernoulli parameter)

        for p in p_list:
            self.add_p(p)

    def add_p(self, p):
        self.p_list.append(p)
        self.sum_one_minus_p += 1. - p

        assert self.sum_one_minus_p <= 1.

        if len(self.p_list) == 1:
            self.zp_list.append(1. - p)
        else:
            self.zp_list.append((1. - p) / (sum(self.p_list[:-1]) - (len(self.p_list) - 2.)))

    def forward(self, x_list):
        assert len(x_list) == len(self.p_list)
        assert len(x_list) > 0

        res = []
        
        if len(self.p_list) > 0:
            z_list = []
            for idx, _ in enumerate(self.p_list):
                z_list.append(torch.bernoulli(torch.ones_like(x_list[0], device=x_list[0].device) * self.zp_list[idx]))
            accumulate_z = torch.ones_like(x_list[0], device=x_list[0].device)
            for idx, _ in enumerate(self.p_list):
                res.append(accumulate_z * z_list[idx] * x_list[idx])
                accumulate_z *= (1. - z_list[idx])

        return res

class NormalDropout(nn.Module):
    def __init__(self, p_list):
        super(NormalDropout, self).__init__()
        self.p_list = p_list          # probability of each z (Bernoulli parameter)

    def forward(self, x_list):
        assert len(x_list) == len(self.p_list)
        assert len(x_list) > 0

        res = []

        for x, p in zip(x_list, self.p_list):
            res.append((1. - torch.bernoulli(torch.ones_like(x, device=x.device) * p)) * x)
        
        return res