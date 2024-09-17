import numpy as np
import torch

def cal_gauss_kl(mu1, sigma1, mu2, sigma2):
    return 0.5 * torch.log(sigma2) - 0.5 * torch.log(sigma1) + 0.5 * ((sigma1 + torch.pow(mu2 - mu1, 2)) / sigma2) - 0.5
