import torch
import random


class SpatialMixup(object):
    def __init__(self, alpha=0.2, trace=True, version=2):
        self.alpha = alpha
        self.trace = trace
        self.version = version

    def mixup_data(self, x):
        """
        return mixed inputs. pairs of targets
        """
        b, t, c, h, w = x.size()
        loss_prob = random.random() * self.alpha
        if self.trace:
            mixed_x = x
        else:
            mixed_x = torch.zeros_like(x)
        for i in range(b):
            tmp = (i+1) % b
            img_index = random.randint(0, t-1)
            for j in range(t):
                mixed_x[i, j, :, :, :] = (1-loss_prob) * x[i, j, :, :, :] + loss_prob * x[tmp, img_index, :, :, :]
        return mixed_x
