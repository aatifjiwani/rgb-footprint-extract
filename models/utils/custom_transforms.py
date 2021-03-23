import torch
import torch.nn.functional as F

def tensor_resize(size):
    def fn(tensor):
        return F.interpolate(tensor, size=size, mode="nearest")

    return fn
