import torch
import torch.nn as nn
import numpy as np
import config

def init_nns(layers): # give initial values to NNs
    for layer in layers:
        layer.weight.data.normal_(0, 0.1)

