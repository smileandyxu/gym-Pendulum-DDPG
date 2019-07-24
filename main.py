import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import os
import config
import utils
from DDPG import DDPG

if __name__ == "__main__":
    model = DDPG()
    model.run()
    model.close()
    model.plot()
