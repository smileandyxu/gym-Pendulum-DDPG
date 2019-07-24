import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import numpy as np
import gym
import os
import config
import utils

class Actor(nn.Module): # actor holds policy network, predicting which action to take
    def __init__(self, states_dim=config.states_dim, action_dim=config.action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(states_dim, config.hidden_dim_pi1)
        self.fc2 = nn.Linear(config.hidden_dim_pi1, config.hidden_dim_pi2)
        self.fc3 = nn.Linear(config.hidden_dim_pi2, config.action_dim)
        utils.init_nns([self.fc1, self.fc2, self.fc3])
    
    def forward(self, s): # predict action
        tmp = self.fc1(s)
        tmp = F.relu(self.fc2(tmp))
        tmp = torch.tanh(self.fc3(tmp))
        return (tmp * 2) # action in 'Pendulum-v0' ranges from -2 to 2
    
    def choose_action(self, s):
        act = self.forward(s)
        return act.detach() # action itself should be seen as a independent variable when calculating gradient


class Critic(nn.Module): # critic holds Q-value network, predicting the value of state-action pair
    def __init__(self, states_dim=config.states_dim, action_dim=config.action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(states_dim, config.hidden_dim_v1)
        self.fc2 = nn.Linear(config.hidden_dim_v1 + action_dim, config.hidden_dim_v2)
        self.fc3 = nn.Linear(config.hidden_dim_v2, 1)
        utils.init_nns([self.fc1, self.fc2, self.fc3])
    
    def forward(self, s, a):
        tmp = F.relu(self.fc1(s))
        tmp = torch.cat((tmp, a), dim=1) # add action into state-action pair
        tmp = F.relu(self.fc2(tmp))
        tmp = self.fc3(tmp) # critic gives a particular value
        return tmp

