import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import numpy as np
import gym
import os
import config
import utils
from brain import Actor, Critic
import matplotlib.pyplot as plt

class DDPG: # DDPG algorithm implementation
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate_a)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate_c)
        self.target_actor = Actor() # here, target networks are Q' and μ' in DDPG's original paper, which are delay networks, aiming to make the model more stable
        self.target_critic = Critic()
        self.target_actor.load_state_dict(self.actor.state_dict()) # let initial parameters be the same
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.memory = [[], [], [], [], []]
        self.env = gym.envs.make(config.game_name).unwrapped
        self.env.seed(1)
        self.rewards = []
        self.steps = []
        self.loss_td = nn.MSELoss()
        self.var = config.initial_var # width of exploration factor, if we don't use it the model might stick in one pattern and doesn't want to try other possibilities

    def close(self):
        self.env.close()
        
    def learn(self):        
        self.var = self.var * config.decay # renew exploration width, it should be small because we want the final performance be stable

        self.update_weights() # update weights using formula mentioned in paper, with given tau in config.py

        S, A, R, S_, D = self.get_sample() # get training data from memory to update the model
        S = torch.FloatTensor(S).reshape(-1, config.states_dim) # convert to PyTorch Tensors
        A = torch.FloatTensor(A).reshape(-1, config.action_dim)
        R = torch.FloatTensor(R).reshape(-1, 1)
        S_ = torch.FloatTensor(S_).reshape(-1, config.states_dim)
        D = torch.FloatTensor(D)

        A_pred = self.actor(S) # optimize actor network
        Q = self.critic(S, A_pred)
        loss_actor = -torch.mean(Q)
        self.actor_opt.zero_grad()
        loss_actor.backward()
        self.actor_opt.step()

        Q_prime = self.target_critic(S_, self.target_actor(S_)) # optimize critic network
        Q_target = R + config.gamma * Q_prime * (1 - D)
        Q_pred = self.critic(S, A)
        loss_critic = self.loss_td(Q_target, Q_pred)
        self.critic_opt.zero_grad()
        loss_critic.backward()
        self.critic_opt.step()

        #print('<<<', loss_actor.data.numpy(), ' | ', loss_critic.data.numpy(), '>>>')

    def run(self): # Training process
        tot_s = 0 # total steps during training
        tot_r = 0 # total reward during training
        for ep in range(config.max_episode): # training episode
            s = self.env.reset()
            ep_r = 0 # reward during this episode
            for t in range(config.max_step):
                if ep > 35:
                    self.env.render() # render the environment, can be open
                a = np.clip(np.random.normal(self.act(s), self.var), -2, 2) # choose an action at this state, here self.var is used to add noise, aiming to let the model do some exploration job
                s_, r, done, _ = self.env.step(a) # get response from envrionment
                self.push_memory(s, a, r / 10, s_, done) # add this step into memory
                ep_r += r # update information
                tot_s += 1
                s = s_

                if len(self.memory[0]) == config.max_memsize: # after reach the memsize, do the learning work
                    self.learn()
                if done: # game over
                    break

            tot_r += ep_r # update information
            self.rewards.append(tot_r / (ep + 1))
            self.steps.append(tot_s)
            print(f'episode {ep + 1} | reward: {ep_r}')
    
    def push_memory(self, s, a, r, s_, done): # add response to memory
        self.memory[0].append(s)
        self.memory[1].append(a)
        self.memory[2].append(r)
        self.memory[3].append(s_)
        self.memory[4].append(done)
        pos = np.random.randint(low=0, high=config.max_memsize + 1)
        if len(self.memory[0]) > config.max_memsize:
            for i in range(5):
                self.memory[i].pop(pos)

    def get_sample(self): # get training data
        idx = np.random.choice(config.max_memsize, size=config.batch_size)
        sample = [[self.memory[i][j] for j in idx] for i in range(5)]
        # print(sample)
        return sample[0], sample[1], sample[2], sample[3], sample[4]

    def act(self, s):
        return self.actor.choose_action(torch.FloatTensor(s)).data.numpy()

    def update_weights(self): # update main networks' parameters
        for w, dw in zip(self.target_actor.parameters(), self.actor.parameters()):
            w.data.copy_(w.data * (1 - config.tau) + dw * config.tau)
        for w, dw in zip(self.target_critic.parameters(), self.critic.parameters()):
            w.data.copy_(w.data * (1 - config.tau) + dw * config.tau)
        
        
    def plot(self):
        plt.figure()
        plt.plot(self.steps, self.rewards, 'b-')
        plt.xlabel('time step')
        plt.ylabel('average reward (per episode)')
        plt.show()



    