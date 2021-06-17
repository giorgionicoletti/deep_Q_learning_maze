import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import collections

Transition = collections.namedtuple('Experience',
                                    field_names=['state', 'action',
                                                 'next_state', 'reward',
                                                 'is_game_on'])


class Agent:
    def __init__(self, maze, memory_buffer, use_softmax = True):
        self.env = maze
        self.buffer = memory_buffer # this is actually a reference
        self.num_act = 4
        self.use_softmax = use_softmax
        self.total_reward = 0
        self.min_reward = -self.env.maze.size
        self.isgameon = True

        
    def make_a_move(self, net, epsilon, device = 'cuda'):
        action = self.select_action(net, epsilon, device)
        current_state = self.env.state()
        next_state, reward, self.isgameon = self.env.state_update(action)
        self.total_reward += reward
        
        if self.total_reward < self.min_reward:
            self.isgameon = False
        if not self.isgameon:
            self.total_reward = 0
        
        transition = Transition(current_state, action,
                                next_state, reward,
                                self.isgameon)
        
        self.buffer.push(transition)
            
        
    def select_action(self, net, epsilon, device = 'cuda'):
        state = torch.Tensor(self.env.state()).to(device).view(1,-1)
        qvalues = net(state).cpu().detach().numpy().squeeze()

        # softmax sampling of the qvalues
        if self.use_softmax:
            p = sp.softmax(qvalues/epsilon).squeeze()
            p /= np.sum(p)
            action = np.random.choice(self.num_act, p = p)
            
        # else choose the best action with probability 1-epsilon
        # and with probability epsilon choose at random
        else:
            if np.random.random() < epsilon:
                action = np.random.randint(self.num_act, size=1)[0]
            else:                
                action = np.argmax(qvalues, axis=0)
                action = int(action)
        
        return action
    
    
    def plot_policy_map(self, net, filename, offset):
        net.eval()
        with torch.no_grad():
            fig, ax = plt.subplots()
            ax.imshow(self.env.maze, 'Greys')

            for free_cell in self.env.allowed_states:
                self.env.current_position = np.asarray(free_cell)
                qvalues = net(torch.Tensor(self.env.state()).view(1,-1).to('cuda'))
                action = int(torch.argmax(qvalues).detach().cpu().numpy())
                policy = self.env.directions[action]

                ax.text(free_cell[1]-offset[0], free_cell[0]-offset[1], policy)
            ax = plt.gca();

            plt.xticks([], [])
            plt.yticks([], [])

            ax.plot(self.env.goal[1], self.env.goal[0],
                    'bs', markersize = 4)
            plt.savefig(filename, dpi = 300, bbox_inches = 'tight')
            plt.show()
