import random
from collections import deque

import fret
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .sp_models import QuesNet
from .dataprep import Questions

# to be update about N_actions and N_states
N_STATES = 100
N_ACTIONS = 50
TARGET_REPLACE_ITER = 100   # target update frequency


@fret.configurable
class DQN:
    def __init__(self,
                 memory_capacity=(2000, 'max saved memory states'),
                 learning_rate=(0.001, 'learning rate'),
                 greedy_epsilon=(0.0, 'greedy policy'),
                 gama=(0.9, 'reward discount rate'),
                 double_q=False,
                 dataset=fret.ref('env.dataset')
                 ):
        self.learn_step_counter = 0
        self.learning_rate = learning_rate
        self.greedy_epsilon = greedy_epsilon
        self.gama = gama

        self.n_knowledges = fret.app['datasets'][dataset]['n_knowledges']
        self.n_actions = fret.app['datasets'][dataset]['n_actions']

        # self.memory_counter = 0
        # self.memory_capacity = memory_capacity
        # self.memory = np.zeros((memory_capacity, 4)) # store state, action, reward, state_

        # build DQN network
        self.current_net = SimpleNet(self.n_actions, 50, self.n_actions)
        self.target_net = SimpleNet(self.n_actions, 50, self.n_actions)
        self.optimizer = torch.optim.Adam(self.current_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    @staticmethod
    def make_replay_memory(size):
        return ReplayMemory(size)

    def select_action(self, state):
        if np.random.uniform() < self.greedy_epsilon:
            action_values = self.current_net(state)
            action = torch.max(action_values, 1)[1].numpy()
        else:
            action = np.random.randint(0, self.n_actions)

        # self._rec_history.append(action)
        return action

    '''
    def store_transition(self, state, action, reward, state_):
        transition = np.hstack((state, action, reward, state_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index:] = transition
        self.memory_counter += 1
    '''

    def init_training(self):
        pass

    # to be updated
    def train_on_batch(self, batch):
        ws = self.ws
        s, a, s_, r, mask = batch

        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).astype(int)
        s_ = torch.FloatTensor(s_)
        r = torch.FloatTensor(r)

        q_current = self.current_net(s).gather(1, a)
        q_next = self.target_net(a, s_).detach()
        q_target = r + self.gama * q_next.max(1)[0]

        loss = self.loss_func(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, tag):
        pass

    def load_model(self, tag):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self):
        pass


# RL net which generates action(recommended question) at each step
# to be updated: input question or question_emb ?
class SimpleNet(nn.Module):
    def __init__(self,
                 n_knowledges=50,
                 ques_h_size=50,
                 n_actions=100):
        super(SimpleNet, self).__init__()

        self.n_actions = n_actions
        self.state_size = n_knowledges
        self.ques_h_size = ques_h_size

        self.input_net = nn.Linear(self.state_size, 200)
        self.out_net = nn.Linear(200, self.n_actions)
        # self.input_net.weight.data.normal_(0, 0.1)
        # self.out_net.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # input current state: (question, score)
        # generate next action values (question_)
        # one hot feature from (question, score)
        # feature_x = self.generate_state_feature()
        # feature_x = torch.unsqueeze(torch.FloatTensor(feature_x), 0)

        x = F.relu(self.input_net(x))
        action_values = self.out_net(x)
        return action_values

    def generate_state_feature(self, action, score):
        feature_hot = np.zeros(self.n_questions)
        feature_hot[action] = score
        return feature_hot


# simple RNN net
class GRUNet(nn.Module):
    def __init__(self,
                 state_feature_size=(100, 'input state feature size'),
                 seq_h_size=(200, 'seq hidden size'),
                 action_size=(100, 'number of questions'),
                 n_layers=1):
        self.state_feature_size = state_feature_size
        self.seq_h_size = seq_h_size
        self.action_size = action_size
        self.n_layers = n_layers

        self.initial_h = nn.Parameter(torch.zeros(n_layers *
                                                  seq_h_size))
        self.seq_net = nn.GRU(self.state_feature_size, self.seq_h_size, self.n_layers)
        self.action_net = nn.Linear(self.seq_h_size, self.action_size)

    def forward(self, x, hidden):
        if hidden is None:
            h = self.initial_h.view(self.n_layers, 1, self.seq_hidden_size)
        else:
            h = hidden

        # feature_x = self.generate_state_feature(action, score)
        # feature_x = torch.FloatTensor(feature_x).view(1, 1, -1)
        _, h = self.seq_net(x, h)
        action_values = self.action_net(h)
        return action_values, h

    def generate_state_feature(self, action, score):
        feature_hot = np.zeros(self.n_questions)
        feature_hot[action] = score
        return feature_hot


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        # self.position = 0
        self.counter = 0

    def push(self, trans):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        '''
        self.memory[self.position] = trans
        self.position = (self.position + 1) % self.capacity
        '''
        position = self.counter % self.capacity
        self.memory[position] = trans
        self.counter += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
