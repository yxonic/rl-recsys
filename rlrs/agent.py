import abc
import copy
import random
from collections import deque

import fret
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence


@fret.configurable
class _PolicyNet(nn.Module, abc.ABC):
    """Network that generates action (recommended question) at each step"""

    def __init__(self, _state_size=20, _n_actions=10, _n_heads=1,
                 _with_state_value=False):
        super(_PolicyNet, self).__init__()
        self.state_size = _state_size
        self.n_actions = _n_actions
        self.n_heads = _n_heads
        self.with_state_value = _with_state_value


@fret.configurable
class SimpleNet(_PolicyNet):
    def __init__(self, hidden_size=200, **cfg):
        super(SimpleNet, self).__init__(**cfg)
        self.hidden_size = hidden_size
        self.input_net = nn.Linear(self.state_size, self.hidden_size)
        self.action_heads = [nn.Linear(self.hidden_size, self.n_actions)
                             for _ in range(self.n_heads)]
        if self.with_state_value:
            self.value_head = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        if isinstance(x, PackedSequence):
            x, lens = pad_packed_sequence(x)
        x = x.sum(dim=0)
        x = F.relu(self.input_net(x))
        action_values = [net(x) for net in self.action_heads]
        if self.with_state_value:
            return action_values, self.value_head(x)
        else:
            return action_values


@fret.configurable
class GRUNet(_PolicyNet):
    def __init__(self, n_layers=1, **cfg):
        super(GRUNet, self).__init__(**cfg)
        self.n_layers = n_layers

        self.initial_h = nn.Parameter(torch.zeros(n_layers *
                                                  self.hidden_size))
        self.seq_net = nn.GRU(self.state_feature_size, self.seq_h_size, self.n_layers)
        self.action_net = nn.Linear(self.seq_h_size, self.action_size)

    def forward(self, x, hidden):
        if hidden is None:
            h = self.initial_h.view(self.n_layers, 1, self.seq_hidden_size)
        else:
            h = hidden

        # pack_x = nn.utils.pack_padded_sequence(x)
        _, h = self.seq_net(x, h)
        action_values = self.action_net(h)
        return action_values, h


@fret.configurable
class DQN:
    def __init__(self, policy, _state_size, _n_actions,
                 greedy_epsilon=(0.9, 'greedy policy'),
                 gama=(0.9, 'reward discount rate'),
                 learning_rate=(0.1, 'learning rate'),
                 target_replace_every=50,
                 double_q=True):
        self.learn_step_counter = 0
        self.greedy_epsilon = greedy_epsilon
        self.gama = gama
        self.target_replace_every = target_replace_every
        # build DQN network
        self.current_net: _PolicyNet = policy(_state_size=_state_size,
                                              _n_actions=_n_actions,
                                              _n_heads=1,
                                              _with_state_value=False)
        self.target_net = copy.deepcopy(self.current_net)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.current_net.parameters(),
                                          lr=learning_rate)

    @staticmethod
    def make_replay_memory(size):
        return ReplayMemory(size)

    def select_action(self, state):
        action_values = self.current_net(state)[0]
        return action_values.argmax(1).item()

    def init_training(self):
        pass

    def train_on_batch(self, batch):
        s, a, s_, r, mask = batch

        if self.learn_step_counter % self.target_replace_every == 0:
            self.target_net.load_state_dict(self.current_net.state_dict())
        self.learn_step_counter += 1

        q_current = self.current_net(s)[0].squeeze(1).gather(1, a).view(-1)
        q_next = self.target_net(s_)[0].squeeze(1).gather(1, a).detach()
        q_target = r + self.gama * q_next.max(1)[0]

        loss = self.loss_func(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_model(self, tag):
        pass

    def load_model(self, tag):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state):
        pass


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.counter = 0

    def push(self, trans):
        self.memory.append(trans)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
