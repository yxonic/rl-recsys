import abc
import copy
import random
from collections import deque

import fret
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence


class _PolicyNet(nn.Module, abc.ABC):
    """Network that generates action (recommended question) at each step"""
    def __init__(self, state_size, n_actions, n_heads, with_state_value):
        super(_PolicyNet, self).__init__()
        self.state_size = state_size
        self.n_actions = n_actions
        self.n_heads = n_heads
        self.with_state_value = with_state_value


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
class LSTMNet(_PolicyNet):
    def __init__(self, n_layers=1, hidden_size=50, **cfg):
        super(LSTMNet, self).__init__(**cfg)
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.initial_h = nn.Parameter(torch.zeros(n_layers * hidden_size))
        self.initial_c = nn.Parameter(torch.zeros(n_layers * hidden_size))
        self.seq_net = nn.LSTM(self.state_size, hidden_size, n_layers)
        self.action_net = nn.Linear(hidden_size, self.n_actions)

    def forward(self, x):
        h = self.initial_h.view(self.n_layers, 1, self.hidden_size)
        c = self.initial_c.view(self.n_layers, 1, self.hidden_size)
        if isinstance(x, PackedSequence):
            bs = x.batch_sizes[0]
            h = h.expand(self.n_layers, bs, self.hidden_size)
            c = c.expand(self.n_layers, bs, self.hidden_size)
        else:
            x = x.unsqueeze(1)
        _, (h, _) = self.seq_net(x, (h, c))
        action_values = self.action_net(h)
        return action_values


@fret.configurable
class DQN:
    submodules = ['policy']

    def __init__(self, policy, state_size, n_actions,
                 greedy_epsilon=(0.9, 'greedy policy'),
                 gama=(0.9, 'reward discount rate'),
                 learning_rate=(1e-3, 'learning rate'),
                 target_replace_every=100,
                 double_q=True):
        self.learn_step_counter = 0
        self.greedy_epsilon = greedy_epsilon
        self.gama = gama
        self.target_replace_every = target_replace_every
        # build DQN network
        self.current_net: _PolicyNet = policy(state_size=state_size,
                                              n_actions=n_actions,
                                              n_heads=1,
                                              with_state_value=False)
        self.target_net = copy.deepcopy(self.current_net)
        for param in self.target_net.parameters():
            param.requires_grad_(False)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.current_net.parameters(),
                                          lr=learning_rate)

    @staticmethod
    def make_replay_memory(size):
        return ReplayMemory(size)

    def select_action(self, state, q_mask=None):
        action_values = self.current_net(state)[0].squeeze(1)
        if q_mask is not None:
            action_values += q_mask.unsqueeze(0)
        return action_values.argmax(1).item()

    def init_training(self):
        pass

    def train_on_batch(self, batch, q_mask=None):
        s, a, s_, r, mask = batch

        if self.learn_step_counter % self.target_replace_every == 0:
            self.target_net.load_state_dict(self.current_net.state_dict())
        self.learn_step_counter += 1

        q_current = self.current_net(s)[0].squeeze(1).gather(1, a).view(-1)
        if q_mask is None:
            a_next = self.current_net(s_)[0].squeeze(1).argmax(1) \
                .unsqueeze(1).detach()
        else:
            a_next = self.current_net(s_)[0].squeeze(1) + q_mask.unsqueeze(0)
            a_next = a_next.argmax(1).unsqueeze(-1).detach()

        q_next = self.target_net(s_)[0].squeeze(1).detach()
        q_target = r + self.gama * q_next.gather(1, a_next).view(-1) * mask

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
