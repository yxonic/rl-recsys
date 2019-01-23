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
            x, _ = pad_packed_sequence(x)
        else:
            x = x.unsqueeze(1)
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
        self.emb = nn.Linear(self.state_size, hidden_size)
        self.act = nn.Tanh()
        self.initial_h = nn.Parameter(torch.zeros(n_layers * hidden_size))
        self.initial_c = nn.Parameter(torch.zeros(n_layers * hidden_size))
        self.seq_net = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.action_heads = [nn.Linear(self.hidden_size, self.n_actions)
                             for _ in range(self.n_heads)]
        if self.with_state_value:
            self.value_head = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        h = self.initial_h.view(self.n_layers, 1, self.hidden_size)
        c = self.initial_c.view(self.n_layers, 1, self.hidden_size)
        if isinstance(x, PackedSequence):
            x = PackedSequence(self.act(self.emb(x.data)), x.batch_sizes)
            bs = x.batch_sizes[0]
            h = h.expand(self.n_layers, bs, self.hidden_size)
            c = c.expand(self.n_layers, bs, self.hidden_size)
        else:
            x = self.act(self.emb(x)).unsqueeze(1)
        _, (h, _) = self.seq_net(x, (h, c))
        h = h[-1]  # last layer hidden
        action_values = [net(h) for net in self.action_heads]
        if self.with_state_value:
            return action_values, self.value_head(h)
        else:
            return action_values


@fret.configurable
class DQN:
    submodules = ['policy']

    def __init__(self, policy, state_size, n_actions,
                 gamma=(0.9, 'reward discount rate'),
                 learning_rate=(1e-3, 'learning rate'),
                 target_replace_every=100):
        self.learn_step_counter = 0
        self.gamma = gamma
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
        action_values = self.current_net(state)[0].view(-1)
        if q_mask is not None:
            action_values += q_mask
        return action_values.argmax().item()

    def train_on_batch(self, batch):
        s, a, s_, r, done, mask = batch
        done = done.float()

        if self.learn_step_counter % self.target_replace_every == 0:
            self.target_net.load_state_dict(self.current_net.state_dict())
        self.learn_step_counter += 1

        q_current = self.current_net(s)[0].gather(1, a).view(-1)  # Q(s,a)

        a_next = self.current_net(s_)[0] + mask
        a_next = a_next.argmax(1).unsqueeze(-1).detach()  # argmax(Q(s',a))

        q_next = self.target_net(s_)[0].squeeze(1).detach()  # target Q

        q_target = r + \
                   self.gamma * q_next.gather(1, a_next).view(-1) * \
                   (1 - done)  # if done, this part becomes zero

        loss = self.loss_func(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def load_model(self, tag):
        cp_path = self.ws.checkpoint_path / ('%s.%s.pt' % (
            self.current_net.__class__.__name__, str(tag)))
        state = torch.load(str(cp_path), map_location=lambda s, loc: s)
        self.current_net.load_state_dict(state)
        self.target_net.load_state_dict(state)

    def save_model(self, tag):
        cp_path = self.ws.checkpoint_path / ('%s.%s.pt' % (
            self.current_net.__class__.__name__, str(tag)))
        torch.save(self.current_net.state_dict(), cp_path)

    def state_dict(self):
        return {
            'current': self.current_net.state_dict(),
            'target': self.target_net.state_dict(),
            'optim': self.optimizer.state_dict()
        }

    def load_state_dict(self, state):
        self.current_net.load_state_dict(state['current'])
        self.target_net.load_state_dict(state['target'])
        self.optimizer.load_state_dict(state['optim'])


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
