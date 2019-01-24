import abc
import copy
import random
from collections import deque

import fret
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence


class _PolicyNet(nn.Module, abc.ABC):
    """Network that generates action (recommended question) at each step"""
    def __init__(self, state_size, action_size):
        super(_PolicyNet, self).__init__()
        self.state_size = state_size
        self.action_size = action_size


@fret.configurable
class SimpleNet(_PolicyNet):
    def __init__(self, hidden_size=200, n_layers=1, **cfg):
        super(SimpleNet, self).__init__(**cfg)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.transform_net = nn.Sequential(
            nn.Linear(self.state_size + self.action_size,
                      hidden_size),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(True))
              for _ in range(n_layers - 1)],
            nn.Linear(hidden_size, 1)
        )

    def forward(self, s, a):
        if isinstance(s, PackedSequence):
            s, _ = pad_packed_sequence(s)
        else:
            s = s.unsqueeze(1)
        h = s.sum(dim=0)  # (_, hs)
        x = torch.cat([h.expand(a.size(0), h.size(1)), a], dim=1)
        return self.transform_net(x)


@fret.configurable
class LSTMNet(SimpleNet):
    def __init__(self, **cfg):
        super(LSTMNet, self).__init__(**cfg)
        n_layers = self.n_layers
        hidden_size = self.hidden_size

        self.initial_h = nn.Parameter(torch.zeros(n_layers * hidden_size))
        self.initial_c = nn.Parameter(torch.zeros(n_layers * hidden_size))
        self.seq_net = nn.LSTM(self.state_size, hidden_size, n_layers)
        self.transform_net = nn.Sequential(
            nn.Linear(hidden_size + self.action_size,
                      hidden_size),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(True))
              for _ in range(n_layers - 1)],
            nn.Linear(hidden_size, 1)
        )

    def forward(self, s, a):
        h = self.initial_h.view(self.n_layers, 1, self.hidden_size)
        c = self.initial_c.view(self.n_layers, 1, self.hidden_size)
        if isinstance(s, PackedSequence):
            bs = s.batch_sizes[0]
            h = h.expand(self.n_layers, bs, self.hidden_size)
            c = c.expand(self.n_layers, bs, self.hidden_size)
        else:
            s = s.unsqueeze(1)
        _, (h, _) = self.seq_net(s, (h, c))
        h = h[-1]  # last layer hidden, (_, hs)
        x = torch.cat([h.expand(a.size(0), h.size(1)), a], dim=1)
        return self.transform_net(x)


@fret.configurable
class DQN:
    submodules = ['policy']

    def __init__(self, policy, questions,
                 gamma=(0.9, 'reward discount rate'),
                 learning_rate=(1e-3, 'learning rate'),
                 target_replace_every=100):
        self.learn_step_counter = 0
        self.gamma = gamma
        self.target_replace_every = target_replace_every

        # get embedding for each question
        state_size = self.state_size = questions.n_knowledge + 2
        action_size = self.action_size = questions.n_knowledge + 1
        self.embs = []
        for q in questions:
            i = np.concatenate([q['knowledge'].reshape(1, -1),
                               np.array([q['difficulty']]).reshape(1, 1)],
                               axis=1)
            self.embs.append(torch.tensor(i).float())
        self.embs = torch.cat(self.embs, dim=0)

        # build DQN network
        self.current_net: _PolicyNet = policy(state_size=state_size,
                                              action_size=action_size)
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
        actions = self.embs
        ind = torch.arange(actions.size(0))
        if q_mask is not None:
            actions = actions.masked_select(q_mask.unsqueeze(1)) \
                .view(-1, self.action_size)
            ind = ind.masked_select(q_mask)
        values = self.current_net(state, actions)
        return ind[values.argmax()].item()

    def train_on_batch(self, batch):
        s, a, s_, r, done, mask = batch
        done = done.float()

        if self.learn_step_counter % self.target_replace_every == 0:
            self.target_net.load_state_dict(self.current_net.state_dict())
        self.learn_step_counter += 1

        q_current = self.current_net(s, self.embs[a])  # Q(s,a), (_, 1)

        a_next = []
        s_padded, lens = pad_packed_sequence(s_)
        for i in range(s_padded.size(1)):  # batch
            a_next.append(self.select_action(s_padded[:lens[i], i, :],
                                             mask[i]))
        a_next = torch.tensor(a_next).long()  # (_, 1)

        q_next = self.target_net(s_, self.embs[a_next])  # target Q, (_, 1)

        q_target = r + self.gamma * q_next * (1 - done)

        loss = self.loss_func(q_current, q_target.detach())

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
