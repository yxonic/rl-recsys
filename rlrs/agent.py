import abc
import copy
import itertools
import math
import os
import random
from collections import deque

import fret
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence

from .sp_models import EERNN


@fret.configurable
class Agent:
    submodules = []

    def __init__(self, questions):
        self.questions = questions

    def select_action(self, state, action_mask=None):
        values, ind = self.get_action_values(state, action_mask)
        v, i = values.max(), values.argmax()
        return int(ind[i]), v

    def get_action_values(self, state, action_mask):
        raise NotImplementedError

    def step(self, state, action, ob):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


@fret.configurable
class RandomAgent(Agent):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def get_action_values(self, state, action_mask):
        vs = []
        ind = []
        for i, m in enumerate(action_mask):
            if not m:
                continue
            vs.append(random.random())
            ind.append(i)
        return np.asarray(vs), np.asarray(ind)

    def step(self, state, action, ob):
        pass

    def reset(self):
        pass


@fret.configurable
class GreedySPAgent(Agent):
    submodules = ['sp_model']

    def __init__(self, sp_model, **cfg):
        super().__init__(**cfg)
        self.sp_model = sp_model(_dataset=self.questions.dataset,
                                 _questions=self.questions)

    def get_action_values(self, state, action_mask):
        vs = []
        ind = []
        for i, m in enumerate(action_mask):
            if not m:
                continue
            q = self.questions[i]
            s, _ = self.sp_model(q, None, state)
            s = s.flatten()[0].item()
            vs.append(1 - s)
            ind.append(i)
        return np.asarray(vs), np.asarray(ind)

    def step(self, state, action, ob):
        q = self.questions[action]
        s = torch.tensor(ob).float()
        with torch.no_grad():
            _, state = self.sp_model(q, s, state)
        return state

    def reset(self):
        return None

    def load_model(self, tag):
        cp_path = self.ws.checkpoint_path / ('%s.%s.pt' % (
            self.sp_model.__class__.__name__, str(tag)))
        self.sp_model.load_state_dict(torch.load(
            str(cp_path), map_location=lambda s, loc: s))


@fret.configurable
class DQN(Agent):
    def __init__(self, policy, sp_model,
                 gamma=(0.9, 'reward discount rate'),
                 learning_rate=(1e-3, 'learning rate'),
                 target_replace_every=100, **cfg):
        super(DQN, self).__init__(**cfg)
        self.learn_step_counter = 0
        self.gamma = gamma
        self.target_replace_every = target_replace_every

        # get embedding for each question
        questions = self.questions
        state_size = self.state_size = questions.n_knowledge + 52
        action_size = self.action_size = questions.n_knowledge + 51
        self.questions = questions

        sp_model: EERNN = sp_model(_dataset=questions.dataset,
                                   _questions=questions)
        cp_path = 'ws/best/%s.%s.pt' % (questions.dataset,
                                        sp_model.__class__.__name__)
        sp_model.load_state_dict(torch.load(
            str(cp_path), map_location=lambda s, loc: s))
        sp_model.eval()
        quesnet = sp_model.question_net
        for param in quesnet.parameters():
            param.requires_grad_(False)

        self.embs = []
        q_text_embs = sp_model.ques_vs
        for i, q in enumerate(questions):
            i = torch.cat([q_text_embs[i].view(1, -1),
                           torch.tensor(q['knowledge']).float().view(1, -1),
                           torch.tensor([q['difficulty']]).float().view(1, 1)],
                          dim=1)
            self.embs.append(i)
        self.embs = torch.cat(self.embs, dim=0)

        self.inputs = []

        # build DQN network
        self.current_net: PolicyNet = policy(state_size=state_size,
                                             action_size=action_size)
        self.target_net = copy.deepcopy(self.current_net)
        for param in self.target_net.parameters():
            param.requires_grad_(False)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.current_net.parameters(),
                                          lr=learning_rate)

    def step(self, state, action, ob):
        i = torch.cat([self.embs[action].view(1, -1),
                       torch.tensor([ob]).float()], dim=1)
        return torch.cat([state, i], dim=0)  # next state

    def reset(self):
        return torch.zeros((1, self.questions.n_knowledge + 52))

    @staticmethod
    def make_replay_memory(size):
        return ReplayMemory(size)

    def get_action_values(self, state, q_mask=None):
        actions = self.embs
        ind = torch.arange(actions.size(0))
        if q_mask is not None:
            if not isinstance(q_mask, torch.Tensor):
                q_mask = torch.tensor(q_mask).byte()
            actions = actions.masked_select(q_mask.unsqueeze(1)) \
                .view(-1, self.action_size)
            ind = ind.masked_select(q_mask)
        values = self.current_net(state, actions).view(-1).detach()
        return values.numpy(), ind.numpy()

    def train_on_batch(self, batch):
        s, a, s_, r, done, mask = batch
        done = done.float()

        if self.learn_step_counter % self.target_replace_every == 0:
            self.target_net.load_state_dict(self.current_net.state_dict())
        self.learn_step_counter += 1

        q_current = self.current_net(s, self.embs[a]).view(-1)  # Q(s,a), (_)

        a_next = []
        s_padded, lens = pad_packed_sequence(s_)
        for i in range(s_padded.size(1)):  # batch
            a_next.append(
                self.select_action(s_padded[:lens[i], i, :], mask[i])[0])
        a_next = torch.tensor(a_next).long()  # (_, 1)

        q_next = self.target_net(s_, self.embs[a_next])  # target Q, (_, 1)
        q_next = q_next.detach().view(-1)

        q_target = r + self.gamma * q_next * (1 - done)

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


class PolicyNet(nn.Module, abc.ABC):
    """Network that generates action (recommended question) at each step"""
    def __init__(self, state_size, action_size):
        super(PolicyNet, self).__init__()
        self.state_size = state_size
        self.action_size = action_size


@fret.configurable
class SimpleNet(PolicyNet):
    def __init__(self, hidden_size=200, n_layers=1, **cfg):
        super(SimpleNet, self).__init__(cfg['state_size'], cfg['action_size'])
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
class DQNSimpleNet(PolicyNet):
    """
    Baseline network: Simple DQN network using several dense layers for state embedding,
    State: one-hot of exercise id plus performance
    """
    def __init__(self, _questions, _knowledges, hidden_size=200, n_layers=1, **cfg):
        super(DQNSimpleNet, self).__init__(cfg['state_size'], cfg['action_size'])
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.questions = _questions
        self.qcnt = len(self.questions.stoi)
        self.knowledges = _knowledges
        self.kcnt = len(self.knowledges)

        self.state_net = nn.Sequential(
            nn.Linear(self.qcnt * 2, 200),
            nn.Linear(200, 100),
            nn.Linear(100, self.state_size)
        )

    def forward(self, q, score):
        q_onehot = torch.zeros(self.qcnt)
        q_onehot[self.questions.stoi[q['id']]] = 1

        x = torch.cat([q_onehot * (score >= 0.5).type_as(q_onehot).expand_as(q_onehot),
                       q_onehot * (score < 0.5).type_as(q_onehot).expand_as(q_onehot)])
        return self.state_net(x)
