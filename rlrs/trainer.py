from collections import namedtuple

import fret
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from .environment import SPEnv
from .agent import DQN
from .util import critical

TARGET_REPLACE_ITER = 100
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


@fret.configurable
class ValueBasedTrainer:
    def __init__(self, env: SPEnv, agent: DQN,
                 memory_capacity=(500, 'replay memory size')):
        self.env = env
        self.agent = agent
        self.replay_memory = agent.make_replay_memory(memory_capacity)
        self._inputs = []
        self._inputs_len = []

    def train(self, args):
        logger = self.ws.logger('ValueBasedTrainer.train')
        logger.debug('func: <trainer.train>, args: %s', args)

        if args.resume:
            state = self.load_training_state()
        else:
            self.agent.init_training()

        # RL agent training process here
        # TODO: logging, saving checkpoints
        for i_episode in range(args.n_episodes):
            self.env.reset()
            state = self.init_state()

            ep_sum_reward = 0

            try:
                for _ in critical():
                    # select action
                    action = self.agent.select_action(state)

                    # take action in env
                    ob, reward, done, info = self.env.step(action)

                    state_ = self.make_state(action, ob, done)

                    ep_sum_reward += reward

                    # save records
                    self.replay_memory.push(
                        Transition(state, action, state_, reward))

                    # update parameters in agent
                    # sample from replay memory
                    if len(self.replay_memory) < args.batch_size:
                        continue
                    samples = self.replay_memory.sample(args.batch_size)
                    batch = self.make_batch(samples)

                    # train on batch
                    if self.replay_memory.counter > self.replay_memory.capacity:
                        self.agent.train_on_batch(batch)
                        if done:
                            logger.info('Ep: ', i_episode, '| Ep_r: ', round(ep_sum_reward, 3))

                    if done:
                        break
                    state = state_
            except KeyboardInterrupt:
                self.save_training_state({'replay': self.replay_memory,
                                          'agent': self.agent.state_dict()})

    def load_training_state(self):
        cp_path = self.ws.checkpoint_path / 'training_state.pt'
        if cp_path.exists():
            return torch.load(str(cp_path), map_location=lambda s, loc: s)
        else:
            return {}

    def save_training_state(self, state):
        cp_path = self.ws.checkpoint_path / 'training_state.pt'
        torch.save(state, str(cp_path))

    def make_state(self, action, ob, done):
        if done:
            return None
        q = self.env.questions[action]
        '''
        i = torch.cat([(q['knowledge'] * q['difficulty']).view(1, -1),
                       torch.tensor(ob).view(1, 1)], dim=1)
        '''
        kd = (q['knowledge'] * q['difficulty']).view(1, -1)
        i = torch.cat([kd * (ob >= 0.5).type_as(kd).expand_as(kd),
                       kd * (ob < 0.5).type_as(kd).expand_as(kd)])
        self._inputs.append(i)
        self._inputs_len.append(len(self._inputs))
        return torch.cat(self._inputs, dim=0).unsqueeze(1)

    def init_state(self):
        # self._inputs = [torch.zeros(1, self.env.n_knowledge + 1)]
        self._inputs = [torch.zeros(1, self.env.n_knowledge * 2)]
        self._inputs_len.append(len(self._inputs))
        return self._inputs[-1].unsqueeze(1)

    def make_batch(self, samples):
        # TODO: make batched sequential inputs for agent network
        batch_states = [i.state for i in samples]
        batch_actions = [i.action for i in samples]
        batch_states_ = [i.next_state for i in samples]
        batch_rewards = [i.reward for i in samples]

        # TODO: about mask
        return batch_states, batch_actions, batch_states_, batch_rewards, []

        '''
        return [[..., ...],
                ...,
                [..., ...],
                ...,
                ...]  # mask is for setting Q(ending states) to 0
        '''
