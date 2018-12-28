import datetime
from collections import namedtuple

import fret
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from .environment import SPEnv
from .agent import DQN
from .util import critical, make_batch

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

    def train(self, args):
        logger = self.ws.logger('ValueBasedTrainer.train')
        logger.debug('func: <trainer.train>, args: %s', args)

        state = self.load_training_state()
        if args.resume and state:
            current_run = state['run']
            start_episode = state['i_episode']
            i_batch = state['i_batch']
            self.replay_memory = state['replay']
            self.agent.load_state_dict(state['agent'])
        else:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_run = str(self.ws.log_path / ('run-%s/' % now))
            start_episode = 0
            i_batch = 0
            self.agent.init_training()

        writer = SummaryWriter(current_run)

        # RL agent training process here
        for i_episode in range(start_episode, args.n_episodes):
            self.env.reset()
            state = self.init_state()

            ep_sum_reward = 0

            try:
                for _ in critical():
                    i_batch += 1

                    # select action
                    action = self.agent.select_action(state)

                    # take action in env
                    ob, reward, done, info = self.env.step(action)

                    state_ = self.make_state(action, ob, done)

                    ep_sum_reward += reward

                    # save records
                    self.replay_memory.push(
                        Transition(state, action, state_, reward))
                    state = state_

                    if done:
                        writer.add_scalar('ValueBasedTrainer.train/return',
                                          ep_sum_reward, i_episode)
                        break

                    # update parameters in agent
                    # sample from replay memory
                    if len(self.replay_memory) < args.batch_size:
                        continue

                    samples = self.replay_memory.sample(args.batch_size)
                    batch = self.make_batch(samples)

                    # train on batch
                    loss = self.agent.train_on_batch(batch)
                    writer.add_scalar('ValueBasedTrainer.train/loss',
                                      loss, i_batch)

            except KeyboardInterrupt:
                self.save_training_state({'run': current_run,
                                          'i_episode': i_episode,
                                          'i_batch': i_batch,
                                          'replay': self.replay_memory,
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
        kd = (q['knowledge'] * q['difficulty']).reshape(1, -1)
        i = np.concatenate([kd * (ob >= 0.5), kd * (ob < 0.5)], axis=1)
        self._inputs.append(i)
        return np.expand_dims(np.concatenate(self._inputs, axis=0), 1)

    def init_state(self):
        self._inputs = [np.zeros((1, self.env.n_knowledge * 2))]
        return np.expand_dims(self._inputs[-1], 1)

    def make_batch(self, samples):
        # TODO: make batched sequential inputs for agent network
        states = [torch.tensor(i.state).float() for i in samples]
        actions = [torch.tensor(i.action).long().view(1, 1) for i in samples]
        states_ = [torch.tensor(self.init_state()).float()
                   if i.next_state is None
                   else torch.tensor(i.next_state).float()
                   for i in samples]
        rewards = [torch.tensor(i.reward).view(1, 1).float() for i in samples]
        masks = [torch.tensor(0).view(1, 1) if s.next_state is None
                 else torch.tensor(1).view(1, 1)
                 for s in samples]
        states, actions, states_, rewards, masks = \
            make_batch(states, actions, states_, rewards, masks, seq=[0, 2])

        return states, actions, states_, rewards, masks
