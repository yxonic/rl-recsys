import datetime
import random
from collections import namedtuple

import fret
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from .environment import _StuEnv
from .agent import DQN
from .dataprep import load_record
from .util import critical, make_batch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


@fret.configurable
class ValueBasedTrainer:
    def __init__(self, env, agent,
                 memory_capacity=(500, 'replay memory size'),
                 exploration_p=(0.5, ('exploration probability, set to 1 for '
                                      'off policy training'))):
        self.exploration_p = exploration_p
        self.env: _StuEnv = env()
        self.agent: DQN = agent(state_size=self.env.n_knowledge + 2,
                                n_actions=self.env.n_questions)
        self.replay_memory = self.agent.make_replay_memory(memory_capacity)
        self._inputs = []

    def train(self, args):
        logger = self.ws.logger('ValueBasedTrainer.train')
        logger.debug('func: <trainer.train>, args: %s', args)

        rec_file = fret.app['datasets'][self.env.dataset]['record_file']
        records = load_record(rec_file, self.env.questions)
        self.env.set_records(records)

        state = self.load_training_state()
        if not args.restart and state:
            current_run = state['run']
            start_episode = state['i_episode']
            i_batch = state['i_batch']
            self.replay_memory = state['replay']
            self.agent.load_state_dict(state['agent'])
        else:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            current_run = str(self.ws.log_path /
                              ('ValueBasedTrainer.train/run-%s/' % now))
            start_episode = 0
            i_batch = 0
            self.agent.init_training()

        writer = SummaryWriter(current_run)

        # RL agent training process here
        for i_episode in critical(tqdm(range(start_episode, args.n_episodes),
                                       initial=start_episode,
                                       total=args.n_episodes)):
            try:
                self.env.reset()
                state = self.init_state()

                ep_sum_reward = 0

                for _ in critical():
                    i_batch += 1

                    # select action
                    if random.random() <= self.exploration_p:
                        action = self.env.random_action()
                    else:
                        action = self.agent.select_action(
                            torch.tensor(state).float())

                    # take action in env
                    ob, reward, done, info = self.env.step(action)

                    state_ = self.make_state(action, ob)

                    ep_sum_reward += reward

                    # save records
                    self.replay_memory.push(
                        Transition(state, action, state_, reward, done))
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
                raise

        self.save_training_state({'run': current_run,
                                  'i_episode': args.n_episodes,
                                  'i_batch': i_batch,
                                  'replay': self.replay_memory,
                                  'agent': self.agent.state_dict()})

    def load_training_state(self):
        cp_path = self.ws.checkpoint_path / \
            (self.__class__.__name__ + '_state.pt')
        if cp_path.exists():
            return torch.load(str(cp_path), map_location=lambda s, loc: s)
        else:
            return {}

    def save_training_state(self, state):
        cp_path = self.ws.checkpoint_path / \
            (self.__class__.__name__ + '_state.pt')
        torch.save(state, str(cp_path))

    def make_state(self, action, ob):
        q = self.env.questions[action]
        i = np.concatenate([q['knowledge'].reshape(1, -1),
                            np.array([q['difficulty']]).reshape(1, 1),
                            ob.reshape(1, 1)],
                           axis=1)
        self._inputs.append(i)
        return np.concatenate(self._inputs, axis=0)

    def init_state(self):
        self._inputs = [np.zeros((1, self.env.n_knowledge + 2))]
        return self._inputs[-1]

    def make_batch(self, samples):
        # TODO: make batched sequential inputs for agent network
        states = [torch.tensor(i.state).float() for i in samples]
        actions = [torch.tensor(i.action).long().view(1, 1) for i in samples]
        states_ = [torch.tensor(i.next_state).float() for i in samples]
        rewards = [torch.tensor([i.reward]).float() for i in samples]
        masks = [torch.tensor([i.done]).float() for i in samples]
        states, actions, states_, rewards, masks = \
            make_batch(states, actions, states_, rewards, masks, seq=[0, 2])

        return states, actions, states_, rewards, masks
