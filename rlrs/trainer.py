import datetime
import random
from collections import namedtuple, deque

import fret
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from .environment import _StuEnv
from .agent import DQN
from .dataprep import load_record
from .util import critical, make_batch, Accumulator

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward',
                         'done', 'action_mask'))


@fret.configurable
class ValueBasedTrainer:
    def __init__(self, env, agent,
                 memory_capacity=(500, 'replay memory size'),
                 exploration_p=(0.5, ('exploration probability, set to 1 for '
                                      'off policy training')),
                 maxlen=(20, 'max sequence length')):
        self.exploration_p = exploration_p
        self.env: _StuEnv = env()
        self.agent: DQN = agent(questions=self.env.questions)
        self.replay_memory = self.agent.make_replay_memory(memory_capacity)
        self.maxlen = maxlen
        self._inputs = deque(maxlen=maxlen)

    def train(self, args):
        logger = self.ws.logger('ValueBasedTrainer.train')
        logger.debug('func: <trainer.train>, args: %s', args)

        rec_file = fret.app['datasets'][self.env.dataset]['record_file']
        records = load_record(rec_file, self.env.questions)
        self.env.set_records(records)

        if not args.restart:
            state = self.load_training_state()
        else:
            state = None

        if state:
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
        losses = Accumulator()
        returns = Accumulator()

        writer = SummaryWriter(current_run)

        # RL agent training process here
        for i_episode in critical(tqdm(range(start_episode, args.n_episodes),
                                       initial=start_episode,
                                       total=args.n_episodes)):
            try:
                self.env.reset()
                state = self.agent.reset()

                rewards = []
                action_mask = torch.ones(self.env.n_questions).byte()
                if hasattr(self.env, 'qids'):
                    action_mask[[self.env.questions.stoi[x]
                                for x in self.env.qids]] = 0
                    action_mask = 1 - action_mask

                for _ in critical():
                    i_batch += 1

                    # select action
                    if random.random() <= self.exploration_p:
                        action = self.env.random_action()
                    else:
                        action = self.agent.select_action(
                            torch.tensor(state).float(), action_mask)[0]

                    # mask this action for s' (duplicates are not allowed)
                    action_mask[action] = 0

                    # take action in env
                    ob, reward, done, info = self.env.step(action)

                    # s'
                    state_ = self.agent.step(state, action, ob)

                    rewards.append(reward)

                    # save records
                    self.replay_memory.push(
                        Transition(state, action, state_, reward,
                                   done, action_mask))
                    state = state_

                    if done:
                        ret = 0
                        for r in reversed(rewards):
                            ret = r + self.agent.gamma * ret
                        returns += ret

                        if args.log_every > 0 and \
                                i_episode % (args.log_every // 8) == 0:

                            writer.add_scalar('ValueBasedTrainer.train/return',
                                              returns.mean(), i_episode)
                            returns.reset()

                        break

                    # update parameters in agent
                    # sample from replay memory
                    if len(self.replay_memory) < args.batch_size:
                        continue

                    samples = self.replay_memory.sample(args.batch_size)
                    batch = self.make_batch(samples)

                    # train on batch
                    loss = self.agent.train_on_batch(batch)
                    losses += loss

                    if args.log_every > 0 and i_batch % args.log_every == 0:
                        writer.add_scalar('ValueBasedTrainer.train/loss',
                                          losses.mean(), i_batch)
                        losses.reset()

                if self.exploration_p > 0.05:
                    self.exploration_p *= 0.999

                if args.save_every > 0 and i_episode % args.save_every == 0:
                    self.agent.save_model(i_episode)

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

    def make_batch(self, samples):
        # TODO: make batched sequential inputs for agent network
        states = [i.state for i in samples]
        actions = [torch.tensor([i.action]).long() for i in samples]
        states_ = [i.next_state for i in samples]
        rewards = [torch.tensor([i.reward]).float() for i in samples]
        done = [torch.tensor([i.done]) for i in samples]
        mask = [torch.tensor(i.action_mask).unsqueeze(0) for i in samples]
        states, actions, states_, rewards, done, mask = \
            make_batch(states, actions, states_, rewards, done, mask,
                       seq=[0, 2])
        return states, actions, states_, rewards, done, mask
