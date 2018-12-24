import abc
import datetime
import sys
from itertools import islice

import fret
import gym
import gym.spaces as spaces
import numpy as np
import torch
import torch.nn.functional as F
from statistics import mean
from torchtext import data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from .dataprep import Questions
from .util import critical


@fret.configurable
class SPEnv(gym.Env, abc.ABC):
    """Simulated environment based on some kind of Score Prediction."""

    def __init__(self,
                 dataset=('zhixue', 'student record dataset',
                          ['zhixue', 'poj', 'ustcoj']),
                 expected_avg=(0.5, 'expected average score')):
        self.dataset = dataset
        self.questions = Questions(dataset)
        self.n_questions = self.questions.n_questions
        self.n_knowledge = self.questions.n_knowledge

        self.expected_avg = expected_avg

        # session history
        self._questions = []
        self._scores = []

        self.action_space = spaces.Discrete(self.n_questions)
        # only provide current step observation: score
        # agent should keep track of the history separately
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,),
                                            dtype=np.float32)

        self.reset()

    def reset(self):
        # new session
        self._questions = []
        self._scores = []
        self.sample_student()

    def step(self, action):
        self._questions.append(action)
        q = self.ques_list[action]
        score = self.exercise(q)
        self._scores.append(score)

        observation = [score]
        reward = self.get_reward()
        done = len(self._scores) > 20  # TODO: configure stop condition
        return observation, reward, done, {}

    def render(self, mode='human'):
        pass

    def get_reward(self):
        # get_reward from session history

        # four reward
        # R_coverage, R_change, R_difficulty, R_expected

        question = self._questions[-1]
        predict_score = self._scores[-1]

        question_diff = question['difficulty']
        question_know = [self.know_ind_map[k] for k in question['knowledge']]

        # calculate reward in current state
        length = len(self._questions)
        if length == 0:
            reward = 0
        else:
            # coverage reward: R = -1 if current know exists in past know lists
            past_know_list = []
            for _q in self._questions:
                _q_know = [self.know_ind_map[k] for k in _q['knowledge']]
                past_know_list = past_know_list + _q_know
            past_know_list = set(past_know_list)
            if set(question_know).issubset(past_know_list):
                R_coverage = -1
            else:
                R_coverage = 0

            # change reward: R = 1 if score of current ques = 1
            if predict_score == 1:
                R_change = 1
            else:
                R_change = 0

            # difficulty reward
            _q = self._questions[-1]
            R_difficulty = - ((question_diff - _q['difficulty']) ** 2)

            # expected reward
            step = 5
            if length < step:
                _qs = self._scores[1:]
            else:
                _qs = self._questions[-step:]
            R_expected = 1 - np.abs(1 - np.mean(np.asarray(_qs)))

            reward = R_expected + R_coverage + R_difficulty + R_change

        return reward

    @abc.abstractmethod
    def sample_student(self):
        raise NotImplementedError

    @abc.abstractmethod
    def exercise(self, q):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, records, args):
        raise NotImplementedError


@fret.configurable
class RandomEnv(SPEnv):
    def __init__(self, **cfg):
        self.know_state = None
        super(RandomEnv, self).__init__(**cfg)

    def sample_student(self):
        """Reset environment state. Here we sample a new student."""
        self.know_state = np.random.rand(self.n_knowledge,)

    def exercise(self, q):
        """Receive an action, returns observation, reward of current step,
        whether the game is done, and some other information."""
        diff = q['difficulty']
        # get index for each knowledge
        know = q['knowledge']

        # set score to 1 if a student masters all knowledge of this question
        if all(self._know_state[i] > diff for i in know):
            score = 1.
        else:
            score = 0.
        return score

    def train(self, records, args):
        pass


@fret.configurable
class DeepSPEnv(SPEnv):
    def __init__(self, sp_model, **cfg):
        self.sp_model = sp_model
        self.state = None
        super(DeepSPEnv, self).__init__(**cfg)

    def sample_student(self):
        # sample some records
        records = []
        # feed into self.sp_model to get state
        for q, s in records:
            _, self.state = self.sp_model(q, s, None)

    def exercise(self, q):
        s, self.state = self.sp_model(q, None, self.state)
        return s.mean().item()

    def train(self, records, args):
        ws = self.ws
        logger = ws.logger('DeepSPEnv.train')

        model = self.sp_model
        model.train()
        optim = torch.optim.Adam(model.parameters())
        train_iter = data.Iterator(records, 1)  # one sequence at a time
        epoch_size = len(train_iter)

        state = self.load_training_state()
        if args.resume and state:
            self.load_model('int')
            train_iter.load_state_dict(state['train_iter'])
            optim.load_state_dict(state['optim'])
            current_run = state['current_run']
            loss_avg = state['loss_avg']
            start_epoch = train_iter.epoch
            n_samples = state['n_samples']
            initial = train_iter._iterations_this_epoch
        else:
            if args.resume:
                logger.info('nothing to resume, starting from scratch')
            n_samples = 0  # track total #samples for plotting
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_run = ws.log_path / ('run-%s/' % now)
            loss_avg = []
            start_epoch = 0
            initial = 0

        writer = SummaryWriter(str(current_run))

        for epoch in range(start_epoch, args.n_epochs):
            epoch_iter = iter(tqdm(islice(train_iter, epoch_size - initial),
                                   total=epoch_size,
                                   initial=initial,
                                   desc=f'Epoch {epoch+1:3d}: ',
                                   unit='bz'))

            initial = 0

            try:
                # training
                for batch in critical(epoch_iter):
                    # critical section on one batch
                    i = train_iter._iterations_this_epoch
                    n_samples += len(batch)

                    # backprop on one batch
                    optim.zero_grad()

                    hidden = None
                    losses = []
                    for q, s in zip(batch.question, batch.score):
                        q = self.questions[q[0].item()]
                        s = s.float()
                        s_, hidden = model(q, s, hidden)
                        losses.append(F.mse_loss(s_.view(1), s).view(1))

                    if not losses:
                        continue

                    loss = torch.cat(losses).mean()
                    loss.backward()
                    optim.step()

                    # log loss
                    loss_avg.append(loss.item())
                    if args.log_every == len(loss_avg):
                        writer.add_scalar('DeepSPEnv.train/loss',
                                          mean(loss_avg),
                                          n_samples)
                        loss_avg = []

                    # save model
                    if args.save_every > 0 and i % args.save_every == 0:
                        cp_path = ws / f'model.{epoch}.{i}.pt'
                        torch.save(model.state_dict(), str(cp_path))

                # save after one epoch
                cp_path = ws.checkpoint_path / f'model.{epoch+1}.pt'
                torch.save(model.state_dict(), str(cp_path))

            except KeyboardInterrupt:
                self.save_training_state({
                    'current_run': current_run,
                    'optim': optim.state_dict(),
                    'train_iter': train_iter.state_dict(),
                    'n_samples': n_samples,
                    'loss_avg': loss_avg
                })
                self.save_model('int')
                raise

    def load_model(self, tag):
        cp_path = self.ws.checkpoint_path / ('%s.%s.pt' % (
            self.sp_model.__class__.__name__, str(tag)))
        self.sp_model.load_state_dict(torch.load(
            str(cp_path), map_location=lambda s, loc: s))

    def save_model(self, tag):
        cp_path = self.ws.checkpoint_path / ('%s.%s.pt' % (
            self.sp_model.__class__.__name__, str(tag)))
        torch.save(self.sp_model.state_dict(), cp_path)

    def load_training_state(self):
        cp_path = self.ws.checkpoint_path / 'training_state.pt'
        if cp_path.exists():
            return torch.load(str(cp_path), map_location=lambda s, loc: s)
        else:
            return {}

    def save_training_state(self, state):
        cp_path = self.ws.checkpoint_path / 'training_state.pt'
        torch.save(state, str(cp_path))
