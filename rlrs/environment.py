import abc
import datetime
import random
from collections import deque
from itertools import islice

import fret
import gym
import gym.spaces as spaces
import numpy as np
import torch
import torch.nn.functional as F
from torchtext import data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from .dataprep import Questions
from .util import critical


@fret.configurable
class _StuEnv(gym.Env, abc.ABC):
    """Simulated environment based on some kind of Score Prediction."""

    def __init__(self,
                 dataset=('example', 'student record dataset',
                          ['example', 'zhixue_small', 'zhixue',
                           'poj', 'ustcoj']),
                 expected_avg=(0.5, 'expected average score'),
                 r_lambdas=([1., 1., 1., 1.], 'lambdas for four rewards')):
        super(_StuEnv, self).__init__()
        self.dataset = dataset
        self.questions = Questions(dataset)
        self.knowledge = self.questions.knowledge
        self.n_questions = self.questions.n_questions
        self.n_knowledge = self.questions.n_knowledge
        self.n_words = self.questions.n_words
        self.records = []
        self.expected_avg = expected_avg
        self.r_lambdas = r_lambdas

        # session history
        self._history = []
        self._seen_knows = set()
        self._scores = []
        self._maxlen = 20

        self.action_space = spaces.Discrete(self.n_questions)
        # only provide current step observation: score
        # agent should keep track of the history separately
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,),
                                            dtype=np.float32)

        self._stop = False

    def random_action(self):
        return self.action_space.sample()

    def reset(self):
        # new session
        # new session
        self._history = []
        self._seen_knows = set()
        self._scores = deque()
        self._stop = False
        self._last_bias = 0.
        self.sample_student()

    def step(self, action):
        q = self.questions[action]

        # generate predicted score
        score = self.exercise(q)

        observation = np.asarray([score])
        reward = self.get_reward(q, score)
        if len(self._history) > self._maxlen:
            self.stop()
        return observation, reward, self._stop, {}

    def stop(self):
        self._stop = True

    def render(self, mode='human'):
        pass

    def get_reward(self, question, score):
        # get_reward from session history
        self._history.append(question['id'])
        self._scores.append(score)
        know = self.questions._ques_know[question['id']]

        r_exploration = 0.
        for k in know:
            if k not in self._seen_knows:
                r_exploration += 1
                self._seen_knows.add(k)

        if len(self._history) == 1:
            return self.r_lambdas[0] * r_exploration

        r_exploitation = 0.
        last_know = self.questions._ques_know[self._history[-2]]
        if not set(last_know) & set(know) and \
                self._scores[-2] < self.expected_avg:
            r_exploitation -= 1

        diff = self.questions._ques_diff[question['id']]
        last_diff = self.questions._ques_diff[self._history[-2]]
        r_smoothness = -(diff - last_diff) ** 2

        r_satisfaction = -abs(np.mean(self._scores) - self.expected_avg)

        rewards = [r_exploration, r_exploitation, r_smoothness, r_satisfaction]
        return sum(a * r for a, r in zip(self.r_lambdas, rewards))

    def set_records(self, records):
        self.records = records

    @abc.abstractmethod
    def sample_student(self):
        raise NotImplementedError

    @abc.abstractmethod
    def exercise(self, q):
        raise NotImplementedError


@fret.configurable
class RandomEnv(_StuEnv):
    def __init__(self, **cfg):
        super(RandomEnv, self).__init__(**cfg)
        self._know_state = None

    def sample_student(self):
        """Reset environment state. Here we sample a new student."""
        self._know_state = np.random.rand(self.n_knowledge,)

    def exercise(self, q):
        """Receive an action, returns observation, reward of current step,
        whether the game is done, and some other information."""
        diff = q['difficulty']
        # get index for each knowledge
        know = q['knowledge']

        # set score to 1 if a student masters all knowledge of this question
        if all(self._know_state[i] > diff
               for i, x in enumerate(know) if x > 0.5):
            score = 1.
        else:
            score = 0.
        return score


@fret.configurable
class OffPolicyEnv(_StuEnv):
    def __init__(self, **cfg):
        super(OffPolicyEnv, self).__init__(**cfg)
        self.record = None
        self.qids = None
        self.scores = None

    def random_action(self):
        q = random.choice(self.qids)
        while len(self._history) > len(self.qids) and q in self._history:
            q = random.choice(self.qids)
        return self.questions.stoi[q]

    def sample_student(self):
        """Reset environment state. Here we sample a new student."""
        assert len(self.records) > 0, 'no record found'
        r = self.record = random.choice(self.records)
        while len(r.question) < 20:
            r = self.record = random.choice(self.records)
        self._maxlen = 12
        self.scores = {q: s for q, s in zip(r.question, r.score)}
        self.qids = r.question

    def exercise(self, q):
        """Receive an action, returns observation, reward of current step,
        whether the game is done, and some other information."""
        return self.scores[q['id']]


@fret.configurable
class DeepSPEnv(_StuEnv):
    def __init__(self, sp_model, **cfg):
        super(DeepSPEnv, self).__init__(**cfg)
        self.sp_model = sp_model(_dataset=self.dataset,
                                 _questions=self.questions)

    def sample_student(self):
        self.state = None
        # sample some records
        records = random.choice(self.records) if self.records else []
        # feed into self.sp_model to get state
        with torch.no_grad():
            for qid, score in zip(records.question, records.score):
                q = self.questions[qid]
                s = torch.tensor([score])
                _, self.state = self.sp_model(q, s, self.state)
        self.qids = random.sample(list(self.questions.stoi), 100)

    def exercise(self, q):
        with torch.no_grad():
            s, self.state = self.sp_model(q, None, self.state)
        return int(s.mean().item() > 0.5)

    def train(self, args):
        ws = self.ws
        records = self.records
        logger = ws.logger('DeepSPEnv.train')

        model = self.sp_model
        model.train()
        optim = torch.optim.Adam(model.parameters())
        train_iter = data.Iterator(records, 1)  # one sequence at a time
        epoch_size = len(train_iter)

        state = self.load_training_state()
        if not args.restart and state:
            self.load_model('int')
            train_iter.load_state_dict(state['train_iter'])
            optim.load_state_dict(state['optim'])
            current_run = state['current_run']
            loss_avg, mae_avg, acc_avg = state['avg']
            start_epoch = train_iter.epoch
            n_samples = state['n_samples']
            initial = train_iter._iterations_this_epoch
        else:
            if not args.restart:
                logger.info('nothing to resume, starting from scratch')
            n_samples = 0  # track total #samples for plotting
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            current_run = str(self.ws.log_path /
                              ('DeepSPEnv.train/run-%s/' % now))
            loss_avg = []
            mae_avg = []
            acc_avg = []
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
                    maes = []
                    for q, s in zip(batch.question, batch.score):
                        q_index = q[0].item()
                        if q_index == -1:
                            continue
                        q = self.questions[q_index]
                        q['text'] = torch.tensor(q['text'])
                        q['knowledge'] = torch.tensor(q['knowledge'])
                        q['difficulty'] = torch.tensor([q['difficulty']])
                        s = s.float()
                        s_, hidden = model(q, s, hidden)
                        losses.append(F.mse_loss(s_.view(1), s).view(1))
                        maes.append(F.l1_loss(s_.view(1), s).item())

                    if not losses:
                        continue

                    loss = torch.cat(losses).mean()
                    loss.backward()
                    optim.step()

                    # log loss
                    loss_avg.append(loss.item())
                    mae_avg.extend(maes)
                    acc_avg.extend(np.asarray(maes) < 0.5)
                    if args.log_every == len(loss_avg):
                        writer.add_scalar('DeepSPEnv.train/loss',
                                          np.mean(loss_avg),
                                          n_samples)
                        writer.add_scalar('DeepSPEnv.train/mae',
                                          np.mean(mae_avg), n_samples)
                        writer.add_scalar('DeepSPEnv.train/acc',
                                          np.mean(acc_avg), n_samples)
                        loss_avg = []
                        mae_avg = []
                        acc_avg = []

                    # save model
                    if args.save_every > 0 and i % args.save_every == 0:
                        self.save_model(f'{epoch}.{i}')

                # save after one epoch
                self.save_model(epoch + 1)

            except KeyboardInterrupt:
                self.save_training_state({
                    'current_run': current_run,
                    'optim': optim.state_dict(),
                    'train_iter': train_iter.state_dict(),
                    'n_samples': n_samples,
                    'avg': (loss_avg, mae_avg, acc_avg)
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
