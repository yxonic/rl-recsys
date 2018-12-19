import fret
import gym
import gym.spaces as spaces
import numpy as np
from .dataprep import load_question, load_knowledge
import torch.nn as nn


@fret.configurable
class RandomEnv(gym.Env):
    def __init__(self, dataset='zhixue', expected_avg=0.5):
        self.dataset = dataset
        self.ques_list = load_question(
            fret.app['datasets'][dataset]['question_file'])
        self.know_list, self.know_ind_map = load_knowledge(
            fret.app['datasets'][dataset]['knowledge_file'])
        self.n_questions = len(self.ques_list)
        self.n_knowledge = len(self.know_list)
        self.expected_avg = expected_avg

        self._scores = []

        self.action_space = spaces.Discrete(self.n_questions)
        # only provide current step observation: score
        # agent should keep track of the history separately
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,),
                                            dtype=np.float32)
        self.reset()

    def reset(self):
        """Reset environment state. Here we sample a new student."""
        self._know_state = np.random.rand(self.n_knowledge,)

    def step(self, action):
        """Receive an action, returns observation, reward of current step,
        whether the game is done, and some other information."""
        q = self.ques_list[action]
        diff = q['difficulty']
        # get index for each knowledge
        know = [self.know_ind_map[k] for k in q['knowledge']]

        # set score to 1 if a student masters all knowledge of this question
        if all(self._know_state[s] > diff for s in know):
            score = 1.
        else:
            score = 0.
        self._scores.append(score)
        observation = [score]
        reward = self.get_reward()
        done = len(self._scores) > 20  # stop after 20 questions
        return observation, reward, done, {}

    def get_reward(self):
        """Calculate reward from internal states."""
        return -abs(np.mean(self._scores) - self.expected_avg)

    def train(self, records, n_epochs):
        """Train environment model here under ws with record data."""
        logger = self.ws.logger('RandomEnv.train')
        logger.debug('func: <RandomEnv.train>, n_epochs=%d', n_epochs)


@fret.configurable
class EKT:
    def __init__(self, n_knowledges=10):
        pass



class ExerciseNet(nn.Module):
    def __init__(self, wcnt, emb_size=100, exc_size=50, n_layers=1):
        super(ExerciseNet, self).__init__()
        self.wcnt = wcnt
        self.emb_size = emb_size
        self.exc_size = exc_size
        self.n_layers = n_layers

        self.embedding_net = nn.Embedding(wcnt, self.emb_size, padding_idx=0)

        self.emb_size = exc_size // 2
        self.exc_net = nn.GRU(self.emb_size, self.exc_size // 2, self.n_layers,
                              bidirectional=True)

    def forward(self, input, hidden):
        x = self.embedding_net(input)
        y, h = self.exc_net(hidden)

        y, _ = torch.max(y, 0)
        return y, h

    def load_emb(self, emb):
        self.embedding.weight.data.copy_(torch.from_numpy(emb))