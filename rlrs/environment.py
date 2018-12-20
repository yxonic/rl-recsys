import fret
import gym
import gym.spaces as spaces
import numpy as np
from .dataprep import load_question, load_knowledge, load_embedding
import torch
import torch.nn as nn
import torch.nn.functional as F


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
class EERNNEnv:
    def __init__(self, n_knowledges=10):
        pass


class EERNNModel(nn.Module):
    def __init__(self, dataset_file, emb_file, exc_size=50, seq_h_size=50, n_layers=1, attn_k=10):
        super(EERNNModel, self).__init__()
        self.emb_file = emb_file
        wcnt, emb_size, words, embs = load_embedding(self.emb_file)
        self.wcnt = wcnt
        self.emb_size = emb_size
        self.words = words
        self.embs = embs
        self.exc_h_size = exc_size
        self.seq_h_size = seq_h_size
        self.n_layers = n_layers
        self.attn_k = 10

        self.exercise_net = ExerciseNet(self.wcnt, self.emb_size, self.exc_h_size, self.n_layers)
        self.exercise_net.load_emb(self.embs)

        self.seq_net = EERNNSeqNet(dataset_file, 10, self.exc_h_size, self.seq_h_size, self.n_layers,
                                   self.n_layers, self.attn_k)

    def forward(self, exec, score, time, hidden=None):
        exec_hidden = None
        exec_v, exec_h = self.exercise_net(exec.view(-1, 1), exec_hidden)
        s, h = self.seq_net(exec_v[0], score, hidden)
        if hidden is None:
            hidden = exec_v, h
        else:
            excs, hs = hidden
            excs = torch.cat([excs, exec_v])
            hs = torch.cat([hs, h])
            hidden = excs, hs

        return s, hidden


class ExerciseNet(nn.Module):
    def __init__(self, wcnt, emb_size=100, exc_size=50, n_layers=1):
        super(ExerciseNet, self).__init__()
        self.wcnt = wcnt
        self.emb_size = emb_size
        self.exc_size = exc_size
        self.n_layers = n_layers

        self.embedding_net = nn.Embedding(wcnt, self.emb_size, padding_idx=0)

        self.emb_size = exc_size // 2
        self.exercise_net = nn.GRU(self.emb_size, self.exc_size // 2, self.n_layers,
                              bidirectional=True)

    def forward(self, input, hidden):
        x = self.embedding_net(input)
        y, h = self.exercise_net(x, hidden)

        y, _ = torch.max(y, 0)
        return y, h

    def load_emb(self, emb):
        self.embedding_net.weight.data.copy_(torch.from_numpy(emb))


class EERNNSeqNet(nn.Module):
    def __init__(self,
                 dataset = 'dataset',
                 n_knowledges=10,
                 exc_size=100,
                 hidden_size=50,
                 n_layers=1,
                 attn_k=10
                 ):
        super(EERNNSeqNet, self).__init__()
        self.dataset = dataset
        self.ques_list = load_question(
            fret.app['datasets'][dataset]['question_file'])
        self.know_list, self.know_ind_map = load_knowledge(
            fret.app['datasets'][dataset]['knowledge_file'])
        self.n_questions = len(self.ques_list)
        self.n_knowledge = len(self.know_list)

        self.exc_size = exc_size  # exercise size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.attn_k = attn_k

        # initialize network
        self.seq_net = nn.GRU(self.exc_size * 2, self.hidden_size, self.n_layers)
        self.score_out_net = nn.Linear(self.exc_size + self.hidden_size, 1)

    def forward(self, exc, score, hidden):
        excs, hs = hidden
        h = hs[-1:]

        # prediction
        alpha = torch.mm(excs, exc.view(-1, 1)).view(-1)
        alpha, idx = alpha.topk(min(len(alpha), self.attn_k), sorted=False)
        alpha = F.softmax(alpha.view(-1, 1), dim=-1)

        hs = hs.view(-1, self.num_layers * self.seq_hidden_size)
        attn_h = torch.mm(alpha, torch.index_select(hs, 0, idx)).view(-1)

        pred_v = torch.cat([exc, attn_h]).view(1, -1)
        pred = self.score_out_net(pred_v)

        # update seq_net
        x = torch.cat([exc * (score >= 0.5).type_as(exc).expand_as(exc),
                       exc * (score < 0.5).type_as(exc).expand_as(exc)])

        _, h_ = self.seq_net(x.view(1, 1, -1), h)
        return pred, h_