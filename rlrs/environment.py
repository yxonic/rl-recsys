import abc
import fret
import gym
import gym.spaces as spaces
import numpy as np
from .dataprep import load_question, load_knowledge


@fret.configurable
class SPEnv(gym.Env, abc.ABC):
    """Simulated environment based on some kind of Score Prediction."""

    def __init__(self,
                 dataset=('zhixue', 'student record dataset',
                          ['zhixue', 'poj', 'ustcoj']),
                 expected_avg=(0.5, 'expected average score')):
        self.dataset = dataset

        self.ques_list = load_question(
            fret.app['datasets'][dataset]['question_file'])
        self.know_list, self.know_ind_map = load_knowledge(
            fret.app['datasets'][dataset]['knowledge_file'])
        self.n_questions = len(self.ques_list)
        self.n_knowledge = len(self.know_list)

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
        self.know_state = np.random.rand(self.n_knowledge, )

    def exercise(self, q):
        """Receive an action, returns observation, reward of current step,
        whether the game is done, and some other information."""
        diff = q['difficulty']
        # get index for each knowledge
        know = [self.know_ind_map[k] for k in q['knowledge']]

        # set score to 1 if a student masters all knowledge of this question
        if all(self._know_state[s] > diff for s in know):
            score = 1.
        else:
            score = 0.
        return score


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
        pass
