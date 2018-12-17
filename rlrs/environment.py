import random

import fret
import gym
import gym.spaces as spaces
import numpy as np
from .dataprep import load_question, load_knowledge


@fret.configurable
class RandomEnv(gym.Env):
    def __init__(self, dataset='zhixue', expected_avg=0.5):
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
        reward = -abs(np.mean(self._scores) - self.expected_avg)
        done = len(self._scores) > 20  # stop after 20 questions
        return observation, reward, done, {}


@fret.configurable
class EKT:
    def __init__(self, n_knowledges=10):
        pass
