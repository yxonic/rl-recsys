import fret
import numpy as np
import random
import torch
import torch.nn as nn
from .dataprep import Questions
from .agent import DQN

TARGET_REPLACE_ITER = 100

@fret.configurable
class Trainer:
    def __init__(self,
                 env=(None, 'environment'),
                 agent=(None, 'RL agent'),
                 dataset=('zhixue', 'student record dataset',
                          ['zhixue', 'poj', 'ustcoj']),
                 memory_capacity=(500, 'memory size'),
                 ):
        self.env = env
        self.agent = agent
        self.dataset = dataset
        self.memory_capacity = memory_capacity

        self.rec_history = []
        self.reward_history = []
        self.pred_score_history = []

        self.dataset = dataset
        self._questions = Questions(dataset)
        self.n_questions = self._questions.n_questions

    def train_agent(self, args):
        logger = self.ws.logger('Trainer.train_agent')
        if self.env is None:
            logger.info('no environment')
            exit(0)
        # self.load_env()

        if self.agent is None:
            logger.info('no agent loading, build new agent')
            self.agent = DQN(memory_capacity=500,
                             learning_rate=0.001,
                             greedy_epsilon=0.9,
                             gama=0.9,
                             n_actions=self.n_questions,
                             double_q=False)
        # self.load_agent()

        # RL agent training process here
        for epoch in range(args.epoch):
            # start process: reset env
            self.rec_history = []
            self.reward_history = []
            self.pred_score_history = []
            self.env.reset()

            ep_reward = 0
            while True:
                if len(self.rec_history) == 0:
                    # TODO: randomly generate a first recommendation
                    action = self.agent.random_select_action()
                else:
                    state_score = self.pred_score_history[-1]
                    action = self.agent.select_action(state_score)

                # take action in env
                state_score_, reward, done, info = self.env.step(action)

                # save records
                self.rec_history.append(action)
                self.reward_history.append(reward)
                self.pred_score_history.append(state_score_)

                ep_reward += reward

                # update parameters in agent
                #TODO: training on batch? and calculate q_current, q_next
                self.agent.learn(self.rec_history[-1], state_score, action, reward, state_score_)

    def load_agent(self, path=None):
        logger = self.ws.logger('Trainer.load_agent')
        logger.debug('func: <trainer.load_agent>, path=%s', path)

    def load_env(self, path=None):
        logger = self.ws.logger('Trainer.load_env')
        logger.debug('func: <trainer.load_env>, path=%s', path)

    def train(self, args):
        logger = self.ws.logger('Trainer.train')
        logger.debug('func: <trainer.train>, args: %s', args)
