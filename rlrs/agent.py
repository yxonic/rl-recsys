import fret
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .sp_models import QuesNet
from .dataprep import Questions

# to be update about N_actions and N_states
N_STATES = 100
N_ACTIONS = 50
TARGET_REPLACE_ITER = 100   # target update frequency


@fret.configurable
class DQN:
    def __init__(self,
                 dataset=('zhixue', 'student record dataset',
                          ['zhixue', 'poj', 'ustcoj']),
                 memory_capacity=(2000, 'max saved memory states'),
                 learning_rate=(0.001, 'learning rate'),
                 greedy_epsilon=(0.9, 'greedy policy'),
                 gama=(0.9, 'reward discount rate'),
                 double_q = False,
                 ):
        self.learn_step_counter = 0
        self.learning_rate = learning_rate
        self.greedy_epsilon = greedy_epsilon
        self.gama = gama

        self.dataset = dataset
        self._questions = Questions(dataset)
        self.n_questions = self._questions.n_questions

        self.memory_counter = 0
        self.memory_capacity = memory_capacity
        self.memory = np.zeros(memory_capacity, self.n_questions * 2 + 2)

        self._rec_history = []
        self._pred_scores = []

        # build DQN network
        self.current_net, self.target_net = SimpleNet(), SimpleNet()

        # self.optimizer = torch.optim.Adam(self.current_net.parameters(), lr=learning_rate)
        # self.loss_func = nn.MSELoss()

    def select_action(self, observation):
        # to be updated of input observation
        [p_score] = observation
        self._pred_scores.append(p_score)

        # generate
        _action = self._rec_history[-1]
        feature = self.generate_state_feature(_action, p_score)
        feature = torch.unsqueeze(torch.FloatTensor(feature), 0)

        if np.random.uniform() < self.greedy_epsilon:
            action_values = self.current_net(feature)
            action = torch.max(action_values, 1)[1].data.numpy()
        else:
            action = np.random.randint(0, self.n_questions)

        self._rec_history.append(action)
        return action

    # to be updated, 需要设定到几步之后才train的嘛
    def store_transition(self, state, action, reward, state_):
        transition = np.hstack((state, action, reward, state_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index:] = transition
        self.memory_counter += 1

    # to be updated
    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.current_net.state_dict())
        self.learn_step_counter += 1

        # to be updated, training on batch ?
        sample_index = np.random.choice(self.memory_capacity, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :1])

        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_current = self.current_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gama * q_next.max(1)[0]

        loss = self.loss_func(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def generate_state_feature(self, action, score):
        feature_hot = np.zeros(self.n_questions)
        feature_hot[action] = score
        return feature_hot

        # feature_hot = torch.zeros(1, self.n_questions)
        # feature_hot = feature_hot.scatter_(dim=0, index=torch.from_numpy(action), value=score)
        # return feature_hot


# RL net which generates action(recommended question) at each step
# to be updated: input question or question_emb ?
class SimpleNet(nn.Module):
    def __init__(self,
                 n_actions = (100, 'number of questions'),
                 state_feature_size = (50, 'size of state'),
                 ques_h_size=50):
        super(SimpleNet, self).__init__()

        self.n_actions = n_actions
        self.state_feature_size = state_feature_size
        self.ques_h_size = ques_h_size

        self.input_net = nn.Linear(self.state_feature_size, 200)
        self.out_net = nn.Linear(200, self.n_actions)
        # self.input_net.weight.data.normal_(0, 0.1)
        # self.out_net.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # input current state: (question, score)
        # generate next action values (question_)
        x = F.relu(self.input_net(x))
        action_values = self.out_net(x)
        return action_values

