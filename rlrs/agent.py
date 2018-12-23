import fret
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .dataprep import load_question, load_knowledge, load_embedding
from .sp_models import QuesNet

# to be update about N_actions and N_states
N_STATES = 100
N_ACTIONS = 50
TARGET_REPLACE_ITER = 100   # target update frequency


@fret.configurable
class DQN:
    def __init__(self,
                 double_q=False,
                 memory_capacity=(2000, 'max saved memory states'),
                 learning_rate=(0.001, 'learning rate'),
                 greedy_epsilon=(0.9, 'greedy policy'),
                 gama=(0.9, 'reward discount rate')
                 ):
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory_capacity = memory_capacity
        self.memory = np.zeros(memory_capacity, )
        self.learning_rate = learning_rate
        self.greedy_epsilon = greedy_epsilon
        self.gama = gama

        # build DQN network
        self.current_net, self.target_net = Net(), Net()
        self.optimizer = torch.optim.Adam(self.current_net.parameters(),
                                          lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def select_action(self, observation):
        # to be updated of input observation
        question, score = observation
        x = torch.unsqueeze(torch.FloatTensor(observation), 0)
        if np.random.uniform() < self.greedy_epsilon:
            action_values = self.current_net(x)
            action = torch.max(action_values, 1)[1].data.numpy()
        else:
            action = np.random.randint(0, N_ACTIONS)

        return action

    # to be updated, 需要设定到几步之后才train的嘛
    def store_transition(self, state, action, reward, state_):
        pass

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.current_net.state_dict())

        # to be updated, training on batch ?
        sample_index = np.random.choice(self.memory_capacity, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
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


# RL net which generates action(recommended question) at each step
# to be updated: input question or question_emb ?
class Net(nn.Module):
    def __init__(self, emb_file='data/emb_50.txt', ques_h_size=50):
        super(Net, self).__init__()

        self.emb_file = emb_file
        wcnt, emb_size, words, embs = load_embedding(self.emb_file)
        self.wcnt = wcnt
        self.emb_size = emb_size
        self.words = words
        self.embs = embs
        self.ques_h_size = ques_h_size

        self.input_net = nn.Linear(N_STATES, 200)
        self.input_net.weight.data.normal_(0, 0.1)
        self.out_net = nn.Linear(200, N_ACTIONS)
        self.out_net.weight.data.normal_(0, 0.1)

        self.question_net = QuesNet(self.wcnt, self.emb_size, self.ques_h_size, 1)

    def forward(self, x):
        ques_h0 = None
        ques_v, ques_h = self.question_net(x.view(-1, 1), ques_h0)

        x = F.relu(self.input_net(ques_v))
        out = self.out_net(x)
        return out

