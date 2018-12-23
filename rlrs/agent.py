import fret
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .dataprep import load_question, load_knowledge, load_embedding
from .environment import QuesNet

# to be update about N_actions and N_states
N_STATES = 100
N_ACTIONS = 50

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
        self.memory = np.zeros(memory_capacity, )
        self.learning_rate = learning_rate
        self.greedy_epsilon = greedy_epsilon
        self.gama = gama

        # build DQN network
        self.current_net, self.target_net = Net(), Net()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def select_action(self, observation):

        pass


# RL net which generates action(recommended question) at each step
class Net(nn.Module):
    def __init__(self,
                 emb_file=(None, 'pretrained embedding file'),
                 ques_size=(50, 'question embedding set'),
                 ):
        super(Net, self).__init__()

        self.input_net = nn.Linear(N_STATES, 200)
        self.input_net.weight.data.normal_(0, 0.1)
        self.out_net = nn.Linear(200, N_ACTIONS)
        self.out_net.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.input_net(x))
        out = self.out_net(x)
        return out

