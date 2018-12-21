import fret
import torch
import torch.nn as nn
import numpy as np
from .dataprep import load_question, load_knowledge, load_embedding
from .environment import QuesNet

# to be update about N_actions and N_states
N_STATES = 10
N_ACTIONS = 5

@fret.configurable
class DQN:
    def __init__(self,
                 double_q=False,
                 ):
        pass


# RL net which generates action(recommended question) at each step
class Net(nn.Module):
    def __init__(self,
                 emb_file=(None, 'pretrained embedding file'),
                 ques_size=(50, 'question embedding set'),
                 ):
        super(Net, self).__init__()

        self.input_net = nn.Linear(N_STATES, 50)
        self.input_net.weight.data.normal_(0, 0.1)
        self.out_net = nn.Linear(50, N_ACTIONS)
        self.out_net.weight.data.normal_(0, 0.1)
