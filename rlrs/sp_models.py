import os

import fret
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

from .dataprep import load_embedding, Questions


@fret.configurable
class EERNN(nn.Module):
    def __init__(self, _dataset, _wcnt,
                 emb_size=(50, 'embedding size'),
                 ques_h_size=(50, 'question embedding set'),
                 seq_h_size=(50, 'hidden size of sequence model'),
                 n_layers=(1, 'number of layers of RNN'),
                 attn_k=(10, 'top k records for attention in EERNN model')):
        super(EERNN, self).__init__()
        self.wcnt = _wcnt
        self.emb_size = emb_size
        self.ques_h_size = ques_h_size
        self.seq_h_size = seq_h_size
        self.n_layers = n_layers
        self.attn_k = attn_k

        self.question_net = QuesNet(_wcnt, ques_h_size, emb_size)
        if _dataset:
            emb_file = 'data/%s/emb_%d.txt' % (_dataset, emb_size)
            if os.path.exists(emb_file):
                embs = load_embedding(emb_file)
                self.question_net.load_emb(embs)

        self.seq_net = EERNNSeqNet(ques_h_size, seq_h_size, n_layers, attn_k)

    def forward(self, question, score, hidden=None):
        # question: {'id': ..., 'text': ...,
        #            'difficulty': ..., 'knowledge': ...}
        ques_text = question['text']
        ques_v = self.question_net(ques_text)
        s, h = self.seq_net(ques_v[0], score, hidden)

        if hidden is None:
            hidden = ques_v, h
        else:
            # concat all qs and hs for attention
            qs, hs = hidden
            qs = torch.cat([qs, ques_v], dim=0)
            hs = torch.cat([hs, h])
            hidden = qs, hs

        return s, hidden


class DKT(nn.Module):
    def __init__(self, _dataset):
        super(DKT, self).__init__()
        self.dataset = _dataset
        self.n_knowledge = self.questions.n_knowledge

        self.seq_net = DKTNet(self.n_knowledge)

    def forward(self, knowledge_vec, score, hidden=None):
        s, hidden = self.seq_net(knowledge_vec, score, hidden)
        return s, hidden


'''
模型各种模块
'''


class QuesNet(nn.Module):
    def __init__(self, wcnt, hidden_size, emb_size=None):
        super().__init__()
        self.wcnt = wcnt
        self.hidden_size = hidden_size
        if not emb_size:
            emb_size = hidden_size // 2  # bidirectional
        self.embedding = nn.Embedding(wcnt, emb_size)
        self.rnn = nn.LSTM(emb_size, hidden_size // 2, 1,
                           bidirectional=True, batch_first=True)
        self.h0 = nn.Parameter(torch.rand(2, 1, hidden_size // 2))
        self.c0 = nn.Parameter(torch.rand(2, 1, hidden_size // 2))

    def forward(self, q):
        if isinstance(q, PackedSequence):
            bs = q.batch_sizes[0]
            emb = self.embedding(q.data)
            emb = PackedSequence(emb, q.batch_sizes)
        else:
            q = q.unsqueeze(0)  # batch
            bs = 1
            emb = self.embedding(q)
        h = self.init_h(bs)
        y, h = self.rnn(emb, h)
        return h[0].view(bs, -1)

    def init_h(self, batch_size):
        size = list(self.h0.size())
        size[1] = batch_size
        return self.h0.expand(size), self.c0.expand(size)

    def load_emb(self, emb):
        self.embedding.weight.data.copy_(torch.from_numpy(emb))
        self.embedding.weight.requires_grad = False


class EERNNSeqNet(nn.Module):
    def __init__(self,
                 ques_size=100,
                 seq_hidden_size=50,
                 n_layers=1,
                 attn_k=10
                 ):
        super(EERNNSeqNet, self).__init__()

        self.initial_h = nn.Parameter(torch.zeros(n_layers *
                                                  seq_hidden_size))
        self.ques_size = ques_size  # exercise size
        self.seq_hidden_size = seq_hidden_size
        self.n_layers = n_layers
        self.attn_k = attn_k

        # initialize network
        self.seq_net = nn.GRU(self.ques_size * 2, self.seq_hidden_size, self.n_layers)
        self.score_net = nn.Linear(self.ques_size + self.seq_hidden_size, 1)

    def forward(self, question, score, hidden):
        if hidden is None:
            h = self.initial_h.view(self.n_layers, 1, self.seq_hidden_size)
            attn_h = self.initial_h
        else:
            questions, hs = hidden
            h = hs[-1:]
            alpha = torch.mm(questions, question.view(-1, 1)).view(-1)
            alpha, idx = alpha.topk(min(len(alpha), self.attn_k), sorted=False)
            alpha = nn.functional.softmax(alpha.view(1, -1), dim=-1)

            # flatten each h
            hs = hs.view(-1, self.n_layers * self.seq_hidden_size)
            attn_h = torch.mm(alpha, torch.index_select(hs, 0, idx)).view(-1)

        # prediction
        pred_v = torch.cat([question, attn_h]).view(1, -1)
        pred = self.score_net(pred_v)

        if score is None:
            score = pred.flatten()

        # update seq_net
        x = torch.cat([question * (score >= 0.5).type_as(question).expand_as(question),
                       question * (score < 0.5).type_as(question).expand_as(question)])

        _, h_ = self.seq_net(x.view(1, 1, -1), h)
        return pred, h_


class DKTNet(nn.Module):
    """
    做题记录序列的RNN（GRU）单元
    """

    def __init__(self, n_knowledge):
        super(DKTNet, self).__init__()
        self.n_knwoledge = n_knowledge
        self.rnn_net = nn.GRU(n_knowledge * 2, n_knowledge, 1)
        self.score_net = nn.Linear(n_knowledge * 2, 1)

    def forward(self, knowledge_v, score, h):
        if h is None:
            h = self.default_hidden()

        knowledge_v = knowledge_v.type_as(h)
        score = self.score_net(torch.cat([h.view(-1), knowledge_v.view(-1)]))

        x = torch.cat([knowledge_v.view(-1),
                       (knowledge_v * (score > 0.5).type_as(knowledge_v).
                        expand_as(knowledge_v).type_as(knowledge_v)).view(-1)])
        _, h = self.rnn_net(x.view(1, 1, -1), h)
        return score.view(1), h

    def default_hidden(self):
        return torch.zeros(1, 1, self.topic_size)
