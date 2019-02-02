import math
import os

import fret
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

from .dataprep import load_embedding
from .util import SeqBatch


@fret.configurable
class EERNN(nn.Module):
    def __init__(self, _dataset, _questions,
                 emb_size=(50, 'embedding size'),
                 ques_h_size=(50, 'question embedding set'),
                 seq_h_size=(50, 'hidden size of sequence model'),
                 n_layers=(1, 'number of layers of RNN'),
                 attn_k=(10, 'top k records for attention in EERNN model')):
        super(EERNN, self).__init__()
        _wcnt = _questions.n_words
        self.questions = _questions
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

        self.ques_vs = []
        q_text = [q['text'] for q in _questions]
        for i_batch in range(int(math.ceil(len(q_text) / 32))):
            batch = q_text[i_batch * 32:(i_batch + 1) * 32]
            seq = SeqBatch(batch)
            hs = self.question_net(seq.packed()).detach()
            hs = seq.invert(hs, 0)
            self.ques_vs.append(hs)
        self.ques_vs = torch.cat(self.ques_vs, dim=0)
        self.seq_net = EERNNSeqNet(ques_h_size, seq_h_size, n_layers, attn_k)

    def forward(self, question, score, hidden=None):
        # question: {'id': ..., 'text': ...,
        #            'difficulty': ..., 'knowledge': ...}
        question['text'] = torch.tensor(question['text'])
        question['knowledge'] = torch.tensor(question['knowledge'])
        question['difficulty'] = torch.tensor([question['difficulty']])
        ques_text = question['text']
        if self.training:
            ques_v = self.question_net(ques_text)
        else:
            ques_v = self.ques_vs[
                self.questions.stoi[question['id']]].unsqueeze(0)
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


@fret.configurable
class DKT(nn.Module):
    def __init__(self, _dataset, _questions):
        super(DKT, self).__init__()
        self.dataset = _dataset
        self.questions = _questions
        self.n_input = len(_questions.stoi)
        self.seq_hidden_size = 18

        self.seq_net = DKTNet(self.n_input, self.seq_hidden_size)

    def forward(self, q, score, hidden=None):
        k = torch.zeros(self.n_input)
        k[self.questions.stoi[q['id']]] = 1
        s, hidden = self.seq_net(
            torch.tensor(k).float(),
            score, hidden)
        return s, hidden


@fret.configurable
class DKVMN(nn.Module):
    """
    Dynamic Key-Value Memory Networks for Knowledge Tracing at WWW'2017
    """
    def __init__(self, _dataset, _questions):
        super(DKVMN, self).__init__()
        self.dataset = _dataset
        self.knowledge_hidden_size = 18
        self.seq_hidden_size = 18

        self.questions = _questions
        self.kcnt = len(_questions.stoi)
        self.valve_size = self.knowledge_hidden_size * 2

        # knowledge embedding module
        self.knowledge_model = KnowledgeModel(self.kcnt,
                                              self.knowledge_hidden_size)
        # student seq module
        self.seq_model = DKVMNSeqModel(self.knowledge_hidden_size, 30,
                                       self.kcnt, self.seq_hidden_size,
                                       self.valve_size)

    def forward(self, q, score, time, hidden=None):
        k = torch.zeros(self.kcnt)
        k[self.questions.stoi[q['id']]] = 1

        if score is None:
            score = 0
            expand_vec = k.float().view(-1) * score
            # print(expand_vec)
            cks = torch.cat([k.float().view(-1),
                             expand_vec]).view(1, -1)
            # print(cks)

            knowledge = self.knowledge_model(k)
            score, _ = self.seq_model(cks, knowledge, score, hidden)
            score = score.flatten()

        expand_vec = k.float().view(-1) * score
        # print(expand_vec)
        cks = torch.cat([k.float().view(-1),
                         expand_vec]).view(1, -1)
        # print(cks)

        knowledge = self.knowledge_model(k)
        s, h = self.seq_model(cks, knowledge, score, hidden)
        return s, h


class KnowledgeModel(nn.Module):
    """
    Transform Knowledge index to knowledge embedding
    """

    def __init__(self, know_len, know_emb_size):
        super(KnowledgeModel, self).__init__()
        self.knowledge_embedding = nn.Linear(know_len, know_emb_size)

    def forward(self, knowledge):
        return self.knowledge_embedding(knowledge.float().view(1, -1))


class DKVMNSeqModel(nn.Module):
    """
    DKVMN seq model
    """

    def __init__(self, know_emb_size, know_length, kcnt, seq_hidden_size, value_size):
        super(DKVMNSeqModel, self).__init__()
        self.know_emb_size = know_emb_size
        self.know_length = know_length
        self.seq_hidden_size = seq_hidden_size
        # self.erase_size = erase_size
        # self.add_size = add_size
        self.value_size = value_size

        # knowledge memory matrix
        self.knowledge_memory = nn.Parameter(torch.zeros(self.know_length, self.know_emb_size))
        self.knowledge_memory.data.uniform_(-1, 1)

        # read process embedding module
        self.ft_embedding = nn.Linear(self.seq_hidden_size + self.know_emb_size, 50)
        self.score_layer = nn.Linear(50, 1)

        # write process embedding module
        # erase_size = add_size = seq_hidden_size
        self.cks_embedding = nn.Linear(kcnt * 2, self.value_size)
        self.erase_embedding = nn.Linear(self.value_size, self.seq_hidden_size)
        self.add_embedding = nn.Linear(self.value_size, self.seq_hidden_size)

        # the first student state
        self.h_initial = nn.Parameter(torch.zeros(know_length, seq_hidden_size))
        self.h_initial.data.uniform_(-1, 1)

    def forward(self, cks, kn, s, h):
        if h is None:
            h = self.h_initial.view(self.know_length * self.seq_hidden_size)

        # calculate alpha weights of knowledges using dot product
        alpha = torch.mm(self.knowledge_memory, kn.view(-1, 1)).view(-1)
        alpha = nn.functional.softmax(alpha.view(1, -1), dim=-1)

        # read process
        rt = torch.mm(alpha, h.view(self.know_length, self.seq_hidden_size)).view(-1)
        com_r_k = torch.cat([rt, kn.view(-1)]).view(1, -1)
        # print(com_r_k.size())
        ft = torch.tanh(self.ft_embedding(com_r_k))
        predict_score = torch.sigmoid(self.score_layer(ft))
        # predict_score = self.score_layer(ft)

        # write process
        vt = self.cks_embedding(cks)
        et = torch.sigmoid(self.erase_embedding(vt))
        at = torch.tanh(self.add_embedding(vt))
        ht = h * (1 - (alpha.view(-1, 1) * et).view(-1))
        h = ht + (alpha.view(-1, 1) * at).view(-1)
        return predict_score.view(1), h


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
        bs = q.batch_sizes[0]
        emb = self.embedding(q.data)
        emb = PackedSequence(emb, q.batch_sizes)

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

    def __init__(self,
                 n_knowledge,
                 seq_hidden_size,
                 n_layers=1):
        super(DKTNet, self).__init__()
        self.n_knwoledge = n_knowledge
        self.seq_hidden_size = seq_hidden_size
        self.n_layers = n_layers
        self.rnn_net = nn.GRU(n_knowledge * 2, seq_hidden_size, 1)
        self.score_net = nn.Linear(n_knowledge + seq_hidden_size, 1)

    def forward(self, knowledge_v, score, h):
        if h is None:
            h = self.default_hidden()

        knowledge_v = knowledge_v.type_as(h)
        pscore = self.score_net(torch.cat([h.view(-1), knowledge_v.view(-1)]))

        if score is None:
            score = pscore.flatten()

        x = torch.cat([knowledge_v.view(-1),
                       (knowledge_v * (score > 0.5).type_as(knowledge_v).
                        expand_as(knowledge_v).type_as(knowledge_v)).view(-1)])
        _, h = self.rnn_net(x.view(1, 1, -1), h)
        return pscore.view(1), h

    def default_hidden(self):
        return torch.zeros(1, 1, self.seq_hidden_size)
