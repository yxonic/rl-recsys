import fret
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataprep import load_embedding


#####
# Deep Score Prediction Models
###

@fret.configurable
class EERNN(nn.Module):
    def __init__(self,
                 emb_file=('data/emb_50.txt', 'pretrained embedding file'),
                 ques_h_size=(50, 'question embedding set'),
                 seq_h_size=(50, 'hidden size of sequence model'),
                 n_layers=(1, 'number of layers of RNN'),
                 attn_k=(10, 'top k records for attention in EERNN model')):
        super(EERNN, self).__init__()
        self.emb_file = emb_file
        wcnt, emb_size, words, embs = load_embedding(self.emb_file)
        self.wcnt = wcnt
        self.emb_size = emb_size
        self.words = words
        self.embs = embs
        self.ques_h_size = ques_h_size
        self.seq_h_size = seq_h_size
        self.n_layers = n_layers
        self.attn_k = attn_k

        self.question_net = QuesNet(wcnt, emb_size, ques_h_size, n_layers)
        self.question_net.load_emb(embs)

        self.seq_net = EERNNSeqNet(ques_h_size, seq_h_size, n_layers, attn_k)

    def forward(self, question, score, hidden=None):
        ques_h0 = None
        ques_v, ques_h = self.question_net(question.view(-1, 1), ques_h0)
        s, h = self.seq_net(ques_v[0], score, hidden)
        if hidden is None:
            hidden = ques_v, h
        else:
            questions, hs = hidden
            questions = torch.cat([questions, ques_v])
            hs = torch.cat([hs, h])
            hidden = questions, hs

        return s, hidden


class QuesNet(nn.Module):
    def __init__(self, wcnt, emb_size=100, ques_size=50, n_layers=1):
        super(QuesNet, self).__init__()
        self.wcnt = wcnt
        self.emb_size = emb_size
        self.ques_size = ques_size
        self.n_layers = n_layers

        self.embedding_net = nn.Embedding(wcnt, self.emb_size, padding_idx=0)

        self.emb_size = ques_size // 2
        self.question_net = nn.GRU(self.emb_size, self.ques_size // 2, self.n_layers,
                                   bidirectional=True)

    def forward(self, question, hidden):
        x = self.embedding_net(question)
        y, h = self.question_net(x, hidden)

        y, _ = torch.max(y, 0)
        return y, h

    def load_emb(self, emb):
        self.embedding_net.weight.data.copy_(torch.from_numpy(emb))


class EERNNSeqNet(nn.Module):
    def __init__(self,
                 ques_size=100,
                 seq_hidden_size=50,
                 n_layers=1,
                 attn_k=10
                 ):
        super(EERNNSeqNet, self).__init__()

        self.ques_size = ques_size  # exercise size
        self.seq_hidden_size = seq_hidden_size
        self.n_layers = n_layers
        self.attn_k = attn_k

        # initialize network
        self.seq_net = nn.GRU(self.ques_size * 2, self.seq_hidden_size, self.n_layers)
        self.score_net = nn.Linear(self.ques_size + self.seq_hidden_size, 1)

    def forward(self, question, score, hidden):
        questions, hs = hidden
        h = hs[-1:]

        # prediction
        alpha = torch.mm(questions, question.view(-1, 1)).view(-1)
        alpha, idx = alpha.topk(min(len(alpha), self.attn_k), sorted=False)
        alpha = F.softmax(alpha.view(-1, 1), dim=-1)

        hs = hs.view(-1, self.n_layers * self.seq_hidden_size)
        attn_h = torch.mm(alpha, torch.index_select(hs, 0, idx)).view(-1)

        pred_v = torch.cat([question, attn_h]).view(1, -1)
        pred = self.score_net(pred_v)

        if score is None:
            score = pred

        # update seq_net
        x = torch.cat([question * (score >= 0.5).type_as(question).expand_as(question),
                       question * (score < 0.5).type_as(question).expand_as(question)])

        _, h_ = self.seq_net(x.view(1, 1, -1), h)
        return pred, h_
