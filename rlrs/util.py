import itertools
import logging
import signal

import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence

default_sigint_handler = signal.getsignal(signal.SIGINT)


def critical(f=None):
    if f is not None:
        it = iter(f)
    else:
        it = itertools.count()
    signal_received = ()

    def handler(sig, frame):
        nonlocal signal_received
        signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    while True:
        try:
            signal.signal(signal.SIGINT, handler)
            yield next(it)
            signal.signal(signal.SIGINT, default_sigint_handler)
            if signal_received:
                default_sigint_handler(*signal_received)
        except StopIteration:
            break


def make_batch(*inputs, seq=None):
    if not seq:
        return [torch.cat(v, dim=0) for v in inputs]
    data = list(zip(*inputs))
    data.sort(key=lambda x: x[seq[0]].size(0), reverse=True)
    outputs = []
    for i, v in enumerate(zip(*data)):
        if i not in seq:
            outputs.append(torch.cat(v, dim=0))
        else:
            outputs.append(pack_sequence(v))
    return outputs


class Accumulator:
    def __init__(self, init=0):
        self.total = init
        self.cnt = 0

    def __iadd__(self, other):
        self.total += other
        self.cnt += 1
        return self

    def mean(self):
        return self.total / self.cnt if self.cnt > 0 else 0

    def sum(self):
        return self.total

    def __int__(self):
        return int(self.total)

    def __float__(self):
        return float(self.total)

    def reset(self):
        self.total = 0
        self.cnt = 0

    def __getstate__(self):
        return {'total': self.total, 'cnt': self.cnt}

    def __setstate__(self, state):
        self.__dict__.update(state)


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(rv, k):
    rv.sort(key=lambda x: x[1], reverse=True)
    r = [1 - x[0] for x in rv]
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, 0)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, 0) / dcg_max


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


class SeqBatch:
    def __init__(self, seqs, dtype=None, device=None):
        self.dtype = dtype
        self.device = device
        self.seqs = seqs
        self.lens = [len(x) for x in seqs]

        self.ind = argsort(self.lens)[::-1]
        self.inv = argsort(self.ind)
        self.lens.sort(reverse=True)
        self._prefix = [0]
        self._index = {}
        c = 0
        for i in range(self.lens[0]):
            for j in range(len(self.lens)):
                if self.lens[j] <= i:
                    break
                self._index[i, j] = c
                c += 1

    def packed(self):
        ind = torch.tensor(self.ind, dtype=torch.long, device=self.device)
        padded = self.padded()[0].index_select(1, ind)
        return pack_padded_sequence(padded, self.lens)

    def padded(self, max_len=None, batch_first=False):
        seqs = [torch.tensor(s, dtype=self.dtype, device=self.device)
                if not isinstance(s, torch.Tensor) else s
                for s in self.seqs]
        if max_len is None:
            max_len = self.lens[0]
        seqs = [s[:max_len] for s in seqs]
        mask = [[1] * len(s) + [0] * (max_len - len(s)) for s in seqs]

        max_size = seqs[0].size()
        trailing_dims = max_size[1:]
        if batch_first:
            out_dims = (len(seqs), max_len) + trailing_dims
        else:
            out_dims = (max_len, len(seqs)) + trailing_dims

        padded = seqs[0].new(*out_dims).fill_(0)
        for i, tensor in enumerate(seqs):
            length = tensor.size(0)
            # use index notation to prevent duplicate references to the tensor
            if batch_first:
                padded[i, :length, ...] = tensor
            else:
                padded[:length, i, ...] = tensor
        return padded, torch.tensor(mask).byte().to(self.device)

    def index(self, item):
        return self._index[item[0], self.inv[item[1]]]

    def invert(self, batch, dim=0):
        return batch.index_select(dim, torch.tensor(self.inv))
