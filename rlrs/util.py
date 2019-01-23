import itertools
import logging
import signal

import torch
from torch.nn.utils.rnn import pack_sequence

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