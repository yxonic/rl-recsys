import itertools
import logging
import signal

import torch
from torch.nn.utils.rnn import pack_sequence


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
            old_handler = signal.signal(signal.SIGINT, handler)
            yield next(it)
            signal.signal(signal.SIGINT, old_handler)
            if signal_received:
                old_handler(*signal_received)
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
