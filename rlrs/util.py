import logging
import signal


def critical(f):
    it = iter(f)
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
