import fret


@fret.configurable
class Trainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def load_env(self, path=None):
        logger = self.ws.logger('Trainer.load_env')
        logger.debug('func: <trainer.load_env>, path=%s', path)

    def train(self, args):
        logger = self.ws.logger('Trainer.train')
        logger.debug('func: <trainer.train>, args: %s', args)
