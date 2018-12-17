import fret
from .dataprep import load_record


@fret.command
def train_env(ws, n_epochs=10):
    logger = ws.logger('train')
    env = ws.build_module('env')
    logger.info("[%s] %s", ws, env)

    # TODO: load record and train model
    records = load_record(fret.app['datasets'][env.dataset]['record_file'])
    env.train(records, n_epochs)


@fret.command
def train_agent(ws, n_epochs=10):
    logger = ws.logger('train')
    trainer = ws.build_module()
    logger.info("[%s] %s", ws, trainer)

    # TODO: train agent on trained env
    trainer.load_env(ws.checkpoint_path / 'env.last.pt')
    trainer.train(n_epochs)
