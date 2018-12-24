import fret
from .environment import SPEnv
from .dataprep import load_record


# noinspection PyUnusedLocal
@fret.command
def train_env(ws, n_epochs=10, resume=True):
    logger = ws.logger('train')
    env: SPEnv = ws.build_module('env')
    logger.info("[%s] %s, %s", ws, env, train_env.args)

    records = load_record(fret.app['datasets'][env.dataset]['record_file'],
                          env.questions)
    env.train(records, train_env.args)


@fret.command
def train_agent(ws, n_epochs=10):
    logger = ws.logger('train')
    trainer = ws.build_module()
    logger.info("[%s] %s", ws, trainer)

    # TODO: train agent on trained env
    trainer.load_env(ws.checkpoint_path / 'env.last.pt')
    trainer.train(n_epochs)
