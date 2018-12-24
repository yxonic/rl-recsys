import fret
from .environment import SPEnv
from .dataprep import load_record


# noinspection PyUnusedLocal
@fret.command
def train_env(ws, n_epochs=3, log_every=32, save_every=2500, resume=True):
    logger = ws.logger('train_env')
    logger.info('building env and loading questions')
    env: SPEnv = ws.build_module('env')
    logger.info("[%s] %s, %s", ws, env, train_env.args)

    rec_file = fret.app['datasets'][env.dataset]['record_file']
    logger.info("loading records: %s", rec_file)
    records = load_record(rec_file, env.questions)
    env.train(records, train_env.args)


@fret.command
def train_agent(ws, n_epochs=10):
    logger = ws.logger('train')
    trainer = ws.build_module()
    logger.info("[%s] %s", ws, trainer)

    # TODO: train agent on trained env
    trainer.load_env(ws.checkpoint_path / 'env.last.pt')
    trainer.train(n_epochs)
