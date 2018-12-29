import fret
from .environment import _SPEnv
from .dataprep import load_record


# noinspection PyUnusedLocal
@fret.command
def train_env(ws, n_epochs=3, log_every=32, save_every=2500, resume=True):
    logger = ws.logger('train_env')
    logger.info('building env and loading questions')
    env: _SPEnv = ws.build_module('env')
    logger.info("[%s] %s, %s", ws, env, train_env.args)

    rec_file = fret.app['datasets'][env.dataset]['record_file']
    logger.info('loading records: %s', rec_file)
    records = load_record(rec_file, env.questions)
    env.train(records, train_env.args)


# noinspection PyUnusedLocal
@fret.command
def train_agent(ws, checkpoint=None, n_episodes=50, batch_size=16,
                resume=True):
    logger = ws.logger('train')

    logger.info('building trainer and loading questions')
    trainer = ws.build_module()
    if checkpoint:
        trainer.env.load_model(checkpoint)

    logger.info("[%s] %s", ws, trainer)

    trainer.train(train_agent.args)
