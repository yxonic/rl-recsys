from itertools import count

import fret
import numpy as np
import torch

from .environment import DeepSPEnv, _StuEnv
from .agent import Agent
from .dataprep import load_record
from .util import ndcg_at_k, Accumulator, critical


# noinspection PyUnusedLocal
@fret.command
def train_deep_env(ws, n_epochs=3, log_every=32, save_every=2500,
                   restart=False, env=None):
    logger = ws.logger('train_env')
    logger.info('building env and loading questions')
    env: DeepSPEnv = ws.build_module(env or 'env')
    logger.info("[%s] %s, %s", ws, env, train_deep_env.args)

    rec_file = fret.app['datasets'][env.dataset]['record_file']
    logger.info('loading records: %s', rec_file)
    records = load_record(rec_file, env.questions)
    env.set_records(records)
    env.train(train_deep_env.args)


# noinspection PyUnusedLocal
@fret.command
def train_agent(ws, checkpoint=None, n_episodes=10000, batch_size=16,
                restart=False, save_every=100, log_every=32):
    logger = ws.logger('train')

    logger.info('building trainer and loading questions')
    trainer = ws.build_module()
    if checkpoint:
        trainer.env.load_model(checkpoint)

    logger.info("[%s] %s", ws, trainer)

    trainer.train(train_agent.args)


@fret.command
def eval_offline(ws: fret.Workspace, tag, seq=False):
    logger = ws.logger('eval_point')

    env: _StuEnv = ws.build_module('env')
    questions = env.questions
    rec_file = fret.app['datasets'][env.dataset]['record_file']
    records = load_record(rec_file, questions)
    rec_file = fret.app['datasets'][env.dataset]['test_record_file']
    test_records = load_record(rec_file, questions)

    agent: Agent = ws.build_module('agent', questions=env.questions)
    if tag:
        agent.load_model(tag)

    torch.set_grad_enabled(False)
    ndcgs = Accumulator()
    results = []
    try:
        for lineno, (r, r_) in critical(enumerate(zip(records, test_records))):
            if len(r_.score) < 20:
                continue

            id_score = {}
            for q, s in zip(r.question, r.score):
                id_score[q] = s
            for q, s in zip(r_.question, r_.score):
                id_score[q] = s

            action_mask = torch.zeros(env.n_questions).byte()
            action_mask[[env.questions.stoi[x] for x in r.question]] = 1

            state = agent.reset()
            for _ in r.question:
                a = agent.select_action(state, action_mask)[0]
                action_mask[a] = 0
                score = np.array([id_score[env.questions.itos[a]]])
                state = agent.step(a, score)

            if seq:
                score = []
                ind = []
                action_mask = torch.zeros(env.n_questions).byte()
                action_mask[[env.questions.stoi[x] for x in r_.question]] = 1
                for _ in r_.question:
                    a, v = agent.select_action(state, action_mask)
                    action_mask[a] = 0
                    score.append(v)
                    ind.append(a)
                    state = agent.step(a, np.asarray([s]))
                score = np.asarray(score)
            else:
                action_mask = torch.zeros(env.n_questions).byte()
                action_mask[[env.questions.stoi[x] for x in r_.question]] = 1
                score, ind = agent.get_action_values(state, action_mask)

            M, m = score.max(), score.min()
            pred = (score - m) / (M - m)
            true = [id_score[env.questions.itos[i]] for i in ind]
            ids = [env.questions.itos[i] for i in ind]

            results.append(list(zip(ids, true, pred)))

            rv = [(true[i], pred[i]) for i in range(len(ind))]
            ndcgs += ndcg_at_k(rv, 10)

            if lineno % 100 == 0:
                logger.info('NDCG: %.4f, MAP: %.4f', ndcgs.mean(), 0)
    except KeyboardInterrupt:
        pass

    logger.info('[FINAL] NDCG: %.4f, MAP: %.4f', ndcgs.mean(), 0)
    result_path = ws.result_path / ('%s.seq.txt' % tag if seq
                                    else '%s.txt' % tag )
    with result_path.open('w') as f:
        for line in results:
            print(' '.join('%s,%f,%f' % rec for rec in line), file=f)


@fret.command
def eval_online(ws: fret.Workspace, env_tag, agent_tag):
    env: DeepSPEnv = ws.build_module('test_env')
    if env_tag:
        env.load_model(env_tag)
