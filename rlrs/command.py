from itertools import count

import fret
import numpy as np
import torch

from .environment import DeepSPEnv, OffPolicyEnv
from .agent import DQN
from .dataprep import load_record
from .util import ndcg_at_k, Accumulator


# noinspection PyUnusedLocal
@fret.command
def train_deep_env(ws, n_epochs=3, log_every=32, save_every=2500,
                   restart=False, train=False):
    logger = ws.logger('train_env')
    logger.info('building env and loading questions')
    env: DeepSPEnv = ws.build_module('env' if train else 'test_env')
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
def test(ws):
    from .environment import OffPolicyEnv
    env: OffPolicyEnv = ws.build_module('env')
    rec_file = fret.app['datasets'][env.dataset]['record_file']
    records = load_record(rec_file, env.questions)
    env.set_records(records)
    env.reset()
    while True:
        a = env.random_action()
        o, r, done, _ = env.step(a)
        print(env.questions.itos[a], o[0], r, sep='\t')
        if done:
            break


@fret.command
def eval_point(ws: fret.Workspace, tag):
    env: OffPolicyEnv = ws.build_module('env')
    questions = env.questions
    rec_file = fret.app['datasets'][env.dataset]['record_file']
    records = load_record(rec_file, questions)
    rec_file = fret.app['datasets'][env.dataset]['test_record_file']
    test_records = load_record(rec_file, questions)

    agent: DQN = ws.build_module('agent', questions=env.questions)
    if tag:
        agent.load_model(tag)

    torch.set_grad_enabled(False)
    ndcgs = Accumulator()
    for lineno, (r, r_) in enumerate(zip(records, test_records)):
        if len(r_.score) < 10:
            continue
        scores = {}
        for q, s in zip(r.question, r.score):
            scores[q] = s
        if len(scores) < 10:
            continue

        action_mask = torch.zeros(env.n_questions).byte()
        action_mask[[env.questions.stoi[x] for x in r.question]] = 1

        state = agent.reset()
        for _ in r.question:
            a = agent.select_action(state, action_mask)
            action_mask[a] = 0
            score = np.array([scores[env.questions.itos[a]]])
            state = agent.step(a, score)

        action_mask = torch.zeros(env.n_questions).byte()
        action_mask[[env.questions.stoi[x] for x in r_.question]] = 1
        score, ind = agent.get_action_values(state, action_mask)
        M, m = score.max(), score.min()
        score = (score - m) / (M - m)
        scores = {}
        for q, s in zip(r_.question, r_.score):
            scores[q] = s
        rv = [(ind[i].item(),
               score[i][0].item(),
               scores[env.questions.itos[ind[i].item()]])
              for i in range(ind.size(0))]
        rv.sort(key=lambda x: x[1], reverse=True)
        r = [1 - x[2] for x in rv]
        ndcgs += ndcg_at_k(r, 10)

        if lineno % 100 == 0:
            print(ndcgs.mean())
    print(ndcgs.mean())


@fret.command
def eval_offline(ws: fret.Workspace, tag):
    env: OffPolicyEnv = ws.build_module('env')
    rec_file = fret.app['datasets'][env.dataset]['offline_test_record_file']
    records = load_record(rec_file, env.questions)

    agent: DQN = ws.build_module('agent', questions=env.questions)
    if tag:
        agent.load_model(tag)

    torch.set_grad_enabled(False)
    for r in records:
        scores = {}
        for q, s in zip(r.question, r.score):
            scores[q] = s
        if len(scores) < 10:
            continue

        action_mask = torch.zeros(env.n_questions).byte()
        action_mask[[env.questions.stoi[x] for x in r.question]] = 1

        state = agent.reset()
        for i in count():
            a = agent.select_action(state, action_mask)
            action_mask[a] = 0
            score = np.array([scores[env.questions.itos[a]]])
            state = agent.step(a, score)

            if i > len(scores) * 0.6:
                break

        score, ind = agent.get_action_values(state, action_mask)
        rv = [(score[i][0].item(), scores[env.questions.itos[ind[i].item()]])
              for i in range(ind.size(0))]
        y_pred, y_true = zip(*rv)


@fret.command
def eval_online(ws: fret.Workspace, env_tag, agent_tag):
    env: DeepSPEnv = ws.build_module('test_env')
    if env_tag:
        env.load_model(env_tag)
