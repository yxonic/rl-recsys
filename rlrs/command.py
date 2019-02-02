import random

import fret
import numpy as np
import torch
from tqdm import tqdm

from .environment import DeepSPEnv, _StuEnv
from .agent import Agent, DQN
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
def eval_offline(ws: fret.Workspace, tag, seq=False, cold_start=False,
                 rtag='test', limit=-1, quiet=None):
    if limit > 0 and quiet is None:
        quiet = True
    logger = ws.logger('eval_point')

    env: _StuEnv = ws.build_module('env')
    questions = env.questions

    rec_file = fret.app['datasets'][env.dataset]['record_file']
    records = load_record(rec_file, questions)

    rec_file = fret.app['datasets'][env.dataset]['test_record_file']
    test_records = load_record(rec_file, questions)

    class null:
        question = score = []

    if cold_start:
        for i in range(len(test_records)):
            test_records[i].question.extend(records[i].question)
            test_records[i].score.extend(records[i].score)
        records = [null] * len(records)

    agent: Agent = ws.build_module('agent', questions=env.questions)

    if isinstance(agent, DQN):
        agent.current_net.eval()
        if tag:
            agent.load_model(tag)

    torch.set_grad_enabled(False)
    ndcgs = Accumulator()
    covs = Accumulator()
    results = []
    try:
        for lineno, (r, r_) in tqdm(enumerate(zip(records, test_records)),
                                    disable=quiet):
            if lineno == limit:
                break
            if len(r_.score) < 20:
                continue
            id_score = {}
            for q, s in zip(r.question, r.score):
                id_score[q] = s
            for q, s in zip(r_.question, r_.score):
                id_score[q] = s

            action_mask = torch.zeros(env.n_questions).byte()
            action_mask[[env.questions.stoi[x] for x in r.question]] = 1

            seen = set()

            state = agent.reset()
            for _ in r.question:
                a = agent.select_action(state, action_mask)[0]
                action_mask[a] = 0
                score = np.array([id_score[env.questions.itos[a]]])
                state = agent.step(state, a, score)

            if seq:
                score = []
                ind = []
                action_mask = torch.zeros(env.n_questions).byte()
                action_mask[[env.questions.stoi[x] for x in r_.question]] = 1
                for i, _ in enumerate(r_.question):
                    if action_mask.sum().item() < 1:
                        break
                    a, v = agent.select_action(state, action_mask)
                    action_mask[a] = 0
                    score.append(v)
                    ind.append(a)
                    state = agent.step(state, a, np.asarray(
                        [id_score[env.questions.itos[a]]]))
                    if i > 20:
                        continue
                    for k in env.questions._ques_know[env.questions.itos[a]]:
                        seen.add(k)
                score = np.asarray(score)
            else:
                action_mask = torch.zeros(env.n_questions).byte()
                action_mask[[env.questions.stoi[x] for x in r_.question]] = 1
                score, ind = agent.get_action_values(state, action_mask)

            covs += len(seen) / questions.n_knowledge

            M, m = score.max(), score.min()
            if M > 1.5 or m < -0.5:
                pred = 1 - (score - m) / (M - m)
            else:
                pred = 1 - score
            true = [id_score[env.questions.itos[i]] for i in ind]
            ids = [env.questions.itos[i] for i in ind]

            results.append(list(zip(ids, true, pred)))

            rv = [(true[i], pred[i]) for i in range(len(ind))]
            ndcgs += ndcg_at_k(rv, 10)

    except KeyboardInterrupt:
        pass

    logger.info('[FINAL] NDCG: %.4f, COV: %.4f', ndcgs.mean(), covs.mean())
    result_path = ws.result_path / ('%s.%s.seq.txt' % (rtag, tag) if seq
                                    else '%s.%s.txt' % (rtag, tag))
    with result_path.open('w') as f:
        for line in results:
            print(' '.join('%s,%f,%f' % rec for rec in line), file=f)

    return results


@fret.command
def eval_online(ws: fret.Workspace, tag=None, n_students=100, seq_len=20,
                quiet=False):
    env: DeepSPEnv = ws.build_module('test_env')

    rec_file = fret.app['datasets'][env.dataset]['test_record_file']
    records = load_record(rec_file, env.questions)
    env.set_records(records)

    cp_path = 'ws/best/%s.%s.pt' % (env.questions.dataset,
                                    env.sp_model.__class__.__name__)
    env.sp_model.load_state_dict(torch.load(
        str(cp_path), map_location=lambda s, loc: s))

    agent: Agent = ws.build_module('agent', questions=env.questions)
    if isinstance(agent, DQN):
        agent.current_net.eval()
        if tag:
            agent.load_model(tag)
    env.sp_model.eval()

    returns = Accumulator()
    results = []
    for _ in tqdm(range(n_students), disable=quiet):
        action_mask = torch.zeros(env.n_questions).byte()
        if hasattr(env, 'qids'):
            action_mask[[env.questions.stoi[x] for x in env.qids]] = 1
        else:
            action_mask[random.sample(list(range(
                env.questions.n_questions)), 100)] = 1

        env.reset()
        state = agent.reset()
        R = 0
        rv = []
        for _ in range(seq_len):
            action, value = agent.select_action(state, action_mask)
            action_mask[action] = 0
            ob, reward, done, info = env.step(action)
            state = agent.step(state, action, ob)
            R += reward
            rv.append((env.questions.itos[action], ob, value))
        results.append(rv)
        returns += R
    if not quiet:
        print(returns.mean())
    return returns.mean(), results
