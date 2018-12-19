import logging
import numpy as np

logger = logging.getLogger(__name__)


def load_question(ques_file):
    # TODO: read questions as a list, along with id->index map
    logger.debug('func: <dataprep.load_question>')
    return [], {}


def load_knowledge(know_file):
    # TODO: read knowledge as a list, along with know->index map
    logger.debug('func: <dataprep.load_knowledge>')
    return [], {}


def load_record(rec_file):
    # TODO: read record as list of sequences
    logger.debug('func: <dataprep.load_record>')
    return []


def load_embedding(emb_file):
    f = open(emb_file, 'r', encoding='utf-8')
    wcnt, emb_size = next(f).strip().split(' ')
    wcnt, emb_size = int(wcnt), int(emb_size)

    words = []
    embs = []
    for line in f:
        fields = line.strip().split(' ')
        word = fields[0]
        emb = np.array([float(x) for x in fields[1:]])
        words.append(word)
        embs.append(emb)

    embs = np.asarray(emb)
    return wcnt, emb_size, words, embs