import csv
import logging
import fret
import numpy as np
import torchtext as tt

logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self, words, max_len=None, split=' '):
        self.words = words
        self.max_len = max_len
        self.split = split

    def __call__(self, s):
        words = s.split(self.split)[:self.max_len] \
            if self.max_len else s.split()
        return np.asarray([self.words.get(w) or 0 for w in words])


class Field:
    def __init__(self, f):
        if isinstance(f, str):
            self._list = open(f).read().strip().split('\n')
        else:
            self._list = f
        self._ind = {k: i for i, k in enumerate(self._list)}

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._list[item]
        else:
            return self._ind[item]

    def get(self, item):
        return self._ind[item] if item in self._ind else None

    def __len__(self):
        return len(self._list)


class Questions:
    def __init__(self, dataset, maxlen=400):
        cfg = fret.app['datasets'][dataset]
        self._word = Field(cfg['word_list'])
        self._know = Field(cfg['knowledge_list'])
        self.n_words = len(self._word)
        self.n_knowledge = len(self._know)

        text_field = tt.data.Field(
            tokenize=Tokenizer(self._word, maxlen),
            use_vocab=False)
        self.ques_text = tt.data.TabularDataset(
            cfg['question_text_file'],
            format='tsv',
            fields=[('id', tt.data.Field(sequential=False)),
                    ('content', text_field)],
            skip_header=True,
            csv_reader_params={'quoting': csv.QUOTE_NONE})
        self.ques_text_ind = {item.id: i
                              for i, item in enumerate(self.ques_text)}

        knowledge_field = tt.data.Field(
            tokenize=Tokenizer(self._know, split=','),
            use_vocab=False)
        self.ques_know = tt.data.TabularDataset(
            cfg['question_knowledge_file'],
            format='tsv',
            fields=[('id', tt.data.Field(sequential=False)),
                    ('knowledge', knowledge_field)],
            skip_header=True,
            csv_reader_params={'quoting': csv.QUOTE_NONE})
        self.ques_know = {item.id: item.knowledge for item in self.ques_know}

        self.ques_diff = {}
        diff_f = open(cfg['question_difficulty_file'])
        next(diff_f)
        for line in diff_f:
            qid, diff = line.strip().split('\t')
            diff = float(diff)
            self.ques_diff[qid] = diff

        self.question_set = set(self.ques_text_ind) & \
            set(self.ques_know) & set(self.ques_diff)
        self.questions = Field(list(sorted(self.question_set)))
        self.n_questions = len(self.questions)

    def __getitem__(self, index):
        if isinstance(index, int):
            qid = self.questions[index]
        else:
            qid = index
        if qid in self.question_set:
            return {
                'id': qid,
                'text': self.ques_text[self.ques_text_ind[qid]].content,
                'knowledge': self.ques_know[qid],
                'difficulty': self.ques_diff[qid]
            }
        else:
            return None

    @property
    def knowledge(self):
        return self._know

    @property
    def word(self):
        return self._word


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

    embs = np.asarray(embs)
    return wcnt, emb_size, words, embs


def load_record(rec_file):
    return []
