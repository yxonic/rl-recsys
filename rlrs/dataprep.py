import csv
import logging
import fret
import numpy as np
import torchtext as tt

logger = logging.getLogger(__name__)


class Cutter:
    def __init__(self, words, max_len=None, split=' '):
        self.words = words
        self.max_len = max_len
        self.split = split

    def __call__(self, s):
        words = s.split(self.split)[:self.max_len] \
            if self.max_len else s.split()
        return [self.words.get(w) or 0 for w in words]


class Vocab:
    def __init__(self, vocab_list):
        if isinstance(vocab_list, str):
            self.itos = open(vocab_list).read().strip().split('\n')
        else:
            self.itos = vocab_list
        self.stoi = {k: i for i, k in enumerate(self.itos)}
        self.stoi['<unk>'] = -1

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.itos[item]
        else:
            return self.stoi[item]

    def get(self, item):
        return self.stoi[item] if item in self.stoi else None

    def __len__(self):
        return len(self.itos)


class Questions:
    def __init__(self, dataset, maxlen=400):
        cfg = fret.app['datasets'][dataset]
        self._word = Vocab(cfg['word_list'])
        self._know = Vocab(cfg['knowledge_list'])
        self.n_words = len(self._word)
        self.n_knowledge = len(self._know)

        text_field = tt.data.Field(
            tokenize=Cutter(self._word, maxlen),
            use_vocab=False)
        self._ques_text = tt.data.TabularDataset(
            cfg['question_text_file'],
            format='tsv',
            fields=[('id', tt.data.Field(sequential=False)),
                    ('content', text_field)],
            skip_header=True,
            csv_reader_params={'quoting': csv.QUOTE_NONE})
        self._ques_text_ind = {item.id: i
                               for i, item in enumerate(self._ques_text)}

        knowledge_field = tt.data.Field(
            tokenize=Cutter(self._know, split=','),
            use_vocab=False)
        self._ques_know = tt.data.TabularDataset(
            cfg['question_knowledge_file'],
            format='tsv',
            fields=[('id', tt.data.Field(sequential=False)),
                    ('knowledge', knowledge_field)],
            skip_header=True,
            csv_reader_params={'quoting': csv.QUOTE_NONE})
        self._ques_know = {item.id: item.knowledge for item in self._ques_know}

        self._ques_diff = {}
        diff_f = open(cfg['question_difficulty_file'])
        next(diff_f)
        for line in diff_f:
            qid, diff = line.strip().split('\t')
            diff = float(diff)
            self._ques_diff[qid] = diff

        self._ques_set = set(self._ques_text_ind) & \
                         set(self._ques_know) & set(self._ques_diff)
        self.vocab = Vocab(list(sorted(self._ques_set)))
        self.stoi = self.vocab.stoi
        self.itos = self.vocab.itos
        self.n_questions = len(self.vocab)

    def __getitem__(self, index):
        if isinstance(index, int):
            qid = self.vocab[index]
        else:
            qid = index
        if qid in self._ques_set:
            know = np.zeros((self.n_knowledge,))
            for k in self._ques_know[qid]:
                know[k] = 1

            return {
                'id': qid,
                'text': self._ques_text[self._ques_text_ind[qid]].content,
                'knowledge': know,
                'difficulty': self._ques_diff[qid]
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
    return embs


class QidField:
    def __init__(self, set):
        self._set = set

    def get(self, item):
        qid = item.split(',')[0]
        if qid in self._set:
            return qid
        else:
            return '<unk>'


class ScoreField:
    def get(self, item):
        return float(item.split(',')[1])


def load_record(rec_file, q_field):
    question = tt.data.Field(tokenize=Cutter(QidField(q_field.stoi)))
    question.vocab = q_field
    score = tt.data.Field(tokenize=Cutter(ScoreField()),
                          use_vocab=False)

    fields = {'question': ('question', question), 'score': ('score', score)}
    reader = csv.reader(open(rec_file), quoting=csv.QUOTE_NONE,
                        delimiter='\t')
    field_to_index = {'question': 0, 'score': 0}
    examples = [tt.data.Example.fromCSV(line, fields, field_to_index)
                for line in reader]

    field_list = []
    for field in fields.values():
        if isinstance(field, list):
            field_list.extend(field)
        else:
            field_list.append(field)
    field_list = field_list

    records = tt.data.Dataset(examples, field_list)
    return records
