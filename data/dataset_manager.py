import logging

import sys
sys.path.append('..')

from downstream_task.semeval.train import parse_semeval_file
from downstream_task.sentiment_classification.train import parse_pickle_file
from downstream_task.newsgroup_classification.train import parse_20newsgroup_file
from utils import load_embeddings, load_vocab#, parse_conll_file


class DatasetManager:
    def __init__(self, cls, train_path, valid_path, test_path, debug_mode=False):
        self.oov = set()
        self.dataset_name = ''
        self.train_sentences = cls.parse_file(train_path)
        self.valid_sentences = cls.parse_file(valid_path)
        self.test_sentences = cls.parse_file(test_path)
        self.debug_mode = debug_mode
        logging.info("Loading data in debug mode: {}".format(debug_mode))

    def __filter_sentences(self, attr, num):
        if self.debug_mode:
            return self.__getattribute__(attr)[:num]
        else:
            return self.__getattribute__(attr)
    
    @classmethod
    def parse_file(cls, filepath):
        raise NotImplementedError

    @property
    def get_train_sentences(self):
        return self.__filter_sentences('train_sentences', 50)

    @property
    def get_valid_sentences(self):
        return self.__filter_sentences('valid_sentences', 20)

    @property
    def get_test_sentences(self):
        return self.__filter_sentences('test_sentences', 0)

    @property
    def get_embeddings(self):
        return self.embeddings

    @property
    def get_test_embeddings(self):
        return self.test_embeddings

    @property
    def get_test_vocab(self):
        return self.test_vocabs

    @property
    def get_oov(self):
        return self.oov


class CoNLL(DatasetManager):
    def __init__(self, debug_mode=False, relative_path='./'):
        super().__init__(cls=type(self),
                         train_path=relative_path+'./conll/train.txt',
                         valid_path=relative_path+'./conll/valid.txt',
                         test_path=relative_path+'./conll/test.txt',
                         debug_mode=debug_mode)
        self.dataset_name = 'conll'
        try:
            self.oov = load_vocab('./conll/oov.txt')
        except FileNotFoundError:
            pass
    
    @classmethod
    def parse_file(cls, filepath):
        sentences = list()
        with open(filepath) as fhandler:
            sentence = list()
            for line in fhandler:
                if not (line.startswith('-DOCSTART-') or line.startswith('\n')):
                    token, _, _, e = line[:-1].split(' ')
                    sentence.append(token.lower())
                else:
                    if len(sentence) > 0:
                        sentences.append(sentence)
                    sentence = list()
        return sentences


class Sentiment(DatasetManager):
    def __init__(self, debug_mode=False, relative_path='./'):
        super().__init__(cls=type(self),
                         train_path=relative_path+'./sentiment/train.pickle',
                         valid_path=relative_path+'./sentiment/dev.pickle',
                         test_path=relative_path+'./sentiment/test.pickle',
                         debug_mode=debug_mode)
        self.dataset_name = 'sentiment'
        try:
            self.oov = load_vocab('./sentiment/oov.txt')
        except FileNotFoundError:
            pass

    @classmethod
    def parse_file(cls, filepath):
        sentences, _ = parse_pickle_file(filepath)
        return sentences


class SemEval(DatasetManager):
    def __init__(self, debug_mode=False, relative_path='./'):
        super().__init__(cls=type(self),
                         train_path=relative_path+'./scienceie/train_spacy.txt',
                         valid_path=relative_path+'./scienceie/valid_spacy.txt',
                         test_path=relative_path+'./scienceie/test_spacy.txt',
                         debug_mode=debug_mode)
        self.dataset_name = 'scienceie'

        try:
            self.oov = load_vocab('./scienceie/oov.txt')
        except FileNotFoundError:
            pass

    @classmethod
    def parse_file(cls, filepath):
        sentences, _ = parse_semeval_file(filepath)
        return sentences


if __name__ == '__main__':
    c = CoNLL(debug_mode=True)
    print(c.get_valid_sentences)