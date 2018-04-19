import logging

from downstream_task.semeval.train import parse_semeval_file
from downstream_task.sentiment_classification.train import parse_pickle_file
from downstream_task.newsgroup_classification.train import parse_20newsgroup_file
from utils import load_embeddings, parse_conll_file, load_vocab


class DataLoader:
    def __init__(self, debug_mode):
        self.test_vocabs = set()
        self.test_embeddings = dict()
        self.embeddings = dict()
        self.test_sentences = []
        self.valid_sentences = []
        self.train_sentences = []
        self.debug_mode = debug_mode
        logging.info("Loading data in debug mode: {}".format(debug_mode))

    def __filter_sentences(self, attr, num):
        if self.debug_mode:
            return self.__getattribute__(attr)[:num]
        else:
            return self.__getattribute__(attr)

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


class CoNLLDataLoader(DataLoader):
    def __init__(self, debug_mode, embedding_dimension):
        super().__init__(debug_mode)
        path_embeddings = './data/conll_embeddings_settings/setting1/glove/train/glove.6B.{}d.txt'.format(
            embedding_dimension)
        self.embeddings = load_embeddings(path_embeddings)
        self.test_vocabs = load_vocab('./data/conll_embeddings_settings/setting2/glove/oov.txt')
        self.test_embeddings = {word: self.embeddings[word] for word in self.test_vocabs if word in self.embeddings}
        self.train_sentences = parse_conll_file('./data/conll/train.txt')
        self.valid_sentences = parse_conll_file('./data/conll/valid.txt')
        self.test_sentences = parse_conll_file('./data/conll/test.txt')
        logging.debug('Loading {}d embeddings from : {}'.format(embedding_dimension, path_embeddings))


class SentimentDataLoader(DataLoader):
    def __init__(self, debug_mode, embedding_dimension):
        super().__init__(debug_mode)
        path_embeddings = './data/sentiment_embeddings_settings/setting1/glove/train/glove.6B.{}d.txt'.format(
            embedding_dimension)
        self.embeddings = load_embeddings(path_embeddings)
        self.test_vocabs = load_vocab('./data/sentiment_embeddings_settings/setting2/glove/oov.txt')
        self.test_embeddings = {word: self.embeddings[word] for word in self.test_vocabs if word in self.embeddings}
        self.train_sentences, _ = parse_pickle_file('./data/sentiment/train.pickle')
        self.valid_sentences, _ = parse_pickle_file('./data/sentiment/dev.pickle')
        self.test_sentences, _ = parse_pickle_file('./data/sentiment/test.pickle')
        logging.debug('Loading {}d embeddings from : {}'.format(embedding_dimension, path_embeddings))


class SemEvalDataLoader(DataLoader):
    def __init__(self, debug_mode, embedding_dimension):
        super().__init__(debug_mode)
        path_embeddings = './data/semeval_embeddings_settings/setting1/glove/train/glove.6B.{}d.txt'.format(
            embedding_dimension)
        self.embeddings = load_embeddings(path_embeddings)
        self.test_vocabs = load_vocab('./data/semeval_embeddings_settings/setting2/glove/oov.txt')
        self.test_embeddings = {word: self.embeddings[word] for word in self.test_vocabs if word in self.embeddings}
        self.train_sentences, _ = parse_semeval_file('./data/scienceie/train_spacy.txt')
        self.valid_sentences, _ = parse_semeval_file('./data/scienceie/valid_spacy.txt')
        self.test_sentences, _ = parse_semeval_file('./data/scienceie/test_spacy.txt')
        logging.debug('Loading {}d embeddings from : {}'.format(embedding_dimension, path_embeddings))


class NewsGroupDataLoader(DataLoader):
    def __init__(self, debug_mode, embedding_dimension):
        super().__init__(debug_mode)
        path_embeddings = './data/newsgroup_embeddings_settings/setting1/glove/train/glove.6B.{}d.txt'.format(
            embedding_dimension)
        self.embeddings = load_embeddings(path_embeddings)
        self.test_vocabs = load_vocab('./data/newsgroup_embeddings_settings/setting2/glove/oov.txt')
        self.test_embeddings = {word: self.embeddings[word] for word in self.test_vocabs if word in self.embeddings}
        self.train_sentences, _ = parse_20newsgroup_file('./data/20newsgroup/train.pickle')
        self.valid_sentences, _ = parse_20newsgroup_file('./data/20newsgroup/dev.pickle')
        self.test_sentences, _ = parse_20newsgroup_file('./data/20newsgroup/test.pickle')
        logging.debug('Loading {}d embeddings from : {}'.format(embedding_dimension, path_embeddings))