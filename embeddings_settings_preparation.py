import logging
import os
from os.path import join, isfile
from os import listdir
import shutil

from nltk import word_tokenize

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

from utils import load_embeddings
from downstream_task.sentiment_classification.train import parse_pickle_file
from downstream_task.part_of_speech.train import parse_pos_file


def parse_conll_file(filename):
    entities = dict()
    tokens = dict()
    with open(filename) as fhandler:
        for line in fhandler:
            if not ((line.startswith('-DOCSTART-') or line.startswith('\n'))):
                token, _, _, e = line[:-1].split(' ')
                if token not in tokens:
                    tokens[token.lower()] = 1
                else:
                    tokens[token.lower()] += 1
                if e is not 'O':
                    if token not in entities:
                        entities[token] = 1
                    else:
                        entities[token] += 1
    return entities, tokens


def create_necessary_dirs(base_path):
    settings = ['setting1', 'setting2']
    for s in settings:
        os.makedirs(join(base_path, s, 'glove', 'train'), exist_ok=True)
        os.makedirs(join(base_path, s, 'glove', 'mimick_predictions'), exist_ok=True)
        os.makedirs(join(base_path, s, 'glove', 'comick_predictions'), exist_ok=True)
        os.makedirs(join(base_path, s, 'glove', 'test'), exist_ok=True)
        os.makedirs(join(base_path, s, 'w2v', 'train'), exist_ok=True)
        os.makedirs(join(base_path, s, 'w2v', 'mimick_predictions'), exist_ok=True)
        os.makedirs(join(base_path, s, 'w2v', 'comick_predictions'), exist_ok=True)
        os.makedirs(join(base_path, s, 'w2v', 'test'), exist_ok=True)


def create_embedding_file(original_embedding_path, target_embedding_path, tokens):
    embeddings = load_embeddings(original_embedding_path)
    with open(target_embedding_path, 'w') as fh:
        for token in tokens:
            if token in embeddings:
                vec = ' '.join([str(i) for i in embeddings[token]])
                fh.write("{} {}\n".format(token, vec))
            elif token.capitalize() in embeddings:
                vec = ' '.join([str(i) for i in embeddings[token.capitalize()]])
                fh.write("{} {}\n".format(token.capitalize(), vec))
            elif token.upper() in embeddings:
                vec = ' '.join([str(i) for i in embeddings[token.upper()]])
                fh.write("{} {}\n".format(token.upper(), vec))
            elif token.lower() in embeddings:
                vec = ' '.join([str(i) for i in embeddings[token.lower()]])
                fh.write("{} {}\n".format(token.lower(), vec))


def create_oov_file(target_vocab_path, tokens):
    with open(target_vocab_path, 'w') as fh:
        for token in tokens:
            fh.write("{}\n".format(token))


def filter_validation_and_test_vocab(train_tokens, validation_tokens, test_tokens):
    validation_vocab = set()
    for word in validation_tokens.keys():
        if word not in train_tokens.keys():
            validation_vocab.add(word)
    for word in test_tokens.keys():
        if word not in train_tokens.keys():
            validation_vocab.add(word)
    return validation_vocab


def find_every_words_not_in_embeddings(embedding_path, vocab):
    oov = set()
    embeddings = load_embeddings(embedding_path)
    for token in vocab:
        if token not in embeddings and token.capitalize() not in embeddings and token.upper() not in embeddings and token.lower() not in embeddings:
            oov.add(token)
    return oov


def prepare_embeddings(dataset_name, train_tokens, validation_tokens, test_tokens):
    base_path = "data/" + dataset_name + '_embeddings_settings'
    create_necessary_dirs(base_path)

    oov_setting2 = filter_validation_and_test_vocab(
        train_tokens,
        validation_tokens,
        test_tokens
    )

    # GloVe Embeddings processing
    # We need to do this only once
    oov_setting1 = find_every_words_not_in_embeddings(
        './data/glove_embeddings/glove.6B.50d.txt',
        train_tokens.keys() | validation_tokens.keys() | test_tokens.keys()
    )
    create_oov_file(join(base_path, 'setting1', 'glove', 'oov.txt'), oov_setting1)
    create_oov_file(join(base_path, 'setting2', 'glove', 'oov.txt'), oov_setting1 | oov_setting2)
    glove_dims = [50, 100, 200, 300]
    for dim in glove_dims:
        original_embedding_path = './data/glove_embeddings/glove.6B.{}d.txt'.format(dim)
        target_original_embedding_path = join(base_path, 'setting1', 'glove', 'train', 'glove.6B.{}d.txt').format(dim)
        target_training_embedding_path = join(base_path, 'setting2', 'glove', 'train', 'glove.6B.{}d.txt').format(dim)
        target_test_embedding_path = join(base_path, 'setting2', 'glove', 'test', 'glove.6B.{}d.txt').format(dim)
        shutil.copy(original_embedding_path, target_original_embedding_path)
        logging.info("Creating train embeddings from {}".format(original_embedding_path))
        create_embedding_file(original_embedding_path, target_training_embedding_path, train_tokens.keys())
        create_embedding_file(original_embedding_path, target_test_embedding_path, oov_setting2)
        logging.info("Done")

    # Word2Vec embeddings processing
    original_embedding_path = './data/word2vec_embeddings/wiki-news-300d-1M-subword.vec'

    oov_setting1 = find_every_words_not_in_embeddings(
        original_embedding_path,
        train_tokens.keys() | validation_tokens.keys() | test_tokens.keys()
    )
    create_oov_file(join(base_path, 'setting1', 'w2v', 'oov.txt'), oov_setting1)
    create_oov_file(join(base_path, 'setting2', 'w2v', 'oov.txt'), oov_setting1 | oov_setting2)

    target_original_embedding_path = join(base_path, 'setting1', 'w2v', 'train', 'wiki-news-300d-1M-subword.vec')
    target_training_embedding_path = join(base_path, 'setting2', 'w2v', 'train', 'wiki-news-300d-1M-subword.vec')
    target_test_embedding_path = join(base_path, 'setting2', 'w2v', 'test', 'wiki-news-300d-1M-subword.vec')
    shutil.copy(original_embedding_path, target_original_embedding_path)
    logging.info("Creating train embeddings from {}".format(original_embedding_path))
    create_embedding_file(original_embedding_path, target_training_embedding_path, train_tokens.keys())
    create_embedding_file(original_embedding_path, target_test_embedding_path, oov_setting2)
    logging.info("Done")


def parse_semeval_files(folder):
    distinct_tokens = dict()
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f)) and '.txt' in f]
    for f in onlyfiles:
        with open(join(folder, f)) as fh:
            for line in fh:
                tokens = word_tokenize(line)
                for token in tokens:
                    if token not in distinct_tokens:
                        distinct_tokens[token] = 1
                    else:
                        distinct_tokens[token] += 1
    return distinct_tokens


def parse_sentiment_analysis_file(filename):
    distinct_tokens = dict()
    sentences, _ = parse_pickle_file(filename)
    for sentence in sentences:
        for word in sentence:
            if word not in distinct_tokens:
                distinct_tokens[word.lower()] = 1
            else:
                distinct_tokens[word.lower()] += 1
    return distinct_tokens

def parse_part_of_speech_file(filename):
    distinct_tokens = dict()
    sentences, _ = parse_pos_file(filename)
    for sentence in sentences:
        for word in sentence:
            if word not in distinct_tokens:
                distinct_tokens[word.lower()] = 1
            else:
                distinct_tokens[word.lower()] += 1
    return distinct_tokens



if __name__ == '__main__':
    train_entities, train_tokens = parse_conll_file('./conll/train.txt')
    validation_entities, validation_tokens = parse_conll_file('./conll/valid.txt')
    test_entities, test_tokens = parse_conll_file('./conll/test.txt')
    prepare_embeddings('conll', train_tokens, validation_tokens, test_tokens)

    # train_tokens = parse_semeval_files('./scienceie/scienceie2017_train/train2')
    # validation_tokens = parse_semeval_files('./scienceie/scienceie2017_dev/dev')
    # test_tokens = parse_semeval_files('./scienceie/semeval_articles_test')
    # prepare_embeddings('semeval', train_tokens, validation_tokens, test_tokens)

    train_tokens = parse_sentiment_analysis_file('./data/sentiment/train.pickle')
    validation_tokens = parse_sentiment_analysis_file('./data/sentiment/dev.pickle')
    test_tokens = parse_sentiment_analysis_file('./data/sentiment/test.pickle')
    prepare_embeddings('sentiment', train_tokens, validation_tokens, test_tokens)

    train_tokens = parse_part_of_speech_file('./data/conll/train.txt')
    validation_tokens = parse_part_of_speech_file('./data/conll/valid.txt')
    test_tokens = parse_part_of_speech_file('./data/conll/test.txt')
    prepare_embeddings('pos', train_tokens, validation_tokens, test_tokens)

