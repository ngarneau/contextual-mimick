import logging
import torch
import random
import pickle
import numpy as np
from nltk import word_tokenize
from pytoune.framework import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger, Model
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.datasets import fetch_20newsgroups

from downstream_task.models import LSTMClassifier
from downstream_task.sequence_classification import acc, collate_examples 
from downstream_task.utils import make_vocab_and_idx
from utils import load_embeddings



def parse_20newsgroup_file(filename):
    data = pickle.load(open(filename, 'rb'))
    sentences = list()
    labels = list()
    for sentence, label in data:
        words = [w.lower() for w in word_tokenize(sentence)]
        if len(words) > 0:
            sentences.append(words)
            labels.append(int(label))
    return sentences, labels



def train(embeddings, model_name='vanilla'):
    train_sentences, train_tags = parse_20newsgroup_file('./data/20newsgroup/train.pickle')
    valid_sentences, valid_tags = parse_20newsgroup_file('./data/20newsgroup/dev.pickle')
    test_sentences, test_tags = parse_20newsgroup_file('./data/20newsgroup/test.pickle')

    labels = set(train_tags + valid_tags + test_tags)
    words_vocab, words_to_idx = make_vocab_and_idx(train_sentences + valid_sentences + test_sentences)
    tags_to_idx = {v: v for v in labels}

    train_sentences = [[words_to_idx[word] for word in sentence] for sentence in train_sentences]
    valid_sentences = [[words_to_idx[word] for word in sentence] for sentence in valid_sentences]
    test_sentences = [[words_to_idx[word] for word in sentence] for sentence in test_sentences]


    train_dataset = list(zip(train_sentences, train_tags))
    valid_dataset = list(zip(valid_sentences, valid_tags))
    test_dataset = list(zip(test_sentences, test_tags))

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_examples
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_examples
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_examples
    )

    net = LSTMClassifier(
        100,
        50,
        words_to_idx,
        len(tags_to_idx)
    )
    net.load_words_embeddings(embeddings)

    lrscheduler = ReduceLROnPlateau(patience=5)
    early_stopping = EarlyStopping(patience=10)
    model_path = './models/'
    checkpoint = ModelCheckpoint(model_path+'newsclassif_'+model_name+'.torch',
                                 save_best_only=True,
                                 restore_best=True,
                                 temporary_filename=model_path+'tmp_newsclassif_'+model_name+'.torch')
    csv_logger = CSVLogger('./train_logs/newsclassif_{}.csv'.format(model_name))
    loss = CrossEntropyLoss()
    model = Model(net, Adam(net.parameters(), lr=0.001), loss, metrics=[acc])
    model.fit_generator(train_loader, valid_loader, epochs=40, callbacks=[lrscheduler, checkpoint, early_stopping, csv_logger])
    loss, metric = model.evaluate_generator(test_loader)
    logging.info("Test loss: {}".format(loss))
    logging.info("Test metric: {}".format(metric))


if __name__ == '__main__':
    seed = 299792458  # "Seed" of light
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logging.getLogger().setLevel(logging.INFO)
    embeddings = load_embeddings('./data/glove_embeddings/glove.6B.100d.txt')
    train(embeddings)
