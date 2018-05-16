import logging
import pickle
import random

import numpy as np
from pytoune.framework import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger, Model
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from downstream_task.models import LSTMClassifier
from downstream_task.sequence_classification import acc, collate_examples
from downstream_task.utils import make_vocab_and_idx
from utils import load_embeddings
import torch


def parse_pickle_file(filename):
    data, target_mapping = pickle.load(open(filename, 'rb'))
    sentences = list()
    labels = list()
    for sentence, label in data:
        sentences.append([w.lower() for w in sentence])
        labels.append(int(np.argmax(label)))
    return sentences, labels


def launch_train(embeddings, model_name, device, debug):
    if debug:
        epochs = 1
    else:
        epochs = 40
    train_sentences, train_tags = parse_pickle_file('./data/sentiment/train.pickle')
    valid_sentences, valid_tags = parse_pickle_file('./data/sentiment/dev.pickle')
    test_sentences, test_tags = parse_pickle_file('./data/sentiment/test.pickle')

    words_vocab, words_to_idx = make_vocab_and_idx(train_sentences + valid_sentences + test_sentences)
    tags_to_idx = {
        0: 0,
        1: 1
    }

    train_sentences = [[words_to_idx[word] for word in sentence] for sentence in train_sentences]
    valid_sentences = [[words_to_idx[word] for word in sentence] for sentence in valid_sentences]
    test_sentences = [[words_to_idx[word] for word in sentence] for sentence in test_sentences]

    train_dataset = list(zip(train_sentences, train_tags))
    valid_dataset = list(zip(valid_sentences, valid_tags))
    test_dataset = list(zip(test_sentences, test_tags))

    def cuda_collate(samples):
        words_tensor, labels_tensor = collate_examples(samples)
        return words_tensor.cuda(), labels_tensor.cuda()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cuda_device = device
        torch.cuda.set_device(cuda_device)
        logging.info('Using GPU')

    if use_gpu:
        collate_fn = cuda_collate
    else:
        collate_fn = collate_examples

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    net = LSTMClassifier(
        100,
        50,
        words_to_idx,
        len(tags_to_idx)
    )
    net.load_words_embeddings(embeddings)
    if use_gpu:
        net.cuda()

    lrscheduler = ReduceLROnPlateau(patience=2)
    early_stopping = EarlyStopping(patience=5)
    checkpoint = ModelCheckpoint('./models/sentiment_{}.torch'.format(model_name), save_best_only=True,
                                 restore_best=True)
    csv_logger = CSVLogger('./train_logs/sentiment_{}.csv'.format(model_name))
    loss = CrossEntropyLoss()
    model = Model(net, Adam(net.parameters(), lr=0.001), loss, metrics=[acc])
    model.fit_generator(train_loader, valid_loader, epochs=epochs,
                        callbacks=[lrscheduler, checkpoint, early_stopping, csv_logger])
    loss, metric = model.evaluate_generator(test_loader)
    logging.info("Test loss: {}".format(loss))
    logging.info("Test metric: {}".format(metric))


def train(embeddings, model_name='vanilla', device=0, debug=False):
    for i in range(5):
        # Control of randomization
        model_name = '{}_i{}'.format(model_name, i)
        seed = 42 + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        launch_train(embeddings, model_name, device, debug)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    embeddings = load_embeddings('./data/glove_embeddings/glove.6B.100d.txt')
    train(embeddings, 'vanilla')
