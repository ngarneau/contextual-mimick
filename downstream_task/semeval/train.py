import logging
import random

from pytoune.framework import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger, Model
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader

from downstream_task.models import LSTMTagger
from downstream_task.sequence_tagging import sequence_cross_entropy, acc, collate_examples, make_vocab_and_idx, f1
from utils import load_embeddings
import torch
import numpy as np


def parse_semeval_file(filename):
    sentences = list()
    targets = list()
    with open(filename, encoding='utf-8') as fhandler:
        sentence = list()
        tags = list()
        for line in fhandler:
            if not (line.startswith('-DOCSTART-') or line.startswith('\n')):
                token, id, start, end, e = line[:-1].split(' ')
                sentence.append(token.lower())
                tags.append(e)
            else:
                if len(sentence) > 0:
                    sentences.append(sentence)
                    targets.append(tags)
                sentence = list()
                tags = list()
    return sentences, targets


def train(embeddings, model_name='vanilla', device=0):
    train_sentences, train_tags = parse_semeval_file('./data/scienceie/train_spacy.txt')
    valid_sentences, valid_tags = parse_semeval_file('./data/scienceie/valid_spacy.txt')
    test_sentences, test_tags = parse_semeval_file('./data/scienceie/test_spacy.txt')

    words_vocab, words_to_idx = make_vocab_and_idx(train_sentences + valid_sentences + test_sentences)
    tags_vocab, tags_to_idx = make_vocab_and_idx(train_tags + valid_tags + test_tags)

    train_sentences = [[words_to_idx[word] for word in sentence] for sentence in train_sentences]
    train_tags = [[tags_to_idx[word] for word in sentence] for sentence in train_tags]

    valid_sentences = [[words_to_idx[word] for word in sentence] for sentence in valid_sentences]
    valid_tags = [[tags_to_idx[word] for word in sentence] for sentence in valid_tags]

    test_sentences = [[words_to_idx[word] for word in sentence] for sentence in test_sentences]
    test_tags = [[tags_to_idx[word] for word in sentence] for sentence in test_tags]


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

    net = LSTMTagger(
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
    checkpoint = ModelCheckpoint('./models/semeval_{}.torch'.format(model_name), save_best_only=True, restore_best=True)
    csv_logger = CSVLogger('./train_logs/semeval_{}.csv'.format(model_name))
    model = Model(net, Adam(net.parameters(), lr=0.001), sequence_cross_entropy, metrics=[f1])
    model.fit_generator(train_loader, valid_loader, epochs=40, callbacks=[lrscheduler, checkpoint, early_stopping, csv_logger])

    loss, metric = model.evaluate_generator(test_loader)
    logging.info("Test loss: {}".format(loss))
    logging.info("Test metric: {}".format(metric))


if __name__ == '__main__':
    for i in range(5):
        seed = 42 + i  # "Seed" of light
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        logging.getLogger().setLevel(logging.INFO)
        embeddings = load_embeddings('./data/glove_embeddings/glove.6B.100d.txt')
        train(embeddings, "vanilla_i{}".format(i))
