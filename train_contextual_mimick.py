import argparse
import math
import logging
import os

from contextual_mimick import get_contextual_mimick

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import numpy
import torch
from pytoune.framework import Model
from pytoune.framework.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from torch.optim import Adam
import random

from utils import load_embeddings, euclidean_distance, \
    square_distance, parse_conll_file, \
    make_vocab, WordsInContextVectorizer, collate_examples, ngrams
from per_class_dataset import *


def split_train_valid(examples, ratio):
    m = int(ratio * len(examples))
    train_examples, valid_examples = [], []
    sorted_examples = sorted(examples)
    numpy.random.shuffle(sorted_examples)
    for i, x in enumerate(sorted_examples):
        if i < m:
            train_examples.append(x)
        else:
            valid_examples.append(x)
    return train_examples, valid_examples


def prepare_data(n=15, ratio=.8, use_gpu=False, k=1):
    train_embeddings = load_embeddings('./embeddings/train_embeddings.txt')
    sentences = parse_conll_file('./conll/train.txt')
    word_to_idx, char_to_idx = make_vocab(sentences)
    vectorizer = WordsInContextVectorizer(word_to_idx, char_to_idx)

    examples = set((ngram, ngram[1]) for sentence in sentences for ngram in ngrams(sentence, n) if
                   ngram[1] in train_embeddings)  # Keeps only different ngrams which have a training embedding
    print('Number of unique examples:', len(examples))

    train_examples, valid_examples = split_train_valid(examples, ratio)

    # filter_cond = lambda x, y: y in train_embeddings
    transform = vectorizer.vectorize_unknown_example
    target_transform = lambda y: train_embeddings[y]

    train_dataset = PerClassDataset(train_examples,
                                    # filter_cond=filter_cond,
                                    transform=transform,
                                    target_transform=target_transform)
    # The filter_cond makes the dataset of different sizes each time. Should we filter before creating the dataset

    valid_dataset = PerClassDataset(valid_examples,
                                    # filter_cond=filter_cond,
                                    transform=transform,
                                    target_transform=target_transform)
    print('Datasets size - Train:', len(train_dataset), 'Valid:', len(valid_dataset))
    print('Datasets labels - Train:', len(train_dataset.dataset), 'Valid:', len(valid_dataset.dataset))

    collate_fn = lambda samples: collate_examples([(*x, y) for x, y in samples])
    train_loader = KPerClassLoader(dataset=train_dataset,
                                   collate_fn=collate_fn,
                                   batch_size=1,
                                   k=k,
                                   use_gpu=use_gpu)
    valid_loader = KPerClassLoader(dataset=valid_dataset,
                                   collate_fn=collate_fn,
                                   batch_size=1,
                                   k=k,
                                   use_gpu=use_gpu)

    return train_loader, valid_loader, word_to_idx, char_to_idx, train_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", default=41, nargs='?')
    parser.add_argument("k", default=1, nargs='?')
    parser.add_argument("device", default=0, nargs='?')
    args = parser.parse_args()
    n = int(args.n)
    k = int(args.k)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.cuda.set_device(cuda_device)
        print('Using GPU')
        cuda_device = int(args.device)

    seed = 299792458  # "Seed" of light
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

    # Prepare our examples
    train_loader, valid_loader, word_to_idx, char_to_idx, train_embeddings = prepare_data(n=n, ratio=.8,
                                                                                          use_gpu=use_gpu, k=k)

    net = get_contextual_mimick(char_to_idx, word_to_idx)

    if use_gpu:
        net.cuda()
    net.load_words_embeddings(train_embeddings)

    # lrscheduler = MultiStepLR(milestones=[3, 6, 9])
    lrscheduler = ReduceLROnPlateau(patience=2)
    early_stopping = EarlyStopping(patience=10)
    model_path = './models/'
    model_file = 'contextual_mimick_n{}_k{}.torch'.format(n, k)
    os.makedirs(model_path, exist_ok=True)
    ckpt_best = ModelCheckpoint('best_' + model_path + model_file,
                                 save_best_only=True,
                                 temporary_filename='best_' + model_path + 'temp_' + model_file)

    ckpt_last = ModelCheckpoint('last_' + model_path + model_file,
                                 temporary_filename='last_' + model_path + 'temp_' + model_file)

    logger_path = './train_logs/'
    logger_file = 'contextual_mimick_n{}_k{}.csv'.format(n, k)
    os.makedirs(logger_path, exist_ok=True)
    csv_logger = CSVLogger(logger_path + logger_file)

    model = Model(model=net,
                  optimizer=Adam(net.parameters(), lr=0.001),
                  loss_function=square_distance,
                  metrics=[euclidean_distance])
    callbacks = [lrscheduler, ckpt_best, ckpt_last, early_stopping, csv_logger]

    model.fit_generator(train_loader, valid_loader, epochs=1000, callbacks=callbacks)


if __name__ == '__main__':
    from time import time
    t = time()
    try:
        main()
    except:
        print('Execution stopped after {:.2f} seconds.'.format(time() - t))
        raise
    print('Execution completed in {:.2f} seconds.'.format(time() - t))
