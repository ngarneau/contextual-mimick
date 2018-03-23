import argparse
import math
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import numpy
import torch
from nltk.util import ngrams
from itertools import chain
from pytoune.framework import Model
from pytoune.framework.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger, MultiStepLR
from torch.optim import Adam, SGD
from utils import DataLoader
import random

from contextual_mimick import ContextualMimick, get_contextual_mimick
from utils import load_embeddings, random_split, euclidean_distance, square_distance, parse_conll_file, \
    make_vocab, WordsInContextVectorizer, Corpus, collate_examples

def my_ngrams(sequence, n, pad_left=True, pad_right=True, left_pad_symbol='<BOS>', right_pad_symbol='<EOS>'):
    sequence.append(right_pad_symbol)
    sequence.insert(0, left_pad_symbol)
    L = len(sequence)
    m = n//2
    for i, item in enumerate(sequence[1:-1]):
        left_idx = max(0, i-m+1)
        left_side = sequence[left_idx:i+1]
        right_idx = min(L, i+m+2)
        right_side = sequence[i+2:right_idx]
        yield (tuple(left_side), item, tuple(right_side))



def make_training_data_unique(training_data):
    unique_examples = set()
    unique_training_data = list()
    for t in training_data:
        x = t[0]
        k = '-'.join(x[0]) + x[1] + '-'.join(x[2])
        if k not in unique_examples:
            unique_training_data.append(t)
            unique_examples.add(k)
    return unique_training_data


def group_by_target_words(unique_training_data):
    population_sampling = dict()
    for t in unique_training_data:
        target_word = t[0][1].lower()
        if target_word not in population_sampling:
            population_sampling[target_word] = [t]
        else:
            population_sampling[target_word].append(t)
    return population_sampling


def sample_population(population_sampling, k):
    training_data = list()
    for word, e in population_sampling.items():
        if len(e) >= k:
            training_data += random.choices(e, k=k)
        else:
            training_data += e
    return training_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n")
    parser.add_argument("k")
    parser.add_argument("device", default=0)
    args = parser.parse_args()
    n = int(args.n)
    k = int(args.k)
    cuda_device = int(args.device)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.cuda.set_device(cuda_device)

    seed = 299792458  # "Seed" of light
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

    # Prepare our examples
    train_embeddings = load_embeddings('./embeddings/train_embeddings.txt')
    sentences = parse_conll_file('./conll/train.txt')
    word_to_idx, char_to_idx = make_vocab(sentences)

    examples = [ngram for sentence in sentences for ngram in my_ngrams(sentence, n)]
    print('examples', examples[:10])

    training_data = [(x, train_embeddings[x[1].lower()]) for x in examples if x[1].lower() in train_embeddings]

    unique_training_data = make_training_data_unique(training_data)

    population_sampling = group_by_target_words(unique_training_data)

    training_data = sample_population(population_sampling, k)
    # training_data = training_data[:500]


    train_valid_ratio = 0.8
    m = int(len(training_data) * train_valid_ratio)
    train_dataset, valid_dataset = random_split(training_data, [m, len(training_data) - m])
    print(len(train_dataset), len(valid_dataset))


    vectorizer = WordsInContextVectorizer(word_to_idx, char_to_idx)
    train_loader = DataLoader(
        Corpus(train_dataset, 'train', vectorizer.vectorize_example),
        batch_size=16,
        collate_fn=collate_examples,
        shuffle=True,
        use_gpu=use_gpu
    )

    valid_loader = DataLoader(
        Corpus(valid_dataset, 'valid', vectorizer.vectorize_example),
        batch_size=16,
        collate_fn=collate_examples,
        shuffle=True,
        use_gpu=use_gpu
    )

    net = get_contextual_mimick(char_to_idx, word_to_idx)

    if use_gpu:
        net.cuda()
    net.load_words_embeddings(train_embeddings)


    # lrscheduler = MultiStepLR(milestones=[3, 6, 9])
    lrscheduler = ReduceLROnPlateau(patience=5)
    early_stopping = EarlyStopping(patience=20)
    checkpoint = ModelCheckpoint('./models/contextual_mimick_n{}_k{}.torch'.format(n, k), save_best_only=True)
    csv_logger = CSVLogger('./train_logs/contextual_mimick_n{}_k{}.csv'.format(n, k))
    model = Model(net, Adam(net.parameters(), lr=0.001), square_distance, metrics=[euclidean_distance])
    model.fit_generator(train_loader, valid_loader, epochs=1000, callbacks=[lrscheduler, checkpoint, early_stopping, csv_logger])


if __name__ == '__main__':
    main()
