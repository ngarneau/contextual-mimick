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

from contextual_mimick import ContextualMimick
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

def prepare_datasets():
    train_embeddings = load_embeddings('./embeddings/train_embeddings.txt')
    sentences = parse_conll_file('./conll/train.txt')
    n = 15
    # print('sentences', sentences[0])
    # raw_examples = [
    #     ngram for sentence in sentences for ngram in
    #     ngrams(sentence, n, pad_left=True, pad_right=True, left_pad_symbol='<BOS>', right_pad_symbol='<EOS>')
    # ]
    # # print('raw ex', raw_examples[:15])
    # filtered_examples = [e for e in raw_examples if 'OS>' not in e[math.floor(n / 2)]]
    # # print('filt ex', filtered_examples[:10])

    # filtered_examples_splitted = [
    #     (e[:int(n / 2)], e[int(n / 2)], e[int(n / 2) + 1:]) for e in filtered_examples]
    # # print('filt ex splitted', filtered_examples_splitted[0])
    # print('my raw ex', my_raw_ex[:10], my_raw_ex[0] == filtered_examples_splitted[0])

    # # Make sure we dont have multiple begin of string and end of string within left and right context
    # examples = list()
    # for left, middle, right in filtered_examples_splitted:
    #     if left[-1] == '<BOS>':
    #         left = [left[-1]]
    #     if right[0] == '<EOS>':
    #         right = [right[0]]
    #     examples.append((left, middle, right))

    examples = [ngram for sentence in sentences for ngram in my_ngrams(sentence, n)]
    print('examples', examples[:10])

    training_data = [(x, train_embeddings[x[1].lower()]) for x in examples if x[1].lower() in train_embeddings]

    unique_examples = set()
    unique_training_data = list()
    for t in training_data:
        x = t[0]
        k = '-'.join(x[0]) + x[1] + '-'.join(x[2])
        if k not in unique_examples:
            unique_training_data.append(t)
            unique_examples.add(k)

    population_sampling = dict()
    for t in unique_training_data:
        target_word = t[0][1].lower()
        if target_word not in population_sampling:
            population_sampling[target_word] = [t]
        else:
            population_sampling[target_word].append(t)

    k = 1
    training_data = list()
    for word, e in population_sampling.items():
        if len(e) >= k:
            training_data += random.choices(e, k=k)
        else:
            training_data += e
    # training_data = training_data[:20]

    # Vectorize our examples
    word_to_idx, char_to_idx = make_vocab(sentences)
    # x_tensor, y_tensor = collate_examples([vectorizer.vectorize_example(x, y) for x, y in training_data])
    # dataset = TensorDataset(x_tensor, y_tensor)

    train_valid_ratio = 0.8
    m = int(len(training_data) * train_valid_ratio)
    train_dataset, valid_dataset = random_split(training_data, [m, len(training_data) - m])
    return train_dataset, valid_dataset


def main():
    seed = 299792458  # "Seed" of light
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

    # Prepare our examples
    train_dataset, valid_dataset = prepare_datasets()
    print(len(train_dataset), len(valid_dataset))

    use_gpu = torch.cuda.is_available()

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

    net = ContextualMimick(
        characters_vocabulary=char_to_idx,
        words_vocabulary=word_to_idx,
        characters_embedding_dimension=20,
        characters_hidden_state_dimension=50,
        words_hidden_state_dimension=100,
        word_embeddings_dimension=50,
        fully_connected_layer_hidden_dimension=50
    )
    if use_gpu:
        net.cuda()
    net.load_words_embeddings(train_embeddings)

    # lrscheduler = MultiStepLR(milestones=[3, 6, 9])
    lrscheduler = ReduceLROnPlateau(patience=2)
    early_stopping = EarlyStopping(patience=10)
    checkpoint = ModelCheckpoint('./models/contextual_mimick_n{}.torch'.format(n), save_best_only=True)
    csv_logger = CSVLogger('./train_logs/contextual_mimick_n{}.csv'.format(n))
    model = Model(net, Adam(net.parameters(), lr=0.001), square_distance, metrics=[euclidean_distance])
    model.fit_generator(train_loader, valid_loader, epochs=1000, callbacks=[lrscheduler, checkpoint, early_stopping, csv_logger])


if __name__ == '__main__':
    # main()
    tr, val = prepare_datasets()
