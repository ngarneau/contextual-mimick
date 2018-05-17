import logging
import random

import numpy as np

import torch


def make_idx(vocab: set):
    idx = dict()
    idx['PAD'] = 0
    for v in sorted(vocab):
        idx[v] = len(idx)
    return idx


def make_vocab_and_idx(sequences):
    words_vocab = {word for sentence in sequences for word in sentence}
    words_to_idx = make_idx(words_vocab)
    return words_vocab, words_to_idx


def train_with_comick(train_func, model, model_state_path, n, oov_words, model_name='vanilla', device=0, debug=False):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        map_location = lambda storage, loc: storage.cuda(0)
    else:
        map_location = lambda storage, loc: storage
    for i in range(10):
        # Control of randomization
        model_name = '{}_i{}'.format(model_name, i)
        seed = 42 + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        logging.info('Reloading fresh Comick model with {} weights'.format(model_state_path))
        model.load_state_dict(torch.load(model_state_path, map_location))
        train_func(model, n, oov_words, model_name, device, debug)


def train_without_comick(train_func, embeddings, model_name='vanilla', device=0, debug=False):
    for i in range(10):
        # Control of randomization
        model_name = '{}_i{}'.format(model_name, i)
        seed = 42 + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        train_func(embeddings, model_name, device, debug)
