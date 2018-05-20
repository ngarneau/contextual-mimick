import logging
import random

import numpy as np

import torch

from utils import load_embeddings


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


def get_map_location(device):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        map_location = lambda storage, loc: storage.cuda(device)
    else:
        map_location = lambda storage, loc: storage
    return map_location


def refresh_mimick(model, model_state_path):
    use_gpu = torch.cuda.is_available()
    characters_embeddings = load_embeddings('./predicted_char_embeddings/char_Pinter_mimick_glove_d100_c20')
    characters_vocab = make_idx(set(characters_embeddings.keys()))
    model.load_mimick(model_state_path, use_gpu)
    model.load_chars_embeddings(characters_embeddings)


def refresh_comick(model, model_state_path, device=0):
    model.load_state_dict(torch.load(model_state_path, get_map_location(device)))


def train_with_comick(train_func, model, model_state_path, refresh_func, n, oov_words, model_name='vanilla', device=0,
                      debug=False):
    for i in range(5):
        # Control of randomization
        model_name = '{}_i{}'.format(model_name, i)
        seed = 42 + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        logging.info('Reloading fresh Model with {} weights'.format(model_state_path))
        refresh_func(model, model_state_path)
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
