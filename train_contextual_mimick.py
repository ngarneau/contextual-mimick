import argparse
import math
import os
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import numpy
import torch
from pytoune.framework import Model
from pytoune.framework.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger, MultiStepLR
from torch.optim import Adam, SGD
import random
from time import time

from contextual_mimick import ContextualMimick
from utils import load_embeddings, random_split, euclidean_distance,\
    square_distance, parse_conll_file,\
    make_vocab, WordsInContextVectorizer, Corpus, collate_examples
from PerClassDataset import *

def ngrams(sequence, n, pad_left=1, pad_right=1, left_pad_symbol='<BOS>', right_pad_symbol='<EOS>'):
    sequence = [left_pad_symbol]*pad_left + sequence + [right_pad_symbol]*pad_right

    L = len(sequence)
    m = n//2
    for i, item in enumerate(sequence[1:-1]):
        left_idx = max(0, i-m+1)
        left_side = tuple(sequence[left_idx:i+1])
        right_idx = min(L, i+m+2)
        right_side = tuple(sequence[i+2:right_idx])
        yield (left_side, item, right_side)

def split_train_valid(examples, ratio):
    m = int(ratio*len(examples))
    train_examples, valid_examples = [], []
    for i, x in enumerate(examples):
        if i < m:
            train_examples.append(x)
        else:
            valid_examples.append(x)
    return train_examples, valid_examples

def prepare_data(n=15, ratio=.8, use_gpu=False):
    train_embeddings = load_embeddings('./embeddings/train_embeddings.txt')
    sentences = parse_conll_file('./conll/train.txt')[:100]
    word_to_idx, char_to_idx = make_vocab(sentences)
    vectorizer = WordsInContextVectorizer(word_to_idx, char_to_idx)

    examples = set((ngram, ngram[1]) for sentence in sentences for ngram in ngrams(sentence, n) if ngram[1] in train_embeddings) # Keeps only different ngrams which have a training embedding 
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
    print('Datasets size - Train:', len(train_dataset), 'Valid:',  len(valid_dataset))

    collate_fn = lambda samples: collate_examples([(*x,y) for x, y in samples])
    train_loader = KPerClassLoader(dataset=train_dataset,
                                   collate_fn=collate_fn,
                                   batch_size=16,
                                   k=1,
                                   use_gpu=use_gpu)
    valid_loader = KPerClassLoader(dataset=valid_dataset,
                                   collate_fn=collate_fn,
                                   batch_size=16,
                                   k=1,
                                   use_gpu=use_gpu)

    return train_loader, valid_loader, word_to_idx, char_to_idx, train_embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n")
    parser.add_argument("k")
    args = parser.parse_args()

    seed = 299792458  # "Seed" of light
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    n = int(args.n)
    k = int(args.k)

    use_gpu = torch.cuda.is_available()

    # Prepare our examples
    train_loader, valid_loader, word_to_idx, char_to_idx, train_embeddings = prepare_data(n=n, ratio=.8, use_gpu=use_gpu)

    net = get_contextual_mimick(char_to_idx, word_to_idx)

    if use_gpu:
        net.cuda()
    net.load_words_embeddings(train_embeddings)

    # lrscheduler = MultiStepLR(milestones=[3, 6, 9])
    lrscheduler = ReduceLROnPlateau(patience=2)
    early_stopping = EarlyStopping(patience=10)
    model_path = './models/'
    model_file = 'testing_contextual_mimick_n{}.torch'.format(n)
    os.makedirs(model_path, exist_ok=True)
    checkpoint = ModelCheckpoint(model_path+model_file,
                                 save_best_only=True,
                                 temporary_filename=model_path+'temp_'+model_file)
    # There is a bug in Pytoune with the CSVLogger on my computer
    logger_path = './train_logs/'
    logger_file = 'testing_contextual_mimick_n{}.csv'.format(n)
    os.makedirs(logger_path, exist_ok=True)
    csv_logger = CSVLogger(logger_path + logger_file)
    model = Model(net, Adam(net.parameters(), lr=0.001), square_distance, metrics=[euclidean_distance])
    callbacks = [lrscheduler, checkpoint, early_stopping, csv_logger]
    model.fit_generator(train_loader, valid_loader, epochs=1000, callbacks=callbacks)


if __name__ == '__main__':
    from time import time
    t = time()
    try:
        
        main()
        
        
    except:
        print('Execution stopped after {:.2f} seconds.'.format(time()-t))
        raise
    print('Execution completed in {:.2f} seconds.'.format(time()-t))
