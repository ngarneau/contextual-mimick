import os
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
from torch.utils.data.sampler import Sampler
from utils import DataLoader
import random
from time import time

from contextual_mimick import ContextualMimick
from utils import load_embeddings, random_split, euclidean_distance,\
    square_distance, parse_conll_file,\
    make_vocab, WordsInContextVectorizer, Corpus, collate_examples

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

class KPerClassLoader():
    """
    This class implements a dataloader that returns exactly k examples per class per epoch.

    'dataset' (PerClassDataset): Collection of data to be sampled.
    'collate_fn' (Callable): Returns a concatenated version of a list of examples.
    'k' (Integer, optional, default=1): Number of examples from each class loaded per epoch.
    'batch_size' (Integer, optional, default=1): Number of examples returned per batch.
    'use_gpu' (Boolean, optional, default=False): Specify if the loader puts the data to GPU or not.
    """
    def __init__(self, dataset, collate_fn, k=1, batch_size=1, use_gpu=False):
        self.dataset = dataset
        self.epoch = 0
        self.k = k
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.use_gpu = use_gpu

    def to_gpu(self, obj):
        return obj.cuda() if self.use_gpu else obj

    def __iter__(self):
        e = self.epoch
        batch = []
        for j in range(self.k):
            for label, N in iter(self.dataset):
                idx = (e*self.k+j)%N
                sample = self.dataset[label, idx]
                batch.append(sample)
                if len(batch) == self.batch_size:
                    yield self.to_gpu(self.collate_fn(batch))
                    batch = []
        self.epoch += 1
        if len(batch) > 0:
            yield self.to_gpu(self.collate_fn(batch))
    
    def __len__(self):
        """
        Returns the number of minibatchs that will be produced in one epoch.
        """
        return (self.k*len(self.dataset.labels_mapping) + self.batch_size - 1)//self.batch_size

class PerClassDataset():
    """
    This class implements a dataset which keeps examples according to their label.
    The data are organized in a dictionary in the format {label:list_of_examples}
    """
    def __init__(self, dataset, transform=None, target_transform=None, filter_cond=None, labels_mapping={}):
        """
        'dataset' must be an iterable of pair of elements (x, y). y must be hashable (str, tuple, etc.)
        'transform' must be a callable applied to x before item is returned.
        'target_transform' must be a callable applied to y before item is returned.
        'filter_cond' is a callable which takes 2 arguments (x and y), and returns True or False whether or not this example should be included in the dataset. If filter_cond is None, no filtering will be made.
        'labels_mapping' must be a dictionary mapping labels to an index. Labels missing from this mapping will be automatically added while building the dataset. Indices of the mapping should never exceed len(labels_mapping).
        """
        if filter_cond == None:
            filter_cond = lambda x, y: True
        self.labels_mapping = labels_mapping
        self._len = 0
        self.build_dataset(dataset, filter_cond)
        self.transform = transform
        if self.transform == None:
            self.transform = lambda x: x
        self.target_transform = target_transform
        if self.target_transform == None:
            self.target_transform = lambda x: x

    def build_dataset(self, dataset, filter_cond):
        """
        Takes an iterable of examples of the form (x, y) and makes it into a dictionary {class:list_of_examples}, and filters examples not satisfying the 'filter_cond' condition.
        """
        self.dataset = {}
        for x, y in dataset:
            idx = None
            if y in self.labels_mapping:
                idx = self.labels_mapping[y]
            if idx in self.dataset:
                self._len += 1
                self.dataset[idx].append(x)
            elif filter_cond(x, y):
                if idx == None:
                    idx = self.labels_mapping[y] = len(self.labels_mapping)
                self.dataset[idx] = [x]
                self._len += 1
        
        self.nb_examples_per_label = {k:len(v) for k, v in self.dataset.items()}
    
    def nb_examples_for_label(self, label):
        """
        Returns the number of examples there is in the dataset for the specified label.
        """
        return self.nb_examples_for_label[self.labels_mapping[label]]
    
    def __iter__(self):
        """
        Iterates over all labels of the dataset, returning (label, number_of_examples_for_this_label).
        """
        for label, idx in self.labels_mapping.items():
            yield (label, self.nb_examples_per_label[idx])

    def __getitem__(self, label_i):
        """
        Returns the i-th example of a given class label. Expect 2 arguments: the label and the position i of desired example. Labels can be accessed via the labels_mapping attribute or by calling iter() on the dataset.
        """
        label, i = label_i
        x = self.dataset[self.labels_mapping[label]][i]

        return (self.transform(x), self.target_transform(label))
    
    def __len__(self):
        """
        The len() of such a dataset is ambiguous. The len() given here is the total number of examples in the dataset, while calling len(obj.dataset) will yield the number of classes in the dataset.
        """
        return self._len

def split_train_valid(examples, ratio):
    m = int(ratio*len(examples))
    print('m', m)
    train_examples, valid_examples = [], []
    for i, x in enumerate(examples):
        if i < m:
            train_examples.append(x)
        else:
            valid_examples.append(x)
    return train_examples, valid_examples

def prepare_data(n=15, ratio=.8, use_gpu=False):
    train_embeddings = load_embeddings('./embeddings/train_embeddings.txt')
    sentences = parse_conll_file('./conll/train.txt')
    word_to_idx, char_to_idx = make_vocab(sentences)
    vectorizer = WordsInContextVectorizer(word_to_idx, char_to_idx)

    examples = set((ngram, ngram[1]) for sentence in sentences for ngram in ngrams(sentence, n)) # Keeps only different ngrams
    print('nb of total examples', len(examples))

    train_examples, valid_examples = split_train_valid(examples, ratio)

    filter_cond = lambda x, y: y in train_embeddings
    transform = vectorizer.vectorize_unknown_example
    target_transform = lambda y: train_embeddings[y]

    train_dataset = PerClassDataset(train_examples,
                                    filter_cond=filter_cond,
                                    transform=transform,
                                    target_transform=target_transform)
    # The filter_cond makes the dataset of different sizes each time. Should we filter before creating the dataset

    valid_dataset = PerClassDataset(valid_examples,
                                    filter_cond=filter_cond,
                                    transform=transform,
                                    target_transform=target_transform)
    print(len(train_dataset), len(valid_dataset))

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
    seed = 299792458  # "Seed" of light
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    n=15

    # use_gpu = torch.cuda.is_available()
    use_gpu = False
    # Prepare our examples
    train_loader, valid_loader, word_to_idx, char_to_idx, train_embeddings = prepare_data(n=n, ratio=.8, use_gpu=use_gpu)

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
    model_path = './models/testing_contextual_mimick_n{}.torch'.format(n)
    os.makedirs(model_path, exist_ok=True)
    checkpoint = ModelCheckpoint(model_path, save_best_only=True)
    # There is a bug in Pytoune with the CSVLogger on my computer
    # logger_path = './train_logs/testing_contextual_mimick_n{}.csv'.format(n)
    # os.makedirs(logger_path, exist_ok=True)
    # csv_logger = CSVLogger(logger_path)
    model = Model(net, Adam(net.parameters(), lr=0.001), square_distance, metrics=[euclidean_distance])
    callbacks = [lrscheduler, checkpoint, early_stopping]#, csv_logger]
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
