"""
This module contains classes to manage dataset by their class for very unbalanced datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import random
import numpy as np

__author__ = "Jean-Samuel Leboeuf"
__date__ = "2018-04-04"
__version__ = "0.2.3"

class PerClassDataset(Dataset):
    """
    This class implements a dataset which keeps examples according to their label.
    The data are organized in a dictionary in the format {label:list_of_examples}

    Arguments:
        dataset (Iterable): Iterable of pairs of examples (x, y). 'y' must be hashable (str, tuple, etc.).
        transform (Callable, optional, default=None): Transformation applied to x before item is returned.
        target_transform (Callable, optional, default=None): Transformation applied to y before item is returned.
        filter_cond (Callable, optional, default=None): Filtering condition that takes 2 arguments (x and y), and returns True or False whether or not the example should be included in the dataset. If filter_cond is None, no filtering is made.
        labels_mapping (Dictionary, optional, default=None): A dictionary mapping each label to an index. Labels missing from this mapping will be automatically added while building the dataset. Indices of the mapping should never exceed len(labels_mapping). This is useful if the same mapping is needed for different datasets.
        """

    def __init__(self, dataset, transform=None, target_transform=None, filter_cond=None, labels_mapping=None):
        if filter_cond == None:
            def filter_cond(x, y): return True
        self.labels_mapping = labels_mapping
        if self.labels_mapping == None:
            self.labels_mapping = {}
        self._len = 0
        self._build_dataset(dataset, filter_cond)
        self.transform = transform
        if self.transform == None:
            self.transform = lambda x: x
        self.target_transform = target_transform
        if self.target_transform == None:
            self.target_transform = lambda x: x

    def filter_labels(self, condition):
        """
        Filters the examples of the dataset according to their label. 'condition' must be a callable of 2 arguments, 'label' and 'N', the label and the number of examples for this label. If 'condition' is True, the label if kept, else the label and all its examples are removed from the dataset.
        """
        for label, N in self:
            if not condition(label, N):
                idx = self.labels_mapping[label]
                del self.dataset[idx]
                del self.nb_examples_per_label[idx]
                self._len -= N

    def _build_dataset(self, dataset, filter_cond):
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

        self.nb_examples_per_label = {
            k: len(v) for k, v in self.dataset.items()}

    def nb_examples_for_label(self, label):
        """
        Returns the number of examples there is in the dataset for the specified label.
        """
        idx = self.labels_mapping[label]
        return self.nb_examples_per_label[idx] if idx in self.nb_examples_per_label else 0

    def __iter__(self):
        """
        Iterates over all labels of the dataset, returning (label, number_of_examples_for_this_label).
        """
        for label, idx in self.labels_mapping.items():
            yield (label, self.nb_examples_for_label(label))

    def __getitem__(self, label_i):
        """
        Returns the i-th example of a given class label. Expect 2 arguments: the label and the position i of desired example. Labels can be accessed via the labels_mapping attribute or by calling iter() on the dataset.
        """
        if isinstance(label_i, tuple) and len(label_i) == 1: label_i = label_i[0]
        label, i = label_i
        x = self.dataset[self.labels_mapping[label]][i]

        return (self.transform(x), self.target_transform(label))

    def split(self, ratio=.8, shuffle=True, reuse_label_mappings=False):
        """
        Splits the dataset in two disjoint subdatasets where labels are unique in each, according to the ratio. Labels are selected at random if shuffle is set to True.
        """        
        non_empty_labels = [(label, idx) for label, idx in self.labels_mapping.items() if self.nb_examples_for_label(label) > 0]
        if shuffle: random.shuffle(non_empty_labels)
        
        m = int(ratio*len(self.dataset))
        subdataset_1 = [(x, label) for label, i in non_empty_labels[:m] for x in self.dataset[i]]
        subdataset_2 = [(x, label) for label, i in non_empty_labels[m:] for x in self.dataset[i]]

        labels_mapping = None
        if reuse_label_mappings:
            labels_mapping = self.labels_mapping
        
        subdataset_1 = PerClassDataset(subdataset_1, transform=self.transform, target_transform=self.target_transform, labels_mapping=labels_mapping)
        subdataset_2 = PerClassDataset(subdataset_2, transform=self.transform, target_transform=self.target_transform, labels_mapping=labels_mapping)
        return subdataset_1, subdataset_2

    def __len__(self):
        """
        The len() of such a dataset is ambiguous. The len() given here is the total number of examples in the dataset, while calling len(obj.dataset) will yield the number of classes in the dataset.
        """
        return self._len
    
    def stats(self, inferior_bounds=[]):
        """
        Compute some statistics of the dataset and returns a dictionary. A list of inclusive inferior bounds can be provided to compare the statistics below and above each bounds.
        """
        if isinstance(inferior_bounds, int):
            inferior_bounds = [inferior_bounds]

        stats = {}
        stats['number of examples'] = len(self)
        stats['number of labels'] = len(self.labels_mapping)
        stats['number of non-empty labels'] = len(self.dataset)
        stats['mean number of examples per labels'] = np.mean([N for N in self.nb_examples_per_label.values()])
        stats['median number of examples per labels'] = np.median([N for N in self.nb_examples_per_label.values()])

        max_N, min_N = 0, len(self)
        max_N_labels, min_N_labels = [], []
        nb_labels_with_N_examples = {}
        bounds_stats = [{'bound':bound,
                         'number of labels with less or equal examples':0,
                         'number of examples for these labels':0} for bound in inferior_bounds]
        for label, N in iter(self):
            if N in nb_labels_with_N_examples:
                nb_labels_with_N_examples[N] += 1
            else:
                nb_labels_with_N_examples[N] = 1
            
            if N > max_N:
                max_N = N
                max_N_labels = [label]
            elif N == max_N:
                max_N_labels.append(label)
            
            if N < min_N:
                min_N = N
                min_N_labels = [label]
            elif N == min_N:
                min_N_labels.append(label)
            
            for bound, bound_stats in zip(inferior_bounds, bounds_stats):
                if N <= bound:
                    bound_stats['number of labels with less or equal examples'] += 1
                    bound_stats['number of examples for these labels'] += N

        stats['number of labels with N examples'] = nb_labels_with_N_examples
        # stats['least common labels'] = min_N_labels
        stats['least common labels number of examples'] = min_N
        # stats['most common labels'] = max_N_labels
        stats['most common labels number of examples'] = max_N
        if inferior_bounds != []:
            stats['bounds statistics'] = bound_stats
        
        return stats


class PerClassSampler():
    """
    Samples iteratively exemples of a PerClassDataset, one label at a time.
    
    Arguments:
        dataset (PerClassDataset): Source of the data to be sampled from.
        k (int, optional, default=1): Number of examples per class to be sampled in each epoch. If k=-1, all examples are sampled per epoch, without any up- or downsampling (this is useful for validation or test).
        shuffle (Boolean, optional, default=False): If False, examples are sampled in a cyclic way, else they are selected randomly (with replacement if k != -1 and without replacement if k=-1).
        filter_labels_cond (Callable, optional, default=None): A callable which takes 2 arguments: label, N (the label and an int denoting the number of examples available for this label) and returns whether or not this class should be sampled.
    """
    def __init__(self, dataset, k=-1, shuffle=True, filter_labels_cond=None):
        self.dataset = dataset
        self.k = k
        self.epoch = 0
        self.shuffle = shuffle
        self.cond = filter_labels_cond
        if self.cond == None:
            self.cond = lambda label, N: True
        self._len = len(self._generate_indices(reset_epoch=True))
    
    def _generate_indices(self, reset_epoch=False):
        if self.k == -1:
            indices = [(label, i) for label, N in self.dataset for i in range(N) if self.cond(label, N)]
        elif self.shuffle:
            indices = [(label, random.randrange(N)) for label, N in self.dataset for j in range(self.k) if N > 0 and self.cond(label, N)]
        else:
            indices = [(label, (self.epoch*self.k+j)%N) for label, N in self.dataset for j in range(self.k) if N > 0 and self.cond(label, N)]

        if self.shuffle:
            random.shuffle(indices)
        
        self.epoch += 1
        if reset_epoch:
            self.epoch = 0
        
        return indices

    def __iter__(self):
        indices = self._generate_indices()
        return (idx for idx in indices)
    
    def __len__(self):
        """
        Returns the number of minibatchs that will be produced in one epoch.
        """
        return self._len


class BatchSampler(object):
    """
    Wraps another sampler to yield a mini-batch of indices.
    This class is identical to the BatchSampler of PyTorch at the exception of the index NOT casted as an integer. This is needed to be compatible with the PerClassDataset getitem.
    """
    def __init__(self, sampler, batch_size=1, drop_last=False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class DataLoader(DataLoader):
    """
    Overloads the DataLoader Class of PyTorch so that it can copy the data to the GPU if desired.
    """
    def __init__(self, *args, use_gpu=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gpu = use_gpu and torch.cuda.is_available()

    def _to_gpu(self, *obj):
        """
        Applies .cuda() on obj. Recursively unpacks the tuples of the object before applying .cuda().
        """
        if not self.use_gpu:
            if len(obj) == 1:
                obj = obj[0]
            return obj
        if len(obj) == 1 and not isinstance(obj[0], tuple):
            return obj[0].cuda()
        if len(obj) == 1 and isinstance(obj[0], tuple):
            obj = obj[0]
        return tuple(self._to_gpu(o) for o in obj)

    def __iter__(self):
        for x, y in super().__iter__():
            yield self._to_gpu(x), self._to_gpu(y)

class PerClassLoader():
    """
    This class implements a dataloader that returns exactly k examples per class per epoch. This is simply a pipeline of PerClassSampler -> BatchSampler -> DataLoader.

    Arguments:
        dataset (PerClassDataset): Source of the data to be sampled from.
        collate_fn (Callable, optional, default=None): Returns a concatenated version of a list of examples.
        k (int, optional, default=-1): Number of examples per class to be sampled in each epoch. If k=-1, all examples are sampled per epoch, without any up- or downsampling (this is useful for validation or test for example).
        batch_size (Integer, optional, default=1): Number of examples returned per batch.
        use_gpu (Boolean, optional, default=False): Specify if the loader puts the data to GPU or not.
        """
    def __init__(self, dataset, collate_fn=None, k=1, batch_size=1, use_gpu=False):
        self.dataset = dataset
        self.sampler = PerClassSampler(dataset, k)
        self.batch_sampler = BatchSampler(self.sampler, batch_size)
        if collate_fn == None:
            collate_fn = lambda batch: [*zip(*batch)]
        self.loader = DataLoader(dataset, collate_fn=collate_fn, batch_sampler=self.batch_sampler, use_gpu=use_gpu)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


if __name__ == '__main__':
    # Script to test the dataset and dataloader
    M = 9
    N_labels = 15
    data = [(i,str(j)) for i in range(M) for j in range(N_labels)]
    dataset = PerClassDataset(data)
    stats = dataset.stats(9)
    for stats, value in stats.items():
        print(stats+': '+str(value))
    print('total number of examples:', len(dataset))
    print('number of classes:', len(dataset.dataset))
    
    
    print('\n\nTesting with PerClassLoader')
    loader = PerClassLoader(dataset, k=-1, batch_size=16)

    print('len loader:', len(loader))

    n_ex = 0
    for i, batch in enumerate(loader):
        batch_size = len(batch[0])
        print('batch size:', batch_size)
        n_ex += batch_size
    print('number of examples per epoch:', n_ex)
    print('number of steps:', i+1)

    print('\nTesting splitted datasets\n')
    d1, d2 = dataset.split(ratio=.5, shuffle=False, reuse_label_mappings=True)
    for d in [d1, d2]:
        print('total number of examples:', len(d))
        print('number of classes:', len(d.dataset))
        loader = PerClassLoader(d, k=-1, batch_size=16)

        print('len loader:', len(loader))

        n_ex = 0
        for i, batch in enumerate(loader):
            batch_size = len(batch[0])
            print('batch size:', batch_size)
            n_ex += batch_size
        print('number of examples per epoch:', n_ex)
        print('number of steps:', i+1, '\n')

    print('\n\nTesting with PerClassSampler')
    sampler = PerClassSampler(dataset=dataset, k=-1)
    batch_sampler = BatchSampler(sampler=sampler, batch_size=16)
    loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler)

    iterator = iter(loader)
    print('len loader:', len(loader))
    n_ex = 0
    i = 0
    for x, y in iterator:
        i += 1
        batch_size = len(y)
        print('batch size:', batch_size)
        n_ex += batch_size

    print('number of examples per epoch:', n_ex)
    print('number of steps:', i, '\n')
