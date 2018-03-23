import torch
from torch.utils.data import Dataset
import random


class PerClassDataset(Dataset):
    """
    This class implements a dataset which keeps examples according to their label.
    The data are organized in a dictionary in the format {label:list_of_examples}
    """

    def __init__(self, dataset, transform=None, target_transform=None, filter_cond=None, labels_mapping=None):
        """
        'dataset' must be an iterable of pair of elements (x, y). y must be hashable (str, tuple, etc.)
        'transform' must be a callable applied to x before item is returned.
        'target_transform' must be a callable applied to y before item is returned.
        'filter_cond' is a callable which takes 2 arguments (x and y), and returns True or False whether or not this example should be included in the dataset. If filter_cond is None, no filtering will be made.
        'labels_mapping' must be a dictionary mapping labels to an index. Labels missing from this mapping will be automatically added while building the dataset. Indices of the mapping should never exceed len(labels_mapping).
        """
        if filter_cond == None:
            def filter_cond(x, y): return True
        self.labels_mapping = labels_mapping
        if self.labels_mapping == None:
            self.labels_mapping = {}
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
        print(non_empty_labels[:m])
        subdataset_1 = [(x, label) for label, i in non_empty_labels[:m] for x in self.dataset[i]]
        print(len(subdataset_1))
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


class PerClassLoader():
    """
    This class implements a dataloader that returns exactly k examples per class per epoch.

    'dataset' (PerClassDataset): Collection of data to be sampled.
    'collate_fn' (Callable): Returns a concatenated version of a list of examples.
    'k' (Integer, optional, default=1): Number of examples from each class loaded per epoch. If k is set to -1, all examples are loaded.
    'batch_size' (Integer, optional, default=1): Number of examples returned per batch.
    'use_gpu' (Boolean, optional, default=False): Specify if the loader puts the data to GPU or not.
    """

    def __init__(self, dataset, collate_fn=None, k=1, batch_size=1, use_gpu=False):
        self.dataset = dataset
        self.epoch = 0
        self.k = k
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        if collate_fn == None:
            self.collate_fn = lambda batch: [*zip(*batch)]
        self.use_gpu = use_gpu

    def to_gpu(self, *obj):
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
        return tuple(self.to_gpu(o) for o in obj)
    
    def __iter__(self):
        if self.k == -1:
            idx_iterator = ((label, i) for label, N in self.dataset for i in range(N))
        else:
            idx_iterator = ((label, (self.epoch*self.k+j)%N) for label, N in self.dataset for j in range(self.k) if N > 0)
        
        batch = []
        for label, i in idx_iterator:
            batch.append(self.dataset[label, i])
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
        if self.k == -1:
            length = (len(self.dataset) + self.batch_size - 1)//self.batch_size
        else:
            length = (self.k*len(self.dataset.dataset) + self.batch_size - 1)//self.batch_size
        return length

if __name__ == '__main__':
    # Script to test the dataset and dataloader
    M = 9
    N_labels = 15
    data = [(i,str(j)) for i in range(M) for j in range(N_labels)]
    
    dataset = PerClassDataset(data)
    print('total number of examples:', len(dataset))
    print('number of classes:', len(dataset.dataset))
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
    for dataset in [d1, d2]:
        print('total number of examples:', len(dataset))
        print('number of classes:', len(dataset.dataset))
        loader = PerClassLoader(dataset, k=-1, batch_size=16)

        print('len loader:', len(loader))

        n_ex = 0
        for i, batch in enumerate(loader):
            batch_size = len(batch[0])
            print('batch size:', batch_size)
            n_ex += batch_size
        print('number of examples per epoch:', n_ex)
        print('number of steps:', i+1, '\n')
