import torch
from torch.utils.data import Dataset


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

    def __len__(self):
        """
        The len() of such a dataset is ambiguous. The len() given here is the total number of examples in the dataset, while calling len(obj.dataset) will yield the number of classes in the dataset.
        """
        return self._len


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
        e = self.epoch
        batch = []
        for j in range(self.k):
            for label, N in iter(self.dataset):
                if N > 0:
                    idx = (e*self.k+j) % N
                    sample = self.dataset[label, idx]
                    batch.append(sample)
                    if len(batch) == self.batch_size:
                        z = self.to_gpu(self.collate_fn(batch))
                        # print(z, len(z[0]))
                        yield z  # self.to_gpu(self.collate_fn(batch))
                        batch = []
        self.epoch += 1
        if len(batch) > 0:
            yield self.to_gpu(self.collate_fn(batch))

    def __len__(self):
        """
        Returns the number of minibatchs that will be produced in one epoch.
        """
        return (self.k*len(self.dataset.labels_mapping) + self.batch_size - 1)//self.batch_size
