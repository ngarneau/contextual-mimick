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
from torch.utils.data.sampler import Sampler
from utils import DataLoader
import random
from time import time

from contextual_mimick import ContextualMimick
from utils import load_embeddings, random_split, euclidean_distance,\
    square_distance, parse_conll_file,\
    make_vocab, WordsInContextVectorizer, Corpus, collate_examples

def make_ngrams(sequence, n, pad_left=True, pad_right=True, left_pad_symbol='<BOS>', right_pad_symbol='<EOS>'):
    sequence.append(right_pad_symbol)
    sequence.insert(0, left_pad_symbol)

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

    dataset must be a dictionary of the form {class:[label, list_of_examples]}.

    :TODO: valid dataset, use_gpu
    """
    def __init__(self, dataset, collate_fn, k=1, batch_size=1, transform=lambda x:x):
        self.dataset = dataset
        self.epoch = 0
        self.k = k
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.transform = transform
    
    def __iter__(self):
        e = self.epoch
        batch = []
        for j in range(self.k):
            for y, xs in self.dataset.values():
                x = xs[(e*self.k+j)%len(xs)]
                sample = self.transform((x, y))
                batch.append(sample)
                if len(batch) == self.batch_size:
                    yield self.collate_fn(batch)
                    batch = []
        self.epoch += 1
        if len(batch) > 0:
            yield self.collate_fn(batch)
    
    def __len__(self):
        """
        Returns the number of minibatchs that will be produced in one epoch.
        """
        return (self.k*len(self.dataset) + self.batch_size - 1)//self.batch_size


def prepare_datasets():
    train_embeddings = load_embeddings('./embeddings/train_embeddings.txt')
    sentences = parse_conll_file('./conll/train.txt')
    n = 15

    ngrams = set(ngram for sentence in sentences for ngram in make_ngrams(sentence, n)) # Keeps only different ngrams

    t = time()
    # Creates a dictionary where each key has for value a list of [embedding, list_of_ngrams]
    ngrams_per_token = {}
    for ngram in ngrams:
        focus_token = ngram[1]
        if focus_token in ngrams_per_token:
            ngrams_per_token[focus_token][1].append(ngram)
        elif focus_token in train_embeddings:
            ngrams_per_token[focus_token] = [train_embeddings[focus_token], [ngram]]
    print(time()-t)

    word_to_idx, char_to_idx = make_vocab(sentences)
    vectorizer = WordsInContextVectorizer(word_to_idx, char_to_idx)


    loader = KPerClassLoader(dataset=ngrams_per_token,
                                collate_fn=collate_examples,
                                batch_size=16,transform=vectorizer.vectorize_example,
                                k=3)
    print('len loader', len(loader))
    print('len dict', len(ngrams_per_token))
    l = iter(loader)
    somme = 0
    for step in range(len(loader)):
        x, y = next(l)
        # print(len(y))
        somme += len(y)
    print(step+1)
    print(somme)
    l = iter(loader)
    next(l)



    # print('ngrams', ngrams[:10])

        
    # ngrams_per_word = {}
    # for sentence in sentences:
    #     for ngram in make_ngrams(sentence, n):
    #         try:
    #             # Check in the training data if this particular ngram already exists.
    #             if ngram not in ngrams_per_word[ngram[1]]:
    #                 ngrams_per_word[ngram[1]].append(ngram)
    #         except KeyError:
    #             # If not, create the entry and create a list of ngrams
    #             ngrams_per_word[ngram[1]] = [ngram]

    # for i, (k, n) in enumerate(ngrams_per_word.items()):
    #     print(k, n)
    #     if i > 3:
    #         break
        

    training_data = [(x, train_embeddings[x[1]]) for x in ngrams if x[1] in train_embeddings]

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
    return train_dataset, valid_dataset, word_to_idx, char_to_idx


def main():
    seed = 299792458  # "Seed" of light
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

    # Prepare our examples
    train_dataset, valid_dataset, word_to_idx, char_to_idx = prepare_datasets()
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
    from time import time

    t = time()
    tr, val, w, c = prepare_datasets()
    print('Execution completed in {:.2f} seconds.'.format(time()-t))
