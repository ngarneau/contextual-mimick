import numpy
import torch
from torch._utils import _accumulate
from torch.nn import MSELoss, functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pytoune.framework import Model, torch_to_numpy
from pytoune.framework.callbacks import *

from model import Mimick


def load_embeddings(path):
    embeddings = {}
    # First we read the embeddings from the file, only keeping vectors for the words we need.
    i = 0
    with open(path, 'r') as embeddings_file:
        for line in embeddings_file:
            if i > 100000:
                break
            i += 1
            fields = line.strip().split(' ')
            word = fields[0]
            vector = numpy.asarray(fields[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def build_vocab(words):
    words = sorted(words)
    chars_to_idx = {
        'PAD': 0,
        'UNK': 1
    }
    for word in words:
        for character in word:
            if character not in chars_to_idx:
                chars_to_idx[character] = len(chars_to_idx)
    return chars_to_idx


class WordsVectorizer:
    def __init__(self, chars_to_idx):
        self.chars_to_idx = chars_to_idx

    def vectorize_sequence(self, word):
        if 'UNK' in self.chars_to_idx:
            unknown_index = self.chars_to_idx['UNK']
            return [self.chars_to_idx.get(char, unknown_index) for char in word]
        else:
            return [self.chars_to_idx[char] for char in word]

    def vectorize_example(self, x, y):
        vectorized_word = self.vectorize_sequence(x)
        return (
            vectorized_word,
            y
        )

    def vectorize_unknown_example(self, x):
        vectorized_word = self.vectorize_sequence(x)
        return vectorized_word


def pad_sequences(vectorized_seqs, seq_lengths):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    return seq_tensor


def collate_examples(samples):
    words, labels = list(zip(*samples))

    seq_lengths = torch.LongTensor([len(s) for s in words])
    padded_words = pad_sequences(words, seq_lengths)
    labels = torch.FloatTensor(numpy.array(labels))
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    _, rev_perm_idx = perm_idx.sort(0)

    padded_words = padded_words[perm_idx]
    labels = labels[perm_idx]

    return (
        padded_words,
        labels
    )


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    indices = torch.randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def euclidean_distance(y_pred_tensor, y_true_tensor):
    y_pred = torch_to_numpy(y_pred_tensor)
    y_true = torch_to_numpy(y_true_tensor)
    dist = numpy.linalg.norm((y_true - y_pred), axis=1).mean()
    return torch.FloatTensor([dist.tolist()])


def square_distance(input, target):
    return F.pairwise_distance(input, target).mean()


def load_vocab(path):
    vocab = set()
    with open(path) as fhandle:
        for line in fhandle:
            vocab.add(line[:-1])
    return vocab


def main():
    seed = 299792458  # "Seed" of light
    torch.manual_seed(seed)
    numpy.random.seed(seed)

    train_embeddings = load_embeddings('./embeddings/train_embeddings.txt')

    sum = 0.0
    for embedding in train_embeddings.values():
        sum += numpy.linalg.norm(embedding)
    print(sum/len(train_embeddings))

    vocab = build_vocab(train_embeddings.keys())
    corpus_vectorizer = WordsVectorizer(vocab)

    # Train dataset
    x_tensor, y_tensor = collate_examples(
        [corpus_vectorizer.vectorize_example(word, embedding) for word, embedding in train_embeddings.items()])
    dataset = TensorDataset(x_tensor, y_tensor)

    train_valid_ratio = 0.8
    m = int(len(dataset) * train_valid_ratio)
    train_dataset, valid_dataset = random_split(dataset, [m, len(dataset) - m])

    print(len(train_dataset), len(valid_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=True
    )

    net = Mimick(
        characters_vocabulary=vocab,
        characters_embedding_dimension=20,
        characters_hidden_state_dimension=50,
        word_embeddings_dimension=50,
        fully_connected_layer_hidden_dimension=50
    )

    lrscheduler = ReduceLROnPlateau(patience=5, factor=.1)
    early_stopping = EarlyStopping(patience=10)
    checkpoint = ModelCheckpoint('mimick.torch')
    model = Model(net, Adam(net.parameters(), lr=0.001), square_distance, metrics=[euclidean_distance])
    model.fit_generator(train_loader, valid_loader, n_epochs=1000, callbacks=[lrscheduler, checkpoint, early_stopping])
    # model.fit_generator(train_loader, valid_loader, n_epochs=1000, callbacks=[])

if __name__ == '__main__':
    main()
