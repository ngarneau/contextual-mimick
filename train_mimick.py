import numpy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from pytoune.framework import Model
from pytoune.framework.callbacks import *

from mimick import Mimick
from utils import load_embeddings, pad_sequences, random_split, euclidean_distance, square_distance

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


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

    lrscheduler = ReduceLROnPlateau(milestones=[3,6,9])
    early_stopping = EarlyStopping(patience=10)
    checkpoint = ModelCheckpoint('./models/mimick.torch', save_best_only=True)
    csv_logger = CSVLogger('./train_logs/mimick.csv')
    model = Model(net, Adam(net.parameters(), lr=0.001), square_distance, metrics=[euclidean_distance])
    model.fit_generator(train_loader, valid_loader, n_epochs=1000, callbacks=[lrscheduler, checkpoint, early_stopping, csv_logger])

if __name__ == '__main__':
    main()
