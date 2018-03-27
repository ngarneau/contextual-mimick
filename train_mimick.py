import argparse

import numpy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from pytoune.framework import Model
from pytoune.framework.callbacks import *

from mimick import Mimick
from per_class_dataset import PerClassDataset, PerClassLoader
from utils import load_embeddings, pad_sequences, random_split, euclidean_distance, square_distance, cosine_sim, \
    parse_conll_file, WordsInContextVectorizer, make_vocab, ngrams

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


def prepare_data(embeddings, sentences, n=15, ratio=.8, use_gpu=False, k=1):
    word_to_idx, char_to_idx = make_vocab(sentences)
    vectorizer = WordsInContextVectorizer(word_to_idx, char_to_idx)

    examples = set((ngram, ngram[1]) for sentence in sentences for ngram in ngrams(sentence, n) if
                   ngram[1] in embeddings)  # Keeps only different ngrams which have a training embedding
    print('Number of unique examples:', len(examples))

    # train_examples, valid_examples = split_train_valid(examples, ratio)

    # filter_cond = lambda x, y: y in embeddings
    transform = vectorizer.vectorize_unknown_example
    target_transform = lambda y: embeddings[y]

    dataset = PerClassDataset(
        examples,
        transform=transform,
        target_transform=target_transform
    )
    train_dataset, valid_dataset = dataset.split(ratio=ratio, shuffle=True, reuse_label_mappings=False)

    print('Datasets size - Train:', len(train_dataset), 'Valid:', len(valid_dataset))
    print('Datasets labels - Train:', len(train_dataset.dataset), 'Valid:', len(valid_dataset.dataset))

    collate_fn = lambda samples: collate_examples([(x[1], y) for x, y in samples])
    train_loader = PerClassLoader(dataset=train_dataset,
                                  collate_fn=collate_fn,
                                  batch_size=1,
                                  k=k,
                                  use_gpu=use_gpu)
    valid_loader = PerClassLoader(dataset=valid_dataset,
                                  collate_fn=collate_fn,
                                  batch_size=1,
                                  k=-1,
                                  use_gpu=use_gpu)

    return train_loader, valid_loader, word_to_idx, char_to_idx


def main(d):
    seed = 299792458  # "Seed" of light
    torch.manual_seed(seed)
    numpy.random.seed(seed)

    path_embeddings = './embeddings_settings/setting2/1_glove_embeddings/glove.6B.{}d.txt'.format(d)
    try:
        train_embeddings = load_embeddings(path_embeddings)
    except:
        if d == 50:
            path_embeddings = './embeddings/train_embeddings.txt'
            train_embeddings = load_embeddings(path_embeddings)
            print('Loading {}d embeddings from: "' + path_embeddings + '"'.format(d))
        else:
            raise
    print('Loading ' + str(d) + 'd embeddings from: "' + path_embeddings + '"')

    path_sentences = './conll/train.txt'
    sentences = parse_conll_file(path_sentences)

    train_loader, valid_loader, word_to_idx, char_to_idx = prepare_data(
        embeddings=train_embeddings,
        sentences=sentences,
        n=1,
        ratio=.8,
        use_gpu=False,
        k=1)

    # sum = 0.0
    # for embedding in train_embeddings.values():
    #     sum += numpy.linalg.norm(embedding)
    # print(sum / len(train_embeddings))
    #
    # vocab = build_vocab(train_embeddings.keys())
    # corpus_vectorizer = WordsVectorizer(vocab)

    # Train dataset
    # x_tensor, y_tensor = collate_examples(
    #     [corpus_vectorizer.vectorize_example(word, embedding) for word, embedding in train_embeddings.items()])
    # dataset = TensorDataset(x_tensor, y_tensor)
    #
    # train_valid_ratio = 0.8
    # m = int(len(dataset) * train_valid_ratio)
    # train_dataset, valid_dataset = random_split(dataset, [m, len(dataset) - m])
    #
    # print(len(train_dataset), len(valid_dataset))
    #
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=1,
    #     shuffle=True
    # )
    #
    # valid_loader = DataLoader(
    #     valid_dataset,
    #     batch_size=1,
    #     shuffle=True
    # )

    net = Mimick(
        characters_vocabulary=char_to_idx,
        characters_embedding_dimension=20,
        characters_hidden_state_dimension=50,
        word_embeddings_dimension=d,
        fully_connected_layer_hidden_dimension=50
    )

    lrscheduler = ReduceLROnPlateau(patience=2)
    early_stopping = EarlyStopping(patience=10)
    checkpoint = ModelCheckpoint('./models/mimick{}.torch'.format(d), save_best_only=True)
    csv_logger = CSVLogger('./train_logs/mimick{}.csv'.format(d))
    model = Model(net, Adam(net.parameters(), lr=0.001), square_distance, metrics=[cosine_sim])
    model.fit_generator(train_loader, valid_loader, epochs=1000,
                        callbacks=[lrscheduler, checkpoint, early_stopping, csv_logger])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("d", default=50, nargs='?')
    args = parser.parse_args()
    d = int(args.d)
    if d not in [50, 100, 200, 300]:
        raise ValueError("The embedding dimension 'd' should of 50, 100, 200 or 300.")
    main(d)
