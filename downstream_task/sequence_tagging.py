from collections import defaultdict
import numpy as np
from pytoune import torch_to_numpy, torch
from sklearn.metrics import f1_score, classification_report
from torch.nn import CrossEntropyLoss

from utils import pad_sequences

loss = CrossEntropyLoss(ignore_index=0)


def sequence_cross_entropy(y_pred, y):
    return loss(y_pred.view(y_pred.shape[0] * y_pred.shape[1], -1), y.view(y.shape[0] * y.shape[1]))


def f1(y_pred_tensor, y_true_tensor):
    y_pred_tensor = y_pred_tensor.view(y_pred_tensor.shape[0] * y_pred_tensor.shape[1], -1)
    y_true_tensor = y_true_tensor.view(y_true_tensor.shape[0] * y_true_tensor.shape[1])
    y_pred = torch_to_numpy(y_pred_tensor)
    y_true = torch_to_numpy(y_true_tensor)
    predictions = list()
    truths = list()
    for yp, yt in zip(y_pred, y_true):
        if yt != 0:
            predictions.append(np.argmax(yp))
            truths.append(yt)
    return torch.FloatTensor([f1_score(truths, predictions, average='macro')])


def acc(y_pred_tensor, y_true_tensor):
    y_pred_tensor = y_pred_tensor.view(y_pred_tensor.shape[0] * y_pred_tensor.shape[1], -1)
    y_true_tensor = y_true_tensor.view(y_true_tensor.shape[0] * y_true_tensor.shape[1])
    y_pred = torch_to_numpy(y_pred_tensor)
    y_true = torch_to_numpy(y_true_tensor)

    predictions = list()
    for yp, yt in zip(y_pred, y_true):
        if yt != 0:
            predictions.append(np.argmax(yp) == yt)

    return y_pred_tensor.data.new([np.mean(predictions) * 100])


def make_idx(vocab: set):
    idx = dict()
    idx['PAD'] = 0
    for v in sorted(vocab):
        idx[v] = len(idx)
    return idx


def collate_examples(samples):
    words, labels = list(zip(*samples))

    seq_lengths = torch.LongTensor([len(s) for s in words])
    padded_words = pad_sequences(words, seq_lengths)
    padded_labels = pad_sequences(labels, seq_lengths)

    return (
        padded_words,
        padded_labels
    )


def collate_examples_multiple_tags(samples):
    examples, labels = list(zip(*samples))

    words = list()
    chars = list()
    bos = list()
    for e in examples:
        words.append(e[0])
        chars.append(e[1])
        bos.append(e[2])

    seq_lengths = torch.LongTensor([len(s) for s in words])
    padded_words = pad_sequences(words, seq_lengths)

    padded_chars = list()
    for char_list in chars:
        chars_seq_lengths = torch.LongTensor([len(s) for s in char_list])
        padded_chars.append(pad_sequences(char_list, chars_seq_lengths))

    padded_bos = list()
    for bos_list in bos:
        bos_seq_lengths = torch.LongTensor([len(s) for s in bos_list])
        padded_bos.append(pad_sequences(bos_list, bos_seq_lengths))

    tags_to_produce = set()
    for example in labels:
        tags_to_produce.update(example.keys())

    labels_splitted = defaultdict(list)
    for tag in tags_to_produce:
        for example in labels:
            if tag in example:
                labels_splitted[tag].append(example[tag])
            else:
                labels_splitted[tag].append([0])

    padded_labels = dict()
    for label, tags in labels_splitted.items():
        padded_labels[label] = pad_sequences(tags, seq_lengths)
        tags_to_produce.add(label)

    return (
        (padded_words, padded_chars, padded_bos, tags_to_produce),
        padded_labels
    )


def make_vocab_and_idx(sequences):
    words_vocab = {word for sentence in sequences for word in sentence}
    words_to_idx = make_idx(words_vocab)
    return words_vocab, words_to_idx
