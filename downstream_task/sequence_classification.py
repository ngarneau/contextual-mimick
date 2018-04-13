import numpy as np
from pytoune import torch_to_numpy, torch
from utils import pad_sequences

def acc(y_pred_tensor, y_true_tensor):
    y_pred = torch_to_numpy(y_pred_tensor)
    y_true = torch_to_numpy(y_true_tensor)

    predictions = list()
    for yp, yt in zip(y_pred, y_true):
        predictions.append(np.argmax(yp) == yt)

    return y_pred_tensor.data.new([np.mean(predictions) * 100])


def collate_examples(samples):
    words, labels = list(zip(*samples))

    seq_lengths = torch.LongTensor([len(s) for s in words])
    padded_words = pad_sequences(words, seq_lengths)

    labels = torch.LongTensor(labels)

    return (
        padded_words,
        labels
    )


