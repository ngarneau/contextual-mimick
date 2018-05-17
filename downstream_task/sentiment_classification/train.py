import logging
import pickle
import random

import numpy as np
from pytoune.framework import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger, Model
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from downstream_task.models import LSTMClassifier
from downstream_task.sequence_classification import acc, collate_examples
from downstream_task.utils import make_vocab_and_idx, train_with_comick
from utils import load_embeddings
import torch


def parse_pickle_file(filename):
    data, target_mapping = pickle.load(open(filename, 'rb'))
    sentences = list()
    labels = list()
    for sentence, label in data:
        sentences.append([w.lower() for w in sentence])
        labels.append(int(np.argmax(label)))
    return sentences, labels


def launch_train(model, n, oov_words, model_name, device, debug):
    if debug:
        epochs = 1
    else:
        epochs = 40
    train_sentences, train_tags = parse_pickle_file('./data/sentiment/train.pickle')
    valid_sentences, valid_tags = parse_pickle_file('./data/sentiment/dev.pickle')
    test_sentences, test_tags = parse_pickle_file('./data/sentiment/test.pickle')

    embeddings = load_embeddings('./data/glove_embeddings/glove.6B.100d.txt')

    # words_vocab, words_to_idx = make_vocab_and_idx(train_sentences + valid_sentences + test_sentences)
    words_to_idx = model.words_vocabulary
    tags_to_idx = {
        0: 0,
        1: 1
    }

    train_sentences = [[words_to_idx[word] for word in sentence] for sentence in train_sentences]
    valid_sentences = [[words_to_idx[word] for word in sentence] for sentence in valid_sentences]
    test_sentences = [[words_to_idx[word] for word in sentence] for sentence in test_sentences]

    train_dataset = list(zip(train_sentences, train_tags))
    valid_dataset = list(zip(valid_sentences, valid_tags))
    test_dataset = list(zip(test_sentences, test_tags))

    def cuda_collate(samples):
        words_tensor, labels_tensor = collate_examples(samples)
        return words_tensor.cuda(), labels_tensor.cuda()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cuda_device = device
        torch.cuda.set_device(cuda_device)
        logging.info('Using GPU')

    if use_gpu:
        collate_fn = cuda_collate
    else:
        collate_fn = collate_examples

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    net = LSTMClassifier(
        100,
        50,
        words_to_idx,
        len(tags_to_idx),
        model,
        oov_words,
        n,
        use_gpu
    )
    net.load_words_embeddings(embeddings)
    if use_gpu:
        net.cuda()

    lrscheduler = ReduceLROnPlateau(patience=2)
    early_stopping = EarlyStopping(patience=5)
    model_path = './models/'
    checkpoint = ModelCheckpoint(model_path+'sentiment_'+model_name+'.torch',
                                 save_best_only=True,
                                 restore_best=True,
                                 temporary_filename=model_path+'tmp_sentiment_'+model_name+'.torch',
                                 verbose=True)
    csv_logger = CSVLogger('./train_logs/sentiment_{}.csv'.format(model_name))
    loss = CrossEntropyLoss()
    model = Model(net, Adam(net.parameters(), lr=0.001), loss, metrics=[acc])
    model.fit_generator(train_loader, valid_loader, epochs=epochs,
                        callbacks=[lrscheduler, checkpoint, early_stopping, csv_logger])
    loss, metric = model.evaluate_generator(test_loader)
    logging.info("Test loss: {}".format(loss))
    logging.info("Test metric: {}".format(metric))


def train(model, model_state_path, n, oov_words, model_name='vanilla', device=0, debug=False):
    train_with_comick(launch_train, model, model_state_path, n, oov_words, model_name, device, debug)


def train_mimick(embeddings, device=0, debug=False):
    previous_mimick_embeddings = load_embeddings('./mimick_oov_predicted_embeddings/sentiment_OOV_embeddings_mimick_glove_d100_c20.txt')
    embeddings.update(previous_mimick_embeddings)
    model_name = 'mimick'
    train(embeddings, model_name, device, debug)


def train_previous_mimick(embeddings, device=0, debug=False):
    previous_mimick_embeddings = load_embeddings('./data/previous_mimick/sent_model_output')
    embeddings.update(previous_mimick_embeddings)
    model_name = 'previous_mimick'
    train(embeddings, model_name, device, debug)


def train_baseline(embeddings, device=0, debug=False):
    model_name = 'baseline'
    train(embeddings, model_name, device, debug)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    embeddings = load_embeddings('./data/glove_embeddings/glove.6B.100d.txt')
    train_mimick(embeddings)
