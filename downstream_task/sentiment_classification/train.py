import logging
import pickle
import numpy as np
from pytoune.framework import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger, Model
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from downstream_task.models import LSTMClassifier
from downstream_task.sequence_classification import acc, collate_examples 
from downstream_task.utils import make_vocab_and_idx
from utils import load_embeddings



def parse_pickle_file(filename):
    data, target_mapping = pickle.load(open(filename, 'rb'))
    sentences = list()
    labels = list()
    for sentence, label in data:
        sentences.append(sentence)
        labels.append(int(np.argmax(label)))
    return sentences, labels



def train(embeddings, model_name='vanilla'):
    train_sentences, train_tags = parse_pickle_file('./data/sentiment/train.pickle')
    valid_sentences, valid_tags = parse_pickle_file('./data/sentiment/dev.pickle')
    test_sentences, test_tags = parse_pickle_file('./data/sentiment/test.pickle')

    words_vocab, words_to_idx = make_vocab_and_idx(train_sentences + valid_sentences + test_sentences)
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_examples
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_examples
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_examples
    )

    net = LSTMClassifier(
        100,
        50,
        words_to_idx,
        len(tags_to_idx)
    )
    net.load_words_embeddings(embeddings)

    lrscheduler = ReduceLROnPlateau(patience=5)
    early_stopping = EarlyStopping(patience=10)
    checkpoint = ModelCheckpoint('./models/sentiment_{}.torch'.format(model_name), save_best_only=True)
    csv_logger = CSVLogger('./train_logs/sentiment_{}.csv'.format(model_name))
    loss = CrossEntropyLoss()
    model = Model(net, Adam(net.parameters(), lr=0.001), loss, metrics=[acc])
    model.fit_generator(train_loader, valid_loader, epochs=40, callbacks=[lrscheduler, checkpoint, early_stopping, csv_logger])
    loss, metric = model.evaluate_generator(test_loader)
    logging.info("Test loss: {}".format(loss))
    logging.info("Test metric: {}".format(metric))


if __name__ == '__main__':
    embeddings = load_embeddings('./data/glove_embeddings/glove.6B.100d.txt')
    train(embeddings)
