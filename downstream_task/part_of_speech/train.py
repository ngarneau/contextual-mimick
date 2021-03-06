import logging
import random

from pytoune.framework import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger, Model
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader

from downstream_task.models import LSTMTagger
from downstream_task.sequence_tagging import sequence_cross_entropy, acc, collate_examples, make_vocab_and_idx
from utils import load_embeddings
import torch
import numpy as np


def parse_pos_file(filename):
    sentences = list()
    targets = list()
    with open(filename, encoding='utf-8') as fhandler:
        sentence = list()
        tags = list()
        for line in fhandler:
            if not (line.startswith('-DOCSTART-') or line.startswith('\n')):
                token, pos, chunk, e = line[:-1].split(' ')
                sentence.append(token.lower())
                tags.append(pos)
            else:
                if len(sentence) > 0:
                    sentences.append(sentence)
                    targets.append(tags)
                sentence = list()
                tags = list()
    return sentences, targets


def launch_train(embeddings, model_name, device, debug):
    if debug:
        epochs = 1
    else:
        epochs = 40
    train_sentences, train_tags = parse_pos_file('./data/conll/train.txt')
    valid_sentences, valid_tags = parse_pos_file('./data/conll/valid.txt')
    test_sentences, test_tags = parse_pos_file('./data/conll/test.txt')

    words_vocab, words_to_idx = make_vocab_and_idx(train_sentences + valid_sentences + test_sentences)
    tags_vocab, tags_to_idx = make_vocab_and_idx(train_tags + valid_tags + test_tags)

    train_sentences = [[words_to_idx[word] for word in sentence] for sentence in train_sentences]
    train_tags = [[tags_to_idx[word] for word in sentence] for sentence in train_tags]

    valid_sentences = [[words_to_idx[word] for word in sentence] for sentence in valid_sentences]
    valid_tags = [[tags_to_idx[word] for word in sentence] for sentence in valid_tags]

    test_sentences = [[words_to_idx[word] for word in sentence] for sentence in test_sentences]
    test_tags = [[tags_to_idx[word] for word in sentence] for sentence in test_tags]

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

    net = LSTMTagger(
        100,
        50,
        words_to_idx,
        len(tags_to_idx)
    )
    net.load_words_embeddings(embeddings)
    if use_gpu:
        net.cuda()

    lrscheduler = ReduceLROnPlateau(patience=2)
    early_stopping = EarlyStopping(patience=5)
    model_path = './models/'
    checkpoint = ModelCheckpoint(model_path+'pos_'+model_name+'.torch',
                                 save_best_only=True,
                                 restore_best=True,
                                 temporary_filename=model_path+'tmp_pos_'+model_name+'.torch',
                                 verbose=True)
    csv_logger = CSVLogger('./train_logs/pos_{}.csv'.format(model_name))
    model = Model(net, Adam(net.parameters(), lr=0.001), sequence_cross_entropy, metrics=[acc])
    model.fit_generator(train_loader, valid_loader, epochs=epochs, callbacks=[lrscheduler, checkpoint, early_stopping, csv_logger])
    loss, metric = model.evaluate_generator(test_loader)
    logging.info("Test loss: {}".format(loss))
    logging.info("Test metric: {}".format(metric))


def train(embeddings, model_name='vanilla', device=0, debug=False):
    for i in range(10):
        # Control of randomization
        model_name = '{}_i{}'.format(model_name, i)
        seed = 42 + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        launch_train(embeddings, model_name, device, debug)

def train_previous_mimick(embeddings, device=0, debug=False):
    previous_mimick_embeddings = load_embeddings('./data/previous_mimick/conll_model_output')
    embeddings.update(previous_mimick_embeddings)
    model_name = 'previous_mimick'
    train(embeddings, model_name, device, debug)


def train_baseline(embeddings, device=0, debug=False):
    model_name = 'baseline'
    train(embeddings, model_name, device, debug)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    embeddings = load_embeddings('./data/glove_embeddings/glove.6B.100d.txt')
    train_previous_mimick(embeddings)

