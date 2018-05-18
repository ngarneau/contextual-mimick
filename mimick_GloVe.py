import argparse
import logging
import os

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

from comick import Mimick
from utils import save_embeddings, load_embeddings, load_vocab
from utils import square_distance, cosine_sim
from utils import pad_sequences
from per_class_dataset import *

import numpy as np
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
from pytoune import torch_to_numpy, tensors_to_variables
from pytoune.framework import Model
from pytoune.framework.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from torch.optim import Adam
from random import shuffle

def make_char_to_idx(words):
    alphabet = set()
    for word in words:
        for char in word:
            alphabet.add(char)
    
    char_to_idx = {'PAD':0}
    for char in sorted(alphabet):
        char_to_idx[char] = len(char_to_idx)
    
    return char_to_idx


class Vectorizer:
    def __init__(self, index, unknown_idx='UNK'):
        self.index = index
        self.unk = unknown_idx
        if self.unk not in self.index:
            self.index[self.unk] = len(self.index)
    
    def vectorize_sequence(self, sequence):
        vectorized_sequence = []
        for item in sequence:
            if item in self.index:
                vectorized_sequence.append(self.index[item])
            else:
                vectorized_sequence.append(self.index[self.unk])

        return vectorized_sequence
    
    def vectorize_example(self, example):
        x, y = example
        return self.vectorize_sequence(x), y


def collate_fn(batch):
    x, y = zip(*batch)

    x_lengths = torch.LongTensor([len(item) for item in x])
    padded_x = pad_sequences(x, x_lengths)

    return (padded_x, torch.FloatTensor(np.array(y)))


def collate_x(batch):
    x, y = zip(*batch)

    x_lengths = torch.LongTensor([len(item) for item in x])
    padded_x = pad_sequences(x, x_lengths)

    return padded_x, y


def predict_OOV(model, char_to_idx, OOV_path, filename, use_gpu, filepath='./mimick_oov_predicted_embeddings/'):
    OOVs = sorted(load_vocab(OOV_path))

    vectorizer = Vectorizer(char_to_idx)
    examples = [(vectorizer.vectorize_sequence(word), word) for word in OOVs]
    loader = DataLoader(examples,
                        collate_fn=collate_x,
                        use_gpu=use_gpu,
                        batch_size=1)

    model.model.eval()
    predicted_embeddings = {}
    for x, y in loader:
        x = tensors_to_variables(x)
        embeddings = torch_to_numpy(model.model(x))
        for label, embedding in zip(y, embeddings):
            predicted_embeddings[label] = embedding

    save_embeddings(predicted_embeddings, filename, filepath)


def evaluate(model, test_loader):
    eucl_dist, [cos_sim] = model.evaluate_generator(test_loader)

    logging.info('\nResults on the test:')
    logging.info('Mean euclidean dist: {}'.format(eucl_dist))
    logging.info('Mean cosine sim: {}'.format(cos_sim))


def save_char_embeddings(model, char_to_idx, filename='mimick_char_embeddings'):
    char_embeddings = {}
    for char, idx in char_to_idx.items():
        char_embeddings[char] = torch_to_numpy(
            model.model.mimick_lstm.embeddings.weight.data[idx])
    save_embeddings(char_embeddings, filename)


def prepare_data(d,
                 split_ratios=[.8,.1,.1],
                 use_gpu=False,
                 batch_size=64,
                 verbose=True,
                 debug_mode=False):
    path_embeddings = './data/glove_embeddings/glove.6B.{}d.txt'.format(d)
    if verbose:
        logging.info('Loading ' + str(d) + 'd embeddings from: "' + path_embeddings + '"')

    embeddings = load_embeddings(path_embeddings)
    words = [word for word in embeddings]
    char_to_idx = make_char_to_idx(words)

    vectorizer = Vectorizer(char_to_idx)
    examples = [(vectorizer.vectorize_sequence(word), embed) for word, embed in embeddings.items()]
    if debug_mode:
        examples = examples[:850]

    shuffle(examples)
    m_train = int(len(examples)*split_ratios[0])
    m_valid = int(len(examples)*split_ratios[1])
    # m_test = len(embeddings)*split_ratios[2]

    train_ex = examples[:m_train]
    valid_ex = examples[m_train:m_train+m_valid]
    test_ex = examples[m_train+m_valid:]
    if verbose:
        logging.info('Training size: ' + str(m_train))
        logging.info('Validation size: ' + str(m_valid))
        logging.info('Test size: ' + str(len(test_ex)))
    train_loader = DataLoader(
        train_ex,
        collate_fn=collate_fn,
        use_gpu=use_gpu,
        batch_size=batch_size)
    valid_loader = DataLoader(valid_ex,
        collate_fn=collate_fn,
        use_gpu=use_gpu,
        batch_size=batch_size)
    test_loader = DataLoader(test_ex,
        collate_fn=collate_fn,
        use_gpu=use_gpu,
        batch_size=batch_size)

    return train_loader, valid_loader, test_loader, char_to_idx


def train(model, model_name, train_loader, valid_loader, epochs=1000):
    # Create callbacks and checkpoints
    lrscheduler = ReduceLROnPlateau(patience=3, verbose=True)
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4, verbose=True)
    model_path = './models/'

    os.makedirs(model_path, exist_ok=True)
    ckpt_best = ModelCheckpoint(model_path + 'best_' + model_name + '.torch',
                                save_best_only=True,
                                restore_best=True,
                                temporary_filename=model_path + 'temp_best_' + model_name + '.torch',
                                verbose=True)

    ckpt_last = ModelCheckpoint(model_path + 'last_' + model_name + '.torch',
                                temporary_filename=model_path + 'temp_last_' + model_name + '.torch')

    logger_path = './train_logs/'
    os.makedirs(logger_path, exist_ok=True)
    csv_logger = CSVLogger(logger_path + model_name + '.csv')

    callbacks = [lrscheduler, ckpt_best, ckpt_last, early_stopping, csv_logger]

    # Fit the model
    model.fit_generator(train_loader, valid_loader,
                        epochs=epochs, callbacks=callbacks)


def main(model_name, device=0, d=100, epochs=100, char_embedding_dimension=20, debug_mode=False):
    # Global parameters
    debug_mode = debug_mode
    verbose = True
    save = True
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    logging.info("Debug mode: {}".format(debug_mode))
    logging.info("Verbose: {}".format(verbose))

    use_gpu = torch.cuda.is_available()
    # use_gpu = False
    if use_gpu:
        cuda_device = device
        torch.cuda.set_device(cuda_device)
        logging.info('Using GPU')

    # Prepare examples
    train_loader, valid_loader, test_loader, char_to_idx = prepare_data(
        d=d,
        use_gpu=use_gpu,
        batch_size=64,
        debug_mode=debug_mode,
        verbose=verbose,
    )
    logging.info('Size of alphabet: ' + str(len(char_to_idx)))

    # Initialize training parameters
    lr = 0.001
    if debug_mode:
        model_name = 'testing_' + model_name
        save = False
        epochs = 3

    # Create the model
    net = Mimick(
        characters_vocabulary=char_to_idx,
        characters_embedding_dimension=char_embedding_dimension,
        comick_compatibility=False
    )
    model = Model(
        model=net,
        optimizer=Adam(net.parameters(), lr=lr),
        loss_function=square_distance,
        metrics=[cosine_sim],
    )
    if use_gpu:
        model.cuda()

    # Set up the callbacks and train
    train(
        model, model_name,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=epochs,
    )

    evaluate(model, test_loader)

    save_char_embeddings(model, char_to_idx, 'char_'+model_name)

    for dataset in ['conll', 'scienceie', 'sentiment']:
        path = './data/'+dataset+'/oov.txt'
        predict_OOV(model, char_to_idx, path, dataset+'_OOV_embeddings_'+model_name+'.txt', use_gpu)


if __name__ == '__main__':
    from time import time

    t = time()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("d", default=100, nargs='?')
        parser.add_argument("device", default=0, nargs='?')
        args = parser.parse_args()
        device = int(args.device)
        d = int(args.d)
        if d not in [50, 100, 200, 300]:
            raise ValueError(
                "The embedding dimension 'd' should of 50, 100, 200 or 300.")
        logger = logging.getLogger()
        for e in [60]:
            for i in range(1):
                # Control of randomization
                seed = 42 + i
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                model_name = 'mimick_glove_d{}'.format(d)
                handler = logging.FileHandler('{}.log'.format(model_name))
                logger.addHandler(handler)
                main(model_name, device=device, d=d)
                logger.removeHandler(handler)
    except:
        logging.info('Execution stopped after {:.2f} seconds.'.format(time() - t))
        raise
    logging.info('Execution completed in {:.2f} seconds.'.format(time() - t))
