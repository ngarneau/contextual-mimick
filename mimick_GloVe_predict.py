import argparse
import logging
import os

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import torch
from comick import Mimick
from mimick_GloVe import prepare_data, evaluate, save_char_embeddings, Vectorizer
from utils import save_embeddings, load_embeddings, load_vocab
from utils import square_distance, cosine_sim
from utils import pad_sequences
from per_class_dataset import DataLoader

import numpy as np
import random
# import pickle as pkl

# from sklearn.metrics.pairwise import cosine_similarity
from pytoune import tensors_to_variables, torch_to_numpy
from pytoune.framework import Model
from torch.optim import Adam


def collate_x(batch):
    x, y = zip(*batch)

    x_lengths = torch.LongTensor([len(item) for item in x])
    padded_x = pad_sequences(x, x_lengths)

    return padded_x, y


def predict_OOV(model, char_to_idx, OOV_path, filename):
    OOVs = load_vocab(OOV_path)

    vectorizer = Vectorizer(char_to_idx)
    examples = [(vectorizer.vectorize_sequence(word), word) for word in OOVs]
    loader = DataLoader(examples,
                        collate_fn=collate_x,
                        use_gpu=False,
                        batch_size=1)

    model.model.eval()
    predicted_embeddings = {}
    for x, y in loader:
        x = tensors_to_variables(x)
        embeddings = torch_to_numpy(model.model(x))
        for label, embedding in zip(y, embeddings):
            predicted_embeddings[label] = embedding
            
    save_embeddings(predicted_embeddings, filename)


if __name__ == '__main__':

    d = 100
    c = 20
    use_gpu = torch.cuda.is_available()
    use_gpu = False
    verbose = False
    debug_mode = False

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Prepare examples
    train_loader, valid_loader, test_loader, char_to_idx = prepare_data(
        d=d,
        use_gpu=use_gpu,
        batch_size=64,
        debug_mode=debug_mode,
        verbose=verbose,
    )

    # Load model
    if use_gpu:
        map_location = lambda storage, loc: storage.cuda(0)
    else:
        map_location = lambda storage, loc: storage

    model_name = 'mimick_glove_d{0}_c{1}'.format(d, c)
    model_path = './models/best_' + model_name + '.torch'
    print("Loading model from: " + model_path)
    net = Mimick(
        characters_vocabulary=char_to_idx,
        characters_embedding_dimension=c,
        word_embeddings_dimension=d,
        fc_dropout_p=0.5,
        comick_compatibility=False,
    )
    net.load_state_dict(torch.load(model_path, map_location))
    model = Model(
        model=net,
        optimizer=Adam(net.parameters(), lr=0.001),
        loss_function=square_distance,
        metrics=[cosine_sim],
    )
    print('Done.')

    # Evaluation
    # evaluate(model, test_loader)

    # save_char_embeddings(model, char_to_idx, 'char_'+model_name)

    for dataset in ['conll', 'semeval', 'sentiment']:
        path = './data/'+dataset+'_embeddings_settings/setting1/glove/oov.txt'
        predict_OOV(model, char_to_idx, path, dataset+'_OOV_embeddings_'+model_name)

