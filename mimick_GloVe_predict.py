import argparse
import logging
import os

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import torch
from comick import Mimick
from mimick_GloVe import prepare_data, evaluate, save_char_embeddings
# from utils import save_embeddings, load_embeddings, load_vocab
from utils import square_distance, cosine_sim

import numpy as np
import random
# import pickle as pkl

# from sklearn.metrics.pairwise import cosine_similarity
# from pytoune import torch_to_numpy, tensors_to_variables
from pytoune.framework import Model
from torch.optim import Adam


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
    evaluate(model, test_loader)

    save_char_embeddings(model, char_to_idx, 'char_'+model_name)
