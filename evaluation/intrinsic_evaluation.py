import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import sys

sys.path.append('..')

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pytoune import torch_to_numpy, tensors_to_variables
from utils import save_embeddings


def eucl_dist(y_true, y_pred):
    return np.linalg.norm(y_pred - y_true)


def cos_sim(y_true, y_pred):
    return float(cosine_similarity(y_pred, y_true))


def evaluate(model, test_loader, test_embeddings, save=True, model_name=None):
    mean_pred_embeddings = predict_mean_embeddings(model, test_loader)

    if save:
        if model_name == None:
            raise ValueError('A filename should be provided.')
        save_embeddings(mean_pred_embeddings, model_name)

    predicted_results = {}

    euclidean_distances = []
    cos_sims = []

    nb_of_pred = 0
    for label in mean_pred_embeddings:
        if label in test_embeddings:
            y_pred = mean_pred_embeddings[label].reshape(1, -1)
            y_true = test_embeddings[label].reshape(1, -1)
            euclidean_distances.append(eucl_dist(y_true, y_pred))
            cos_sims.append(cos_sim(y_true, y_pred))
            nb_of_pred += 1

    logging.info('\nResults on the test:')
    logging.info('Mean euclidean dist: {}'.format(np.mean(euclidean_distances)))
    logging.info('Variance of euclidean dist: {}'.format(np.std(euclidean_distances)))
    logging.info('Mean cosine sim: {}'.format(np.mean(cos_sims)))
    logging.info('Variance of cosine sim: {}'.format(np.std(cos_sims)))
    logging.info('Number of labels evaluated: {}'.format(nb_of_pred))
    return mean_pred_embeddings


def predict_embeddings(model, loader):
    model.model.eval()
    predicted_embeddings = {}
    for x, y in loader:
        x = tensors_to_variables(x)
        embeddings = torch_to_numpy(model.model(x))
        for label, embedding in zip(y, embeddings):
            if label in predicted_embeddings:
                predicted_embeddings[label].append(embedding)
            else:
                predicted_embeddings[label] = [embedding]

    return predicted_embeddings


def predict_mean_embeddings(model, loader):
    predicted_embeddings = predict_embeddings(model, loader)

    mean_pred_embeddings = {}
    for label in predicted_embeddings:
        mean_pred_embeddings[label] = np.mean(
            np.array(predicted_embeddings[label]), axis=0)
    return mean_pred_embeddings


def __evaluate(model, loader, embeddings):
    predicted_embeddings = predict_embeddings(model, loader)

    predicted_cos_sim = {}
    predicted_eucl_dist = {}
    for label, pred_embeddings in predicted_embeddings:
        y_true = embeddings[label]
        predicted_cos_sim[label] = [cos_sim(y_true, y_pred) for y_pred in pred_embeddings]
        predicted_eucl_dist[label] = [eucl_dist(y_true, y_pred) for y_pred in pred_embeddings]

    pass
