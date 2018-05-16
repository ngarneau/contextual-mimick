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


class Evaluator:
    def __init__(self, model, loader, idx_to_word=None, word_embeddings=None):
        self.results = {}
        self.results_per_labels = {}
        self.global_results = {}
        self._build_results(model, loader, idx_to_word, word_embeddings)
    
    def _build_results(self, model, loader, idx_to_word, word_embeddings):
        model.model.eval()
        
        for x, y in loader:
            embeddings = torch_to_numpy(model.model(tensors_to_variables(x)))
            for context, label, embedding in zip(x, y, embeddings):
                idx = len(self.results)
                converted_context = self.convert_context(context, idx_to_word)

                result = EmbeddingResult(label, embedding.reshape(1,-1), context=converted_context)
                if word_embeddings != None and label in word_embeddings:
                    result.compute_metrics(
                        word_embeddings[label].reshape(1, -1))
                self.results[idx] = result
                
                if label in self.results_per_labels:
                    self.results_per_labels[label]['results_idx'].append(idx)
                else:
                    self.results_per_labels[label] = {'results_idx':[idx]}
        
        if word_embeddings != None:
            self._compute_metrics(word_embeddings)
    
    def _compute_metrics(self, word_embeddings):
        for label, results in self.results_per_labels.items():
            pred_embed = np.array([self.results[i].embedding for i in results['results_idx']])
            mean_pred_embed = np.mean(pred_embed, axis=0)
            results['mean_of_pred_embed'] = mean_pred_embed
            true_embedding = word_embeddings[label]
            results['cos_sim_of_mean_pred_embed'] = cos_sim(true_embedding, mean_pred_embed)
            results['eucl_dist_of_mean_pred_embed'] = eucl_dist(true_embedding, mean_pred_embed)
        
            pred_cos_sim = np.array([self.results[i].cos_sim for i in results['results_idx']])
            results['mean_of_cos_sim'] = np.mean(pred_cos_sim)
            results['var_of_cos_sim'] = np.std(pred_cos_sim)
            pred_eucl_dist = np.array([self.results[i].eucl_dist for i in results['results_idx']])
            results['mean_of_eucl_dist'] = np.mean(pred_eucl_dist)
            results['var_of_eucl_dist'] = np.std(pred_eucl_dist)
        

        self.global_results['mean_of_mean_of_cos_sim'] = np.mean(np.array(
            [result['mean_of_cos_sim'] for result in self.results_per_labels.values()]
        ))
        self.global_results['mean_of_mean_of_eucl_dist'] = np.mean(np.array(
            [result['mean_of_eucl_dist'] for result in self.results_per_labels.values()]
        ))

        self.global_results['mean_of_cos_sim_of_mean_pred_embed'] = np.mean(np.array(
            [result['cos_sim_of_mean_pred_embed'] for result in self.results_per_labels.values()]
        ))
        self.global_results['mean_of_eucl_dist_of_mean_pred_embed'] = np.mean(np.array(
            [result['eucl_dist_of_mean_pred_embed'] for result in self.results_per_labels.values()]
        ))

    @staticmethod
    def convert_context(context, idx_to_word):
        if word_to_idx != None:
            CL, w, CR = context
            CL = ' '.join([idx_to_word[i] for i in CL])
            CR = ' '.join([idx_to_word[i] for i in CR])
            w = idx_to_word[w]
            context = [CL, w, CR]  # To be implemented

        return context

    def __getitem__(self, label):
        """
        Returns a list of EmbeddingResult objects for the given label.
        """
        return self.results_per_labels[label]


class EmbeddingResult:
    def __init__(self, label, embedding, context=None):
        self.label = label
        self.embedding = embedding
        self.context = context
    
    def compute_metrics(self, true_embedding):
        self.cos_sim = cos_sim(true_embedding, self.embedding)
        self.eucl_dist = eucl_dist(true_embedding, self.embedding)
