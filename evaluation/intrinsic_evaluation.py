import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import sys
sys.path.append('..')

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pytoune import torch_to_numpy, tensors_to_variables
from utils import save_embeddings
import pickle as pkl


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


class Evaluator:
    def __init__(self, model, loader, idx_to_word=None, idx_to_char=None, word_embeddings=None):
        self.results = {}
        self.results_per_labels = {}
        self.global_results = {}
        self._build_results(model, loader, idx_to_word, idx_to_char, word_embeddings)
    
    def _build_results(self, model, loader, idx_to_word, idx_to_char, word_embeddings):
        model.model.eval()
        
        for x, y in loader:
            embeddings = torch_to_numpy(model.model(tensors_to_variables(x)))
            for *context, label, embedding in zip(*x, y, embeddings):
                idx = len(self.results)
                converted_context = self.convert_context(context, idx_to_word, idx_to_char)

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
            true_embedding = word_embeddings[label].reshape(1, -1)
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
    def convert_context(context, idx_to_word, idx_to_char):
        if idx_to_word != None:
            CL, w, CR = context
            CL = ' '.join([idx_to_word[i] for i in CL if i != 0])
            CR = ' '.join([idx_to_word[i] for i in CR if i != 0])
            w = ''.join([idx_to_char[i] for i in w if i != 0])
            context = [CL, w, CR]

        return context

    def __getitem__(self, label):
        """
        Returns a dictionary of the results for the given label, and a list of individual results in the form of EmbeddingResult objects.
        """
        label_results = [self.results[i] for i in self.results_per_labels[label]['results_idx']]
        return self.results_per_labels[label], label_results

    def sort_by(self, attribute, reverse=False):
        """
        Sort results by the given 'attribute'. If 'attribute' is 'cos_sim' or 'eucl_dist', a list of individual instances of EmbeddingResult is returned, else it is a list of tuples (label, dict_of_results).
        """
        if attribute in ['cos_sim',
                         'eucl_dist']:
            keygetter = lambda r: getattr(r, attribute)
            sorted_list = sorted(self.results.values(), key=keygetter, reverse=reverse)
        elif attribute in ['cos_sim_of_mean_pred_embed',
                           'eucl_dist_of_mean_pred_embed',
                           'mean_of_cos_sim',
                           'var_of_cos_sim',
                           'mean_of_eucl_dist',
                           'var_of_eucl_dist']:
            keygetter = lambda label_res: label_res[1][attribute]
            sorted_list = sorted(self.results_per_labels.items(), key=keygetter, reverse=reverse)
        else:
            raise ValueError('Unvalid attribute.')

        return sorted_list
    
    def get_mean_predicted_embeddings(self):
        return {label: r['mean_of_pred_embed'] for label, r in self.results_per_labels.items()}

class EmbeddingResult:
    def __init__(self, label, embedding, context=None):
        self.label = label
        self.embedding = embedding
        self.context = context
    
    def compute_metrics(self, true_embedding):
        self.cos_sim = cos_sim(true_embedding, self.embedding)
        self.eucl_dist = eucl_dist(true_embedding, self.embedding)
    
    def __str__(self):
        return self.label + '\n' + ' '.join(self.context) + '\n' + 'cos_sim ' + str(self.cos_sim) + '\n' + 'eucl_dist ' + str(self.eucl_dist) + '\n'


if __name__ == '__main__':
    model_name = 'testing_conll_n5_k2_d100_e100_lrcomick_v1.2'
    filepath = './intrinsic_{}.pkl'.format(model_name)
    with open(filepath, 'rb') as file:
        results = pkl.load(file)
    
    print(results.global_results)
    cos_sim_res = results.sort_by('cos_sim', reverse=True)
    print(cos_sim_res[0], cos_sim_res[-1])

    eucl_dist_res = results.sort_by('eucl_dist')
    print(eucl_dist_res[0], eucl_dist_res[-1])

    var_cos_sim = results.sort_by('var_of_cos_sim')
    filtered_var_cos_sim = [(l, r) for l, r in var_cos_sim if len(r['results_idx']) >= 3]

    for l, r in filtered_var_cos_sim[:-6:-1]:
        print(l, len(r['results_idx']), 'var of cos sim', r['var_of_cos_sim'])

