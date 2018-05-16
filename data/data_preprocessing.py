import os
import logging

from gensim.models import KeyedVectors
from tqdm import tqdm

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import sys
sys.path.append('..')

from utils import ngrams, load_examples, save_examples, load_embeddings
from dataset_manager import CoNLL, SemEval, Sentiment
from per_class_dataset import *
import pickle as pkl
import json


def augment_data(examples, embeddings_path, filter_cond=None, topn=5, min_cos_sim=.6):
    if filter_cond == None:
        filter_cond = lambda label: True

    logging.info("Loading embedding model...")
    word2vec_model = KeyedVectors.load_word2vec_format(embeddings_path)
    logging.info("Done.")

    labels = sorted(set(label for x, label in examples))
    logging.info("Computing similar words for {} labels...".format(len(labels)))
    sim_words = dict()
    for label in tqdm(labels):
        sim_words[label] = word2vec_model.most_similar(label, topn=topn)
    logging.info("Done.")

    counter = 0
    new_examples = set()
    for (left_context, word, right_context), label in tqdm(examples):
        for sim_word, cos_sim in sim_words[word]:
            # Add new labels, not new examples to already existing labels.
            if filter_cond(sim_word) and cos_sim >= min_cos_sim:
                new_example = ((left_context, sim_word, right_context), sim_word)
                new_examples.add(new_example)
            else:
                counter += 1
    logging.info("Done.")
    print('number of refused examples:', counter)
    return new_examples


def preprocess_data(dataset,
                    embeddings_path,
                    topn,
                    min_cos_sim):
    path =  './' + dataset.dataset_name + '/examples/'
    # embeddings = load_embeddings(embeddings_path)

    # Training part
    examples = set((ngram, ngram[1]) for sentence in dataset.get_train_sentences for ngram in ngrams(sentence) if ngram[1] in embeddings)
    save_examples(examples, path, 'examples')
    # examples = load_examples(path+'examples.pkl')
    na_dataset = PerClassDataset(examples)

    # Validation part
    valid_examples = set((ngram, ngram[1]) for sentence in dataset.get_valid_sentences for ngram in ngrams(sentence) if ngram[1] not in na_dataset and ngram[1] in embeddings)
    save_examples(valid_examples, path, 'valid_examples')
    valid_dataset = PerClassDataset(valid_examples)

    tr_val_dataset = na_dataset | valid_dataset

    # Test part
    test_examples = set((ngram, ngram[1]) for sentence in dataset.get_test_sentences for ngram in ngrams(
        sentence) if ngram[1] not in tr_val_dataset and ngram[1] in embeddings)
    save_examples(test_examples, path, 'test_examples')
    test_dataset = PerClassDataset(test_examples)

    # valid_examples = load_examples(path+'valid_examples.pkl')
    # test_examples = load_examples(path+'test_examples.pkl')
    save_examples(test_examples | valid_examples, path, 'valid_test_examples')

    # OOV part
    all_sentences = dataset.get_train_sentences + dataset.get_valid_sentences + dataset.get_test_sentences
    oov_examples = set((ngram, ngram[1]) for sentence in all_sentences for ngram in ngrams(sentence) if ngram[1] not in embeddings)
    save_examples(oov_examples, path, 'oov_examples')

    # Augmented part
    all_dataset = tr_val_dataset | test_dataset
    filter_cond = lambda label: label not in all_dataset

    augmented_examples = augment_data(examples, embeddings_path, filter_cond=filter_cond, topn=topn, min_cos_sim=min_cos_sim)    
    augmented_examples |= examples  # Union
    save_examples(augmented_examples, path, 'augmented_examples_topn{topn}_cos_sim{cs}'.format(topn=topn, cs=min_cos_sim))


def create_oov(dataset, embeddings_path):
    sentences = dataset.get_train_sentences + dataset.get_valid_sentences + dataset.get_test_sentences
    
    embeddings = load_embeddings(embeddings_path)

    oov = sorted(set(word for sentence in sentences for word in sentence if word not in embeddings))

    filepath = './' + dataset.dataset_name + '/oov.txt'
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write('\n'.join(oov)+'\n')


def compute_statistics(dataset, relative_threshold):
    path = './' + dataset.dataset_name + '/examples/'
    
    na_ex = load_examples(path+'examples.pkl')
    na_train = PerClassDataset(na_ex)

    a_ex = load_examples(path+'augmented_examples_topn5_cos_sim0.6.pkl')
    a_train = PerClassDataset(a_ex)
    threshold = int(a_train.stats()['most common labels number of examples'] / relative_threshold)

    valid_ex = load_examples(path+'valid_examples.pkl')
    valid = PerClassDataset(valid_ex)

    test_ex = load_examples(path+'test_examples.pkl')
    test = PerClassDataset(test_ex)

    val_test_ex = load_examples(path+'valid_test_examples.pkl')
    val_test = PerClassDataset(val_test_ex)

    oov_ex = load_examples(path+'oov_examples.pkl')
    oov = PerClassDataset(oov_ex)

    stats = {'non-augmented':na_train.stats(),
             'augmented':a_train.stats(threshold),
             'valid':valid.stats(),
             'test':test.stats(),
             'valid_test':val_test.stats(),
             'oov':oov.stats(),
             'sentences':{'train':len(dataset.get_train_sentences),
                          'valid':len(dataset.get_valid_sentences),
                          'test':len(dataset.get_test_sentences)
                         }
            }
    
    filepath = './'+dataset.dataset_name+'/'
    os.makedirs(filepath, exist_ok=True)
    with open(filepath + 'statistics.json', 'w') as file:
        json.dump(stats, file, indent=4)


if __name__ == '__main__':
    d = 50
    topn = 5
    min_cos_sim = .6
    relative_threshold = 100
    embeddings_path = './glove_embeddings/glove.6B.50d.txt'

    # Create a list of OOV first
    for dataset in [CoNLL(),
                    Sentiment(),
                    SemEval(),
                    ]:
        create_oov(dataset, embeddings_path)
    
    # Create the examples
    for dataset in [CoNLL(),
                    Sentiment(),
                    SemEval(),
                    ]:
        preprocess_data(dataset, embeddings_path, topn, min_cos_sim)
        compute_statistics(dataset, relative_threshold)
