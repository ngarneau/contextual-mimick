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
import pickle as pkl


def augment_data(examples, embeddings_path, topn=5, min_cos_sim=.6):
    logging.info("Loading embedding model...")
    word2vec_model = KeyedVectors.load_word2vec_format(embeddings_path)
    logging.info("Done.")

    labels = sorted(set(label for x, label in examples))
    logging.info(
        "Computing similar words for {} labels...".format(len(labels)))
    sim_words = dict()
    for label in tqdm(labels):
        sim_words[label] = word2vec_model.most_similar(label, topn=topn)
    logging.info("Done.")

    new_examples = set()
    for (left_context, word, right_context), label in tqdm(examples):
        sim_words_for_label = sim_words[word]
        for sim_word, cos_sim in sim_words_for_label:
            # Add new labels, not new examples to already existing labels.
            if sim_word not in labels and cos_sim >= min_cos_sim:
                new_example = (
                    (left_context, sim_word, right_context), sim_word)
                new_examples.add(new_example)
    logging.info("Done.")
    return new_examples


def preprocess_data(dataset,
                    embeddings_path,
                    topn,
                    min_cos_sim):
    path =  './' + dataset.dataset_name + '/examples/'
    embeddings = load_embeddings(embeddings_path)

    # Training part
    examples = set((ngram, ngram[1]) for sentence in dataset.get_train_sentences for ngram in ngrams(sentence) if ngram[1] in embeddings)
    save_examples(examples, path, 'examples')

    augmented_examples = augment_data(examples, embeddings_path, topn=topn, min_cos_sim=min_cos_sim)    
    augmented_examples |= examples  # Union
    save_examples(augmented_examples, path, 'augmented_examples_topn{topn}_cos_sim{cs}'.format(topn=topn, cs=min_cos_sim))
    # augmented_examples = load_examples(path+'augmented_examples_topn5_cos_sim0.6.pkl')

    # Validation part
    valid_examples = set((ngram, ngram[1]) for sentence in dataset.get_valid_sentences for ngram in ngrams(sentence) if ngram[1] not in augmented_examples and ngram[1] in embeddings)
    save_examples(valid_examples, path, 'valid_examples')

    tr_val_ex = augmented_examples | valid_examples

    # Test part
    test_examples = set((ngram, ngram[1]) for sentence in dataset.get_test_sentences for ngram in ngrams(sentence) if ngram[1] not in tr_val_ex and ngram[1] in embeddings)
    save_examples(test_examples, path, 'test_examples')


def create_oov(dataset, embeddings_path):
    sentences = dataset.get_train_sentences + dataset.get_valid_sentences + dataset.get_test_sentences
    
    embeddings = load_embeddings(embeddings_path)

    oov = sorted(set(word for sentence in sentences for word in sentence if word not in embeddings))

    filepath = './' + dataset.dataset_name + '/oov.txt'
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write('\n'.join(oov)+'\n')


if __name__ == '__main__':
    d = 50
    topn = 5
    min_cos_sim = .6
    embeddings_path = './glove_embeddings/glove.6B.50d.txt'
    for dataset in [CoNLL(),
                    Sentiment(),
                    SemEval()]:
        preprocess_data(dataset, embeddings_path, topn, min_cos_sim)
        create_oov(dataset, embeddings_path,)
