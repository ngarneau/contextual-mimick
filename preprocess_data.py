import os
import logging

from gensim.models import KeyedVectors
from tqdm import tqdm

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

from utils import ngrams
from data_loaders import CoNLLDataLoader, SemEvalDataLoader, SentimentDataLoader
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
                    name,
                    topn,
                    min_cos_sim):
    path =  './data/'+name+'/examples/'

    # Training part
    examples = set((ngram, ngram[1]) for sentence in dataset.get_train_sentences for ngram in ngrams(sentence) if ngram[1] in dataset.get_embeddings)
    save_examples(examples, path, 'examples')

    augmented_examples = augment_data(examples, './data/glove_embeddings/glove.6B.{}d.txt'.format(d), topn=topn, min_cos_sim=min_cos_sim)    
    augmented_examples |= examples  # Union
    save_examples(augmented_examples, path, 'augmented_examples_topn{topn}_cos_sim{cs}'.format(topn, min_cos_sim))

    # Test part
    test_examples = set((ngram, ngram[1]) for sentence in dataset.get_test_sentences for ngram in ngrams(sentence) if ngram[1] in dataset.get_test_vocab)
    save_examples(test_examples, path, 'test_examples')


def save_examples(examples, path, filename):
    os.makedirs(path, exist_ok=True)
    with open(path + filename + '.pkl', 'wb') as file:
        pkl.dump(examples, file)


if __name__ == '__main__':
    d = 100
    topn = 5
    min_cos_sim = .6
    for dataset, name in [(CoNLLDataLoader(d), 'conll'),
                          (SentimentDataLoader(d), 'sentiment'), 
                          (SemEvalDataLoader(d), 'scienceie')]:
        preprocess_data(dataset, name, topn, min_cos_sim)

