import os
import logging

from gensim.models import KeyedVectors
from tqdm import tqdm

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

from utils import load_embeddings, parse_conll_file, preprocess_token, ngrams
from utils import collate_fn, collate_x
from per_class_dataset import *
import pickle as pkl

def load_data(d, corpus, verbose=True):
    path_embeddings = './data/conll_embeddings_settings/setting1/glove/train/glove.6B.{}d.txt'.format(
        d)
    embeddings = load_embeddings(path_embeddings)

    train_sentences = parse_conll_file('./data/conll/train.txt')
    valid_sentences = parse_conll_file('./data/conll/valid.txt')
    test_sentences = parse_conll_file('./data/conll/test.txt')

    if verbose:
        logging.info('Loading ' + str(d) +
                     'd embeddings from: "' + path_embeddings + '"')

    return embeddings, (train_sentences, valid_sentences, test_sentences)


def augment_data(examples, embeddings_path):

    logging.info("Loading embedding model...")
    word2vec_model = KeyedVectors.load_word2vec_format(embeddings_path)
    logging.info("Done.")

    labels = sorted(set(label for x, label in examples))

    logging.info("Getting new examples for {} labels...".format(len(labels)))
    new_examples = dict()
    for (left_context, word, right_context), label in tqdm(examples):
        sim_words = word2vec_model.most_similar(label, topn=5)
        for sim_word, cos_sim in sim_words:
            # Add new labels, not new examples to already existing labels.
            if sim_word not in labels and cos_sim >= 0.6:
                new_example = ((left_context, sim_word, right_context), sim_word)
                new_examples[new_example] = 1
    logging.info("Done.")
    return new_examples


def preprocess_examples(examples):
    preprocessed_examples = list()
    for (left_context, word, right_context), label in examples:
        # preprocessed_word = preprocess_token(word)
        preprocessed_examples.append((
            (left_context, word, right_context),
            label
        ))
    return preprocessed_examples


def prepare_data(embeddings,
                 test_vocabs,
                 train_sentences,
                 test_sentences,
                 vectorizer,
                 n=15,
                 ratio=.8,
                 use_gpu=False,
                 k=1,
                 over_population_threshold=100,
                 relative_over_population=True,
                 verbose=True,
                 data_augmentation=False):
    # Train-validation part
    examples = dict()
    examples_without_embeds = dict()
    for sentence in train_sentences:
        for ngram in ngrams(sentence, n):
            key = (ngram, ngram[1])
            if ngram[1] in embeddings:
                # Keeps only different ngrams which have a training embeddings
                examples[key] = 1
            else:
                examples_without_embeds[key] = 1

    if data_augmentation:
        augmented_examples = augment_data(examples, './data/glove_embeddings/glove.6B.100d.txt')
        if verbose:
            logging.info(
                "Number of non-augmented examples: {}".format(len(examples)))
        examples.update(augmented_examples)  # Union
    examples = preprocess_examples(examples)

    transform = vectorizer.vectorize_unknown_example

    def target_transform(y): return embeddings[y]

    train_valid_dataset = PerClassDataset(
        examples,
        transform=transform,
        target_transform=target_transform,
    )

    train_dataset, valid_dataset = train_valid_dataset.split(
        ratio=.8, shuffle=True, reuse_label_mappings=False)

    filter_labels_cond = None
    if over_population_threshold != None:
        if relative_over_population:
            over_population_threshold = int(train_valid_dataset.stats()['most common labels number of examples']/over_population_threshold)
        def filter_labels_cond(label, N): return N <= over_population_threshold

    train_loader = PerClassLoader(dataset=train_dataset,
                                  collate_fn=collate_fn,
                                  batch_size=64,
                                  k=k,
                                  use_gpu=use_gpu,
                                  filter_labels_cond=filter_labels_cond)
    valid_loader = PerClassLoader(dataset=valid_dataset,
                                  collate_fn=collate_fn,
                                  batch_size=64,
                                  k=k,
                                  use_gpu=use_gpu,
                                  filter_labels_cond=filter_labels_cond)

    # Test part
    test_examples = set((ngram, ngram[1]) for sentence in test_sentences for ngram in ngrams(sentence, n) if
                        ngram[1] in test_vocabs)
    test_examples = preprocess_examples(test_examples)

    test_dataset = PerClassDataset(dataset=test_examples,
                                   transform=transform)
    test_loader = PerClassLoader(dataset=test_dataset,
                                 collate_fn=collate_x,
                                 k=-1,
                                 batch_size=64,
                                 use_gpu=use_gpu)

    if verbose:
        logging.info('Number of unique examples: {}'.format(len(examples)))
        logging.info('Number of unique examples wo embeds:'.format(
            len(examples_without_embeds)))

        logging.info('\nGlobal statistics:')
        stats = train_valid_dataset.stats()
        for stats, value in stats.items():
            logging.info(stats + ': ' + str(value))

        logging.info('\nStatistics on the training dataset:')
        stats = train_dataset.stats(over_population_threshold)
        for stats, value in stats.items():
            logging.info(stats + ': ' + str(value))

        logging.info('\nStatistics on the validation dataset:')
        stats = valid_dataset.stats(over_population_threshold)
        for stats, value in stats.items():
            logging.info(stats + ': ' + str(value))

        logging.info('\nStatistics on the test dataset:')
        stats = test_dataset.stats()
        for stats, value in stats.items():
            logging.info(stats + ': ' + str(value))

        logging.info('\nFor training, loading ' + str(k) +
                     ' examples per label per epoch.')

    return train_loader, valid_loader, test_loader
