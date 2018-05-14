import os
import logging

from gensim.models import KeyedVectors
from tqdm import tqdm

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

from utils import load_embeddings, load_examples
from utils import collate_fn, collate_x
from per_class_dataset import *
import pickle as pkl


def prepare_data(dataset,
                 embeddings,
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
    path = './data/' + dataset.dataset_name + '/examples/'
    if data_augmentation:
        examples = load_examples(path + 'augmented_examples_topn5_cos_sim0.6.pkl')
    else:
        examples = load_examples(path + 'examples')

    transform = vectorizer.vectorize_unknown_example

    def target_transform(y):
        return embeddings[y]

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
            over_population_threshold = int(
                train_valid_dataset.stats()['most common labels number of examples'] / over_population_threshold)

        def filter_labels_cond(label, N):
            return N <= over_population_threshold

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
    test_examples = set((ngram, ngram[1]) for sentence in test_sentences for ngram in ngrams(sentence, n) if ngram[1] in test_vocabs)
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
