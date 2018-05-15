import os
import logging

from utils import load_embeddings, load_examples
from utils import collate_fn, collate_x
from per_class_dataset import *
import pickle as pkl

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def truncate_examples(examples, n):
    m = n // 2
    truncated_examples = []
    for (CL, word, CR), label in examples:
        truncated_examples.append( ((CL[-m:], word, CR[:m]), word) )
    
    return truncated_examples


def prepare_data(dataset,
                 embeddings,
                 vectorizer,
                 n=15,
                 ratio=.8,
                 use_gpu=False,
                 k=1,
                 data_augmentation=False,
                 over_population_threshold=100,
                 relative_over_population=True,
                 verbose=True,
                 ):
    # Train-validation part
    path = './data/' + dataset.dataset_name + '/examples/'
    if data_augmentation:
        examples = load_examples(path+'augmented_examples_topn5_cos_sim0.6.pkl')
    else:
        examples = load_examples(path + 'examples.pkl')

    examples = truncate_examples(examples)

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
    test_examples = load_examples(path + 'valid_test_examples.pkl')
    test_examples = truncate_examples(test_examples)
    test_dataset = PerClassDataset(dataset=test_examples,
                                   transform=transform,
                                   target_transform=target_transform)
    test_loader = PerClassLoader(dataset=test_dataset,
                                 collate_fn=collate_fn,
                                 k=-1,
                                 batch_size=64,
                                 use_gpu=use_gpu)

    oov_examples = load_examples(path + 'oov_examples.pkl')
    oov_examples = truncate_examples(oov_examples)
    oov_dataset = PerClassDataset(dataset=oov_examples,
                                   transform=transform)
    oov_loader = PerClassLoader(dataset=oov_dataset,
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

    return train_loader, valid_loader, test_loader, oov_loader
