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
    truncated_examples = set()
    for (CL, word, CR), label in examples:
        truncated_examples.add(((CL[-m:], word, CR[:m]), word))

    return sorted(truncated_examples)


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
                 debug_mode=False,
                 verbose=True,
                 no_number=False
                 ):
    # Train-validation part
    path = './data/' + dataset.dataset_name + '/examples/'

    # Paths to example files
    examples_path_file = path + 'examples{}.pkl'
    augmented_path_file = path + 'augmented_examples_topn5_cos_sim0.6{}.pkl'
    test_path_file = path + "valid_test_examples{}.pkl"
    if no_number:
        examples_path_file = examples_path_file.format("_nonumbers")
        augmented_path_file = augmented_path_file.format("_nonumbers")
        test_path_file = test_path_file.format("_nonumbers")
    else:
        examples_path_file = examples_path_file.format("")
        augmented_path_file = augmented_path_file.format("")
        test_path_file = test_path_file.format("")

    if data_augmentation:
        examples = load_examples(augmented_path_file)
    else:
        examples = load_examples(examples_path_file)
    if debug_mode:
        examples = list(examples)[:128]

    examples = truncate_examples(examples, n)

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
    test_examples = load_examples(test_path_file)
    test_examples = truncate_examples(test_examples, n)
    test_dataset = PerClassDataset(dataset=test_examples,
                                   transform=transform)
    test_loader = PerClassLoader(dataset=test_dataset,
                                 collate_fn=collate_x,
                                 k=-1,
                                 shuffle=False,
                                 batch_size=64,
                                 use_gpu=use_gpu)

    if verbose:
        logging.info('Number of unique examples: {}'.format(len(examples)))

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
