import argparse
import logging
import os

from data_loaders import CoNLLDataLoader, SentimentDataLoader, SemEvalDataLoader, NewsGroupDataLoader

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

from comick import ComickDev
from utils import save_embeddings
from utils import square_distance, cosine_sim
from utils import make_vocab, WordsInContextVectorizer
from utils import collate_fn, collate_x
from data_preparation import *
from per_class_dataset import *
from downstream_task.part_of_speech.train import train as train_pos
from downstream_task.named_entity_recognition.train import train as train_ner
from downstream_task.sentiment_classification.train import train as train_sent
from downstream_task.chunking.train import train as train_chunk
from downstream_task.semeval.train import train as train_semeval
from downstream_task.newsgroup_classification.train import train as train_newsgroup

import numpy as np
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
from pytoune import torch_to_numpy, tensors_to_variables
from pytoune.framework import Model
from pytoune.framework.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from torch.optim import Adam

from gensim.models import KeyedVectors


<<<<<<< HEAD
def load_data(d, corpus, verbose=True):
    path_embeddings = './data/conll_embeddings_settings/setting1/glove/train/glove.6B.{}d.txt'.format(d)
    embeddings = load_embeddings(path_embeddings)

    train_sentences = parse_conll_file('./data/conll/train.txt')
    valid_sentences = parse_conll_file('./data/conll/valid.txt')
    test_sentences = parse_conll_file('./data/conll/test.txt')

    if verbose:
        logging.info('Loading ' + str(d) + 'd embeddings from: "' + path_embeddings + '"')

    return embeddings, (train_sentences, valid_sentences, test_sentences)


def augment_data(examples, embeddings_path):

    logging.info("Loading embedding model...")
    word2vec_model = KeyedVectors.load_word2vec_format(embeddings_path)
    logging.info("Done.")

    labels = sorted(set(label for x, label in examples))

    logging.info("Getting new examples for {} labels...".format(len(labels)))
    new_examples = dict()
    for (left_context, word, right_context), label in examples:
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
        preprocessed_word = preprocess_token(word)
        preprocessed_examples.append((
            (left_context, preprocessed_word, right_context),
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
                 over_population_threshold=None,
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
        augmented_examples = augment_data(examples, './data/glove_embeddings/glove.6B.{}d.txt'.format(d))
        if verbose:
            logging.info("Number of non-augmented examples: {}".format(len(examples)))
        examples.update(augmented_examples)  # Union
    examples = preprocess_examples(examples)

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
        logging.info('Number of unique examples wo embeds:'.format(len(examples_without_embeds)))

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

        logging.info('\nFor training, loading ' + str(k) + ' examples per label per epoch.')

    return train_loader, valid_loader, test_loader


=======
>>>>>>> b718048a19c7def15c726f25ba2965a8bb0cc904
def train(model, model_name, train_loader, valid_loader, epochs=1000):
    # Create callbacks and checkpoints
    lrscheduler = ReduceLROnPlateau(patience=3)
    early_stopping = EarlyStopping(patience=10)
    model_path = './models/'

    os.makedirs(model_path, exist_ok=True)
    ckpt_best = ModelCheckpoint(model_path + 'best_' + model_name + '.torch',
                                save_best_only=True,
                                restore_best=True,
                                temporary_filename=model_path + 'temp_best_' + model_name + '.torch')

    ckpt_last = ModelCheckpoint(model_path + 'last_' + model_name + '.torch',
                                temporary_filename=model_path + 'temp_last_' + model_name + '.torch')

    logger_path = './train_logs/'
    os.makedirs(logger_path, exist_ok=True)
    csv_logger = CSVLogger(logger_path + model_name + '.csv')

    callbacks = [lrscheduler, ckpt_best, ckpt_last, early_stopping, csv_logger]

    # Fit the model
    model.fit_generator(train_loader, valid_loader,
                        epochs=epochs, callbacks=callbacks)


def predict_mean_embeddings(model, loader):
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

    mean_pred_embeddings = {}
    for label in predicted_embeddings:
        mean_pred_embeddings[label] = np.mean(
            np.array(predicted_embeddings[label]), axis=0)
    return mean_pred_embeddings


def evaluate(model, test_loader, test_embeddings, save=True, model_name=None):
    mean_pred_embeddings = predict_mean_embeddings(model, test_loader)

    if save:
        if model_name == None:
            raise ValueError('A filename should be provided.')
        save_embeddings(mean_pred_embeddings, model_name)

    predicted_results = {}

    def norm(y_true, y_pred):
        return np.linalg.norm(y_pred - y_true)

    euclidean_distances = []

    def cos_sim(y_true, y_pred):
        return float(cosine_similarity(y_pred, y_true))

    cos_sims = []
    nb_of_pred = 0
    for label in mean_pred_embeddings:
        if label in test_embeddings:
            y_pred = mean_pred_embeddings[label].reshape(1, -1)
            y_true = test_embeddings[label].reshape(1, -1)
            euclidean_distances.append(norm(y_true, y_pred))
            cos_sims.append(cos_sim(y_true, y_pred))
            nb_of_pred += 1

    logging.info('\nResults on the test:')
    logging.info('Mean euclidean dist: {}'.format(np.mean(euclidean_distances)))
    logging.info('Variance of euclidean dist: {}'.format(np.std(euclidean_distances)))
    logging.info('Mean cosine sim: {}'.format(np.mean(cos_sims)))
    logging.info('Variance of cosine sim: {}'.format(np.std(cos_sims)))
    logging.info('Number of labels evaluated: {}'.format(nb_of_pred))
    return mean_pred_embeddings


def get_data_loader(task, debug_mode, embedding_dimension):
    if task == 'ner':
        return CoNLLDataLoader(debug_mode, embedding_dimension)
    else:
        raise NotImplementedError("Task {} as no suitable data loader".format(task))


def main(model_name, task_config, n=41, k=1, device=0, d=100, epochs=100):
    # Global parameters
    debug_mode = False
    verbose = True
    save = True
    freeze_word_embeddings = True
    over_population_threshold = 100
    relative_over_population = True
    data_augmentation = True
    if debug_mode:
        data_augmentation = False
        over_population_threshold = None

    logging.info("Task name: {}".format(task_config['name']))
    logging.info("Debug mode: {}".format(debug_mode))
    logging.info("Verbose: {}".format(verbose))
    logging.info("Freeze word embeddings: {}".format(freeze_word_embeddings))
    logging.info("Over population threshold: {}".format(over_population_threshold))
    logging.info("Relative over population: {}".format(relative_over_population))
    logging.info("Data augmentation: {}".format(data_augmentation))

    use_gpu = torch.cuda.is_available()
    use_gpu = False
    if use_gpu:
        cuda_device = device
        torch.cuda.set_device(cuda_device)
        logging.info('Using GPU')

    # Load data
    dataloader = task_config['dataloader'](debug_mode, d)
    train_sentences = dataloader.get_train_sentences
    valid_sentences = dataloader.get_valid_sentences
    test_sentences = dataloader.get_test_sentences
    embeddings = dataloader.get_embeddings
    test_embeddings = dataloader.get_test_embeddings
    test_vocabs = dataloader.get_test_vocab
    all_sentences = train_sentences + valid_sentences + test_sentences

    # Prepare vectorizer
    word_to_idx, char_to_idx = make_vocab(all_sentences)
    vectorizer = WordsInContextVectorizer(word_to_idx, char_to_idx)
    vectorizer = vectorizer

    # Prepare examples
    train_loader, valid_loader, test_loader = prepare_data(
        embeddings=embeddings,
        test_vocabs=test_vocabs,
        train_sentences=train_sentences,
        test_sentences=all_sentences,
        vectorizer=vectorizer,
        n=n,
        use_gpu=use_gpu,
        k=k,
        over_population_threshold=over_population_threshold,
        relative_over_population=relative_over_population,
        data_augmentation=data_augmentation,
        verbose=verbose,
    )

    # Initialize training parameters
    lr = 0.001
    if debug_mode:
        model_name = 'testing_' + model_name
        save = False
        epochs = 3

    # Create the model
    net = ComickDev(
        characters_vocabulary=char_to_idx,
        words_vocabulary=word_to_idx,
        characters_embedding_dimension=20,
        word_embeddings_dimension=d,
        words_embeddings=embeddings,
        context_dropout_p=0.5,
        fc_dropout_p=0.5,
        freeze_word_embeddings=freeze_word_embeddings
    )
    model = Model(
        model=net,
        optimizer=Adam(net.parameters(), lr=lr),
        loss_function=square_distance,
        metrics=[cosine_sim],
    )
    if use_gpu:
        model.cuda()

    # Set up the callbacks and train
    train(
        model, model_name,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=epochs,
    )

    predicted_evaluation_embeddings = evaluate(
        model,
        test_loader=test_loader,
        test_embeddings=test_embeddings,
        save=save,
        model_name=model_name + '.txt'
    )

    # Override embeddings with the training ones
    # Make sure we only have embeddings from the corpus data
    logging.info("Evaluating embeddings...")
    predicted_evaluation_embeddings.update(embeddings)

    for task in task_config['tasks']:
        logging.info("Using predicted embeddings on {} task...".format(task['name']))
        task['script'](predicted_evaluation_embeddings, task['name'] + "_" + model_name, device)


def get_tasks_configs():
    return [
        # {
        #     'name': 'newsgroup',
        #     'dataloader': NewsGroupDataLoader,
        #     'tasks': [
        #         {
        #             'name': 'newsgroup',
        #             'script': train_newsgroup
        #         },
        #     ]
        # },
        {
            'name': 'conll',
            'dataloader': CoNLLDataLoader,
            'tasks': [
                {
                    'name': 'ner',
                    'script': train_ner
                },
                {
                    'name': 'pos',
                    'script': train_pos
                },
                {
                    'name': 'chunk',
                    'script': train_chunk
                },
            ]
        },
        {
            'name': 'semeval',
            'dataloader': SemEvalDataLoader,
            'tasks': [
                {
                    'name': 'semeval',
                    'script': train_semeval
                }
            ]
        },
        {
            'name': 'sent',
            'dataloader': SentimentDataLoader,
            'tasks': [
                {
                    'name': 'sent',
                    'script': train_sent
                },
            ]
        },
    ]


if __name__ == '__main__':
    from time import time

    t = time()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("n", default=3, nargs='?')
        parser.add_argument("k", default=1, nargs='?')
        parser.add_argument("device", default=0, nargs='?')
        parser.add_argument("d", default=100, nargs='?')
        parser.add_argument("e", default=100, nargs='?')
        parser.add_argument("t", default='ner', nargs='?')
        args = parser.parse_args()
        n = int(args.n)
        k = int(args.k)
        device = int(args.device)
        d = int(args.d)
        epochs = int(args.e)
        task = args.t
        if d not in [50, 100, 200, 300]:
            raise ValueError(
                "The embedding dimension 'd' should of 50, 100, 200 or 300.")
        logger = logging.getLogger()
        for i in range(5):
            # Control of randomization
            seed = 42 + i  # "Seed" of light
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            for task_config in get_tasks_configs():
                model_name = '{}_{}_n{}_k{}_d{}_i{}_e{}'.format('comick', task_config['name'], n, k, d, i, epochs)
                handler = logging.FileHandler('{}.log'.format(model_name))
                logger.addHandler(handler)
                main(model_name, task_config, n=n, k=k, device=device, d=d, epochs=epochs)
                logger.removeHandler(handler)
    except:
        logging.info('Execution stopped after {:.2f} seconds.'.format(time() - t))
        raise
    logging.info('Execution completed in {:.2f} seconds.'.format(time() - t))
