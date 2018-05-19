import os
import argparse
import logging
import pickle
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pytoune import torch_to_numpy, tensors_to_variables
from pytoune.framework import Model
from pytoune.framework.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from torch.optim import Adam

from data.dataset_manager import CoNLL, Sentiment, SemEval
from data_preparation import prepare_data
from evaluation.intrinsic_evaluation import evaluate, predict_mean_embeddings, Evaluator
from per_class_dataset import *

from comick import ComickDev, ComickUniqueContext, LRComick, LRComickContextOnly, TheFinalComick

from utils import load_embeddings
from utils import square_distance, cosine_sim
from utils import make_vocab, WordsInContextVectorizer
from utils import collate_fn, collate_x

from downstream_task.part_of_speech.train import train as train_pos
from downstream_task.named_entity_recognition.train import train as train_ner
from downstream_task.sentiment_classification.train import train as train_sent
from downstream_task.chunking.train import train as train_chunk
from downstream_task.semeval.train import train as train_semeval


def train(model, model_name, train_loader, valid_loader, epochs=1000):
    # Create callbacks and checkpoints
    lrscheduler = ReduceLROnPlateau(patience=3)
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    model_path = './models/'

    os.makedirs(model_path, exist_ok=True)
    ckpt_best = ModelCheckpoint(model_path + 'best_' + model_name + '.torch',
                                save_best_only=True,
                                restore_best=True,
                                temporary_filename=model_path + 'temp_best_' + model_name + '.torch',
                                verbose=True,
                                )

    ckpt_last = ModelCheckpoint(model_path + 'last_' + model_name + '.torch',
                                temporary_filename=model_path + 'temp_last_' + model_name + '.torch')

    logger_path = './train_logs/'
    os.makedirs(logger_path, exist_ok=True)
    csv_logger = CSVLogger(logger_path + model_name + '.csv')

    callbacks = [
        lrscheduler,
        ckpt_best,
        ckpt_last,
        early_stopping,
        csv_logger
    ]

    # Fit the model
    model.fit_generator(train_loader, valid_loader,
                        epochs=epochs, callbacks=callbacks)


def main(task_config, n=21, k=2, device=0, d=100, epochs=100):
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
    # use_gpu = False
    if use_gpu:
        cuda_device = device
        torch.cuda.set_device(cuda_device)
        logging.info('Using GPU')

    # Load dataset
    dataset = task_config['dataset'](debug_mode, relative_path='./data/')
    
    all_sentences = dataset.get_train_sentences + dataset.get_valid_sentences + dataset.get_test_sentences

    word_embeddings = load_embeddings('./data/glove_embeddings/glove.6B.{}d.txt'.format(d))
    chars_embeddings = load_embeddings('./predicted_char_embeddings/char_mimick_glove_d100_c20')

    # Prepare vectorizer
    word_to_idx, char_to_idx = make_vocab(all_sentences)
    vectorizer = WordsInContextVectorizer(word_to_idx, char_to_idx)
    vectorizer = vectorizer


    # Initialize training parameters
    model_name = '{}_n{}_k{}_d{}_e{}'.format(task_config['name'], n, k, d, epochs)
    lr = 0.001
    if debug_mode:
        model_name = 'testing_' + model_name
        save = False
        epochs = 3

    # Create the model
    net = TheFinalComick(
        characters_vocabulary=char_to_idx,
        words_vocabulary=word_to_idx,
        characters_embedding_dimension=20,
        word_embeddings_dimension=d,
        words_embeddings=word_embeddings,
        chars_embeddings=chars_embeddings,
        freeze_word_embeddings=freeze_word_embeddings,
        freeze_mimick=True,
        mimick_model_path='./models/best_Pinter_mimick_glove_d100_c20.torch',
        use_gpu=use_gpu,
        lstm_dropout=0
    )
    model_name = "{}_{}_v{}".format(model_name, net.__class__.__name__.lower(), net.version)
    handler = logging.FileHandler('{}.log'.format(model_name))
    logger.addHandler(handler)

    model = Model(
        model=net,
        optimizer=Adam(net.parameters(), lr=lr),
        loss_function=square_distance,
        metrics=[cosine_sim],
    )
    if use_gpu:
        model.cuda()

    # Prepare examples
    train_loader, valid_loader, test_loader, oov_loader = prepare_data(
        dataset=dataset,
        embeddings=word_embeddings,
        vectorizer=vectorizer,
        n=n,
        use_gpu=use_gpu,
        k=k,
        over_population_threshold=over_population_threshold,
        relative_over_population=relative_over_population,
        data_augmentation=data_augmentation,
        debug_mode=debug_mode,
        verbose=verbose,
    )

    # Set up the callbacks and train
    train(
        model, model_name,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=epochs,
    )

    # test_embeddings = evaluate(
    #     model,
    #     test_loader=test_loader,
    #     test_embeddings=word_embeddings,
    #     save=save,
    #     model_name=model_name + '.txt'
    # )

    intrinsic_results = Evaluator(model,
                                  test_loader,
                                  idx_to_word={v: k for k, v in word_to_idx.items()},
                                  idx_to_char={v: k for k, v in char_to_idx.items()},
                                  word_embeddings=word_embeddings)
    for k, v in intrinsic_results.global_results.items():
        logging.info("{} {}".format(k, v))
    
    results_pathfile = './evaluation/intrinsic/'
    fname = 'intrinsic_{}.pkl'.format(model_name)
    os.makedirs(results_pathfile, exist_ok=True)
    pickle.dump(intrinsic_results, open(results_pathfile + fname, 'wb'))
    
    # oov_results = Evaluator(model, oov_loader)
    # predicted_oov_embeddings = oov_results.get_mean_predicted_embeddings()
    predicted_oov_embeddings = predict_mean_embeddings(model, oov_loader)

    # Override embeddings with the training ones
    # Make sure we only have embeddings from the corpus data
    # logging.info("Evaluating embeddings...")
    predicted_oov_embeddings.update(word_embeddings)

    for task in task_config['tasks']:
        logging.info("Using predicted embeddings on {} task...".format(task['name']))
        task['script'](predicted_oov_embeddings, task['name'] + "_" + model_name, device, debug_mode)
    logger.removeHandler(handler)


def get_tasks_configs():
    return [
        {
            'name': 'conll',
            'dataset': CoNLL,
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
            'name': 'sent',
            'dataset': Sentiment,
            'tasks': [
                {
                    'name': 'sent',
                    'script': train_sent
                },
            ]
        },
        {
            'name': 'semeval',
            'dataset': SemEval,
            'tasks': [
                {
                    'name': 'semeval',
                    'script': train_semeval
                }
            ]
        },
    ]


if __name__ == '__main__':
    from time import time
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    t = time()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("n", default=5, nargs='?')
        parser.add_argument("k", default=2, nargs='?')
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
        for task_config in get_tasks_configs():
            main(task_config, n=n, k=k, device=device, d=d, epochs=epochs)
    except:
        logging.info('Execution stopped after {:.2f} seconds.'.format(time() - t))
        raise
    logging.info('Execution completed in {:.2f} seconds.'.format(time() - t))
