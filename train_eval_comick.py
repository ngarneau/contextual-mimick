import argparse
import os
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

from comick import ComickUniqueContext, LRComick, ComickDev
from utils import load_embeddings, save_embeddings, parse_conll_file
from utils import square_distance, euclidean_distance, cosine_sim, cosine_distance
from utils import make_vocab, WordsInContextVectorizer, ngrams
from utils import collate_fn
from per_class_dataset import *

import numpy as np
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
from pytoune import torch_to_numpy, tensors_to_variables
from pytoune.framework import Model
from pytoune.framework.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from torch.optim import Adam


def load_data(d, verbose=True):
    path_embeddings = './embeddings_settings/setting1/1_glove_embeddings/glove.6B.{}d.txt'.format(
        d)
    try:
        embeddings = load_embeddings(path_embeddings)
    except:
        if d == 50:
            path_embeddings = './embeddings/train_embeddings.txt'
            embeddings = load_embeddings(path_embeddings)
        else:
            raise

    train_sentences = parse_conll_file('./conll/train.txt')
    valid_sentences = parse_conll_file('./conll/valid.txt')
    test_sentences = parse_conll_file('./conll/test.txt')

    if verbose:
        print('Loading ' + str(d) + 'd embeddings from: "' + path_embeddings + '"')

    return embeddings, (train_sentences, valid_sentences, test_sentences)


def augment_data(examples, embeddings):
    labels = set(label for x, label in examples)
    similar_words = pkl.load(open('similar_words.p', 'rb'))

    new_examples = set()
    for (left_context, word, right_context), label in examples:
        sim_words = similar_words[label]
        for sim_word, cos_sim in sim_words:
            # Add new labels, not new examples to already existing labels.
            if sim_word not in labels and cos_sim >= 0.6:
                new_example = (
                    (left_context, sim_word, right_context), sim_word)
                new_examples.add(new_example)

    return new_examples


def prepare_data(embeddings,
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
    examples = set()
    examples_without_embeds = set()
    for sentence in train_sentences:
        for ngram in ngrams(sentence, n):
            if ngram[1] in embeddings:
                # Keeps only different ngrams which have a training embeddings
                examples.add((ngram, ngram[1]))
            else:
                examples_without_embeds.add((ngram, ngram[1]))

    if data_augmentation:
        augmented_examples = augment_data(examples, embeddings)
        if verbose:
            print("Number of non-augmented examples:", len(examples))
        examples |= augmented_examples  # Union

    transform = vectorizer

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
        def filter_labels_cond(label, N): return N <= over_population_threshold
    train_loader = PerClassLoader(dataset=train_dataset,
                                  collate_fn=collate_fn,
                                  batch_size=1,
                                  k=k,
                                  use_gpu=use_gpu,
                                  filter_labels_cond=filter_labels_cond)
    valid_loader = PerClassLoader(dataset=valid_dataset,
                                  collate_fn=collate_fn,
                                  batch_size=64,
                                  k=-1,
                                  use_gpu=use_gpu,
                                  filter_labels_cond=filter_labels_cond)

    # Test part
    test_examples = set((ngram, ngram[1]) for sentence in test_sentences for ngram in ngrams(
        sentence, n) if ngram[1] not in train_valid_dataset)

    test_dataset = PerClassDataset(dataset=test_examples,
                                   transform=vectorizer)

    test_loader = PerClassLoader(dataset=test_dataset,
                                 collate_fn=collate_fn,
                                 k=-1,
                                 batch_size=64,
                                 use_gpu=use_gpu)

    if verbose:
        print('Number of unique examples:', len(examples))
        print('Number of unique examples wo embeds:',
              len(examples_without_embeds))

        print('\nGlobal statistics:')
        stats = train_valid_dataset.stats()
        for stats, value in stats.items():
            print(stats+': '+str(value))

        print('\nStatistics on the training dataset:')
        stats = train_dataset.stats(over_population_threshold)
        for stats, value in stats.items():
            print(stats+': '+str(value))

        print('\nStatistics on the validation dataset:')
        stats = valid_dataset.stats(over_population_threshold)
        for stats, value in stats.items():
            print(stats+': '+str(value))

        print('\nStatistics on the test dataset:')
        stats = test_dataset.stats()
        for stats, value in stats.items():
            print(stats+': '+str(value))

    return train_loader, valid_loader, test_loader


def train(model, model_name, train_loader, valid_loader, epochs=1000):

    # Create callbacks and checkpoints
    lrscheduler = ReduceLROnPlateau(patience=2)
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
    for label in predicted_embeddings.keys():
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

    def norm(y_true, y_pred): return np.linalg.norm(
        y_pred.reshape(1, -1)-y_true.reshape(1, -1))
    sum_norm = 0

    def cos_sim(y_true, y_pred): return float(
        cosine_similarity(y_pred.reshape(1, -1), y_true.reshape(1, -1)))
    sum_cos_sim = 0
    nb_of_pred = 0
    for label, idx in test_loader.dataset.labels_mapping.items():
        if label in test_embeddings and idx in mean_pred_embeddings:
            y_true = test_embeddings[label]
            y_pred = mean_pred_embeddings[idx]
            sum_norm += norm(y_true, y_pred)
            sum_cos_sim += cos_sim(y_true, y_pred)
            nb_of_pred += 1

    print('mean euclidean dist:', sum_norm/nb_of_pred,
          'mean cosine sim:', sum_cos_sim/nb_of_pred)


def main(n=41, k=1, device=0, d=50):
    # Control of randomization
    seed = 299792458  # "Seed" of light
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Global parameters
    debug_mode = False
    verbose = True
    save = True
    use_gpu = torch.cuda.is_available()
    # use_gpu = False
    if use_gpu:
        cuda_device = device
        torch.cuda.set_device(cuda_device)
        print('Using GPU')

    # Load data
    embeddings, sentences = load_data(d, verbose)
    train_sentences, valid_sentences, test_sentences = sentences
    if debug_mode:
        train_sentences = train_sentences[:100]
        valid_sentences = valid_sentences[:100]
        test_sentences = test_sentences[:100]
    all_sentences = train_sentences + valid_sentences + test_sentences

    # Prepare vectorizer
    word_to_idx, char_to_idx = make_vocab(all_sentences)
    vectorizer = WordsInContextVectorizer(word_to_idx, char_to_idx)
    vectorizer = vectorizer.vectorize_unknown_example

    # Prepare examples
    over_population_threshold = 80
    data_augmentation = True
    if debug_mode:
        data_augmentation = False
    train_loader, valid_loader, test_loader = prepare_data(
        embeddings=embeddings,
        train_sentences=train_sentences,
        test_sentences=valid_sentences,  # +test_sentences
        vectorizer=vectorizer,
        n=n,
        use_gpu=use_gpu,
        k=k,
        over_population_threshold=over_population_threshold,
        data_augmentation=data_augmentation,
        verbose=verbose,
    )

    # Initialize training parameters
    model_name = 'comick_n{}_k{}_d{}_unique_context'.format(n, k, d)
    epochs = 1000
    lr = 0.001
    freeze_word_embeddings = False
    if debug_mode:
        model_name = 'testing_' + model_name
        save = False
        epochs = 1

    # Create the model
    # net = LRComick(
    #     characters_vocabulary=char_to_idx,
    #     words_vocabulary=word_to_idx,
    #     characters_embedding_dimension=20,
    #     characters_hidden_state_dimension=50,
    #     word_embeddings_dimension=d,
    #     words_hidden_state_dimension=50,
    #     words_embeddings=embeddings,
    #     freeze_word_embeddings=freeze_word_embeddings,
    # )
    net = ComickDev(
        characters_vocabulary=char_to_idx,
        words_vocabulary=word_to_idx,
        characters_embedding_dimension=20,
        word_embeddings_dimension=d,
        words_embeddings=embeddings,
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

    evaluate(
        model,
        test_loader=test_loader,
        test_embeddings=embeddings,
        save=save,
        model_name=model_name + '.txt'
    )


if __name__ == '__main__':
    from time import time

    t = time()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("n", default=7, nargs='?')
        parser.add_argument("k", default=1, nargs='?')
        parser.add_argument("device", default=0, nargs='?')
        parser.add_argument("d", default=50, nargs='?')
        args = parser.parse_args()
        n = int(args.n)
        k = int(args.k)
        device = int(args.device)
        d = int(args.d)
        if d not in [50, 100, 200, 300]:
            raise ValueError(
                "The embedding dimension 'd' should of 50, 100, 200 or 300.")
        main(n=n, k=k, device=device, d=d)
    except:
        print('Execution stopped after {:.2f} seconds.'.format(time() - t))
        raise
    print('Execution completed in {:.2f} seconds.'.format(time() - t))
