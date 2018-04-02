import argparse
import os
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

from comick import Comick, LRComick
from utils import load_embeddings, parse_conll_file, cosine_sim, cosine_distance
from utils import square_distance
from utils import make_vocab, WordsInContextVectorizer, ngrams
from utils import collate_fn
from per_class_dataset import *

import numpy
from pytoune.framework import Model
from pytoune.framework.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from torch.optim import Adam

def load_data(d):
    path_embeddings = './embeddings_settings/setting1/1_glove_embeddings/glove.6B.{}d.txt'.format(d)
    try:
        train_embeddings = load_embeddings(path_embeddings)
    except:
        if d == 50:
            path_embeddings = './embeddings/train_embeddings.txt'
            train_embeddings = load_embeddings(path_embeddings)
            print('Loading {}d embeddings from: "' + path_embeddings + '"'.format(d))
        else:
            raise
    print('Loading ' + str(d) + 'd embeddings from: "' + path_embeddings + '"')

    train_sentences = parse_conll_file('./conll/train.txt')
    valid_sentences = parse_conll_file('./conll/valid.txt')
    test_sentences = parse_conll_file('./conll/test.txt')

    return train_embeddings, (train_sentences, valid_sentences, test_sentences)

def prepare_data(embeddings,
                 sentences,
                 vectorizer,
                 n=15,
                 ratio=.8,
                 use_gpu=False,
                 k=1,
                 over_population_threshold=None,
                 verbose=True):

    examples = set()
    examples_without_embeds = set()
    for sentence in sentences:
        for ngram in ngrams(sentence, n):
            if ngram[1] in embeddings:
                examples.add((ngram, ngram[1])) # Keeps only different ngrams which have a training embeddings
            else:
                examples_without_embeds.add((ngram, ngram[1]))

    transform = vectorizer
    target_transform = lambda y: embeddings[y]

    dataset = PerClassDataset(
        examples,
        transform=transform,
        target_transform=target_transform,
    )
    if over_population_threshold != None:
        dataset.filter_labels(lambda label, N: N <= over_population_threshold)

    train_dataset, valid_dataset = dataset.split(ratio=.8, shuffle=True, reuse_label_mappings=False)

    train_loader = PerClassLoader(dataset=train_dataset,
                                  collate_fn=collate_fn,
                                  batch_size=1,
                                  k=k,
                                  use_gpu=use_gpu)
    valid_loader = PerClassLoader(dataset=valid_dataset,
                                  collate_fn=collate_fn,
                                  batch_size=16,
                                  k=-1,
                                  use_gpu=use_gpu)

    if verbose:
        print('Number of unique examples:', len(examples))
        print('Number of unique examples wo embeds:', len(examples_without_embeds))
        stats = dataset.stats()
        for stats, value in stats.items():
            print(stats+': '+str(value))
    
        print('Datasets size - Train:', len(train_dataset), 'Valid:', len(valid_dataset))
        print('Datasets labels - Train:', len(train_dataset.dataset), 'Valid:', len(valid_dataset.dataset))

    return train_loader, valid_loader


def train(model_name, char_to_idx, word_to_idx, train_embeddings, d, use_gpu=False, freeze_word_embeddings=False):
    
    # Create the model
    net = Comick(characters_vocabulary=char_to_ix,
                 words_vocabulary=word_to_idx,
                 characters_embedding_dimension=20,
                 characters_hidden_state_dimension=50,
                 word_embeddings_dimension=d,
                 words_hidden_state_dimension=50,
                 words_embeddings=train_embeddings,
                 freeze_word_embeddings=freeze_word_embeddings)

    model = Model(model=net,
                  optimizer=Adam(net.parameters(), lr=0.001),
                  loss_function=square_distance,
                  metrics=[cosine_sim])

    if use_gpu:
        model.cuda()

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
    model.fit_generator(train_loader, valid_loader, epochs=1000, callbacks=callbacks)

    return model

def evaluate():
    pass


def main(n=41, k=1, device=0, d=50):

    seed = 299792458  # "Seed" of light
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cuda_device = device
        torch.cuda.set_device(cuda_device)
        print('Using GPU')

    # Load data
    train_embeddings, sentences = load_data(d)
    train_sentences, valid_sentences, test_sentences = sentences
    all_sentences = sentences[0] + sentences[1] + sentences[2]

    word_to_idx, char_to_idx = make_vocab(all_sentences)

    # Prepare our examples
    vectorizer = WordsInContextVectorizer(word_to_idx, char_to_idx)
    train_loader, valid_loader = prepare_data(
        embeddings=train_embeddings,
        sentences=train_sentences,
        vectorizer=vectorizer.vectorize_unknown_example_merged_context,
        n=n,
        use_gpu=use_gpu,
        k=k,
        over_population_threshold=80
    )
    
    # Create and train the model
    model_name = 'comick_n{}_k{}_d{}_unique_context'.format(n, k, d)
    train(model_name, char_to_idx, word_to_idx, train_embeddings,
          d=d,
          use_gpu=use_gpu,
          freeze_word_embeddings=False)


if __name__ == '__main__':
    from time import time

    t = time()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("n", default=7, nargs='?')
        parser.add_argument("k", default=1, nargs='?')
        parser.add_argument("device", default=0, nargs='?')
        parser.add_argument("d", default=100, nargs='?')
        args = parser.parse_args()
        n = int(args.n)
        k = int(args.k)
        device = int(args.device)
        d = int(args.d)
        if d not in [50, 100, 200, 300]:
            raise ValueError("The embedding dimension 'd' should of 50, 100, 200 or 300.")
        main(n=n, k=k, device=device, d=d)
    except:
        print('Execution stopped after {:.2f} seconds.'.format(time() - t))
        raise
    print('Execution completed in {:.2f} seconds.'.format(time() - t))
