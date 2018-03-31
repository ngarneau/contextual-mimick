import pickle
import argparse
import logging
import os

from contextual_mimick import get_contextual_mimick

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import numpy
from pytoune.framework import Model
from pytoune.framework.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from torch.optim import Adam

from utils import load_embeddings, parse_conll_file, cosine_sim
from utils import square_distance
from utils import make_vocab, WordsInContextVectorizer, ngrams
from utils import collate_examples
from per_class_dataset import *


def split_train_valid(examples, ratio):
    m = int(ratio * len(examples))
    train_examples, valid_examples = [], []
    sorted_examples = sorted(examples)
    numpy.random.shuffle(sorted_examples)
    for i, x in enumerate(sorted_examples):
        if i < m:
            train_examples.append(x)
        else:
            valid_examples.append(x)
    return train_examples, valid_examples


def prepare_data(embeddings, sentences, n=15, ratio=.8, use_gpu=False, k=1, word_to_idx=None, char_to_idx=None):
    vectorizer = WordsInContextVectorizer(word_to_idx, char_to_idx)
    similar_words = pickle.load(open('similar_words.p', 'rb'))

    examples = set((ngram, ngram[1]) for sentence in sentences for ngram in ngrams(sentence, n) if
                   ngram[1] in embeddings)  # Keeps only different ngrams which have a training embedding

    train_embeddings = load_embeddings('./embeddings_settings/setting2/1_glove_embeddings/glove.6B.50d.txt')
    new_examples = set()
    for example in examples:
        sws = similar_words[example[1]]
        for sw in sws:
            if sw[0] not in train_embeddings.keys():
                new_example = ((example[0][0], sw[0], example[0][2]), sw[0])
                new_examples.add(new_example)

    print("Number of unique examples before:", len(examples))
    examples = examples.union(new_examples)
    print('Number of unique examples:', len(examples))

    # train_examples, valid_examples = split_train_valid(examples, ratio)

    # filter_cond = lambda x, y: y in embeddings
    transform = vectorizer.vectorize_unknown_example
    target_transform = lambda y: embeddings[y]

    dataset = PerClassDataset(
        examples,
        transform=transform,
        target_transform=target_transform
    )
    dataset.filter_labels(lambda label, n: n < 80)
    train_dataset, valid_dataset = dataset.split(ratio=.8, shuffle=True, reuse_label_mappings=False)

    stats = dataset.stats(8)
    for stats, value in stats.items():
        print(stats+': '+str(value))
    
    print('Datasets size - Train:', len(train_dataset), 'Valid:', len(valid_dataset))
    print('Datasets labels - Train:', len(train_dataset.dataset), 'Valid:', len(valid_dataset.dataset))

    collate_fn = lambda samples: collate_examples([(*x, y) for x, y in samples])
    train_loader = PerClassLoader(dataset=train_dataset,
                                  collate_fn=collate_fn,
                                  batch_size=16,
                                  k=k,
                                  use_gpu=use_gpu)
    valid_loader = PerClassLoader(dataset=valid_dataset,
                                  collate_fn=collate_fn,
                                  batch_size=16,
                                  k=k,
                                  use_gpu=use_gpu)

    return train_loader, valid_loader, word_to_idx, char_to_idx


def main(n=41, k=1, device=0, d=50):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cuda_device = device
        torch.cuda.set_device(cuda_device)
        print('Using GPU')

    seed = 299792458  # "Seed" of light
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

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

    path_sentences = './conll/train.txt'
    train_sentences = parse_conll_file(path_sentences)
    # train_sentences = train_sentences[:100]
    all_sentences = list(train_sentences)
    all_sentences += parse_conll_file('./conll/valid.txt')
    all_sentences += parse_conll_file('./conll/test.txt')
    word_to_idx, char_to_idx = make_vocab(all_sentences)

    # Prepare our examples
    train_loader, valid_loader, word_to_idx, char_to_idx = prepare_data(
        embeddings=train_embeddings,
        sentences=train_sentences,
        n=n,
        ratio=.8,
        use_gpu=use_gpu,
        k=k,
        word_to_idx=word_to_idx,
        char_to_idx=char_to_idx
    )

    net = get_contextual_mimick(char_to_idx, word_to_idx, word_embedding_dim=d)

    if use_gpu:
        net.cuda()
    net.load_words_embeddings(train_embeddings)

    lrscheduler = ReduceLROnPlateau(patience=10)
    early_stopping = EarlyStopping(patience=10)
    model_path = './models/'
    model_name = 'comick_n{}_k{}_d{}'.format(n, k, d)
    os.makedirs(model_path, exist_ok=True)
    ckpt_best = ModelCheckpoint(model_path + 'best_' + model_name + '.torch',
                                save_best_only=True,
                                temporary_filename=model_path + 'temp_best_' + model_name + '.torch')

    ckpt_last = ModelCheckpoint(model_path + 'last_' + model_name + '.torch',
                                temporary_filename=model_path + 'temp_last_' + model_name + '.torch')

    logger_path = './train_logs/'
    os.makedirs(logger_path, exist_ok=True)
    csv_logger = CSVLogger(logger_path + model_name + '.csv')

    model = Model(model=net,
                  optimizer=Adam(net.parameters(), lr=0.001),
                  loss_function=square_distance,
                  metrics=[cosine_sim])
    callbacks = [
        lrscheduler,
        ckpt_best,
        ckpt_last,
        # early_stopping,
        csv_logger
    ]

    model.fit_generator(train_loader, valid_loader, epochs=1000, callbacks=callbacks)


if __name__ == '__main__':
    from time import time

    t = time()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("n", default=41, nargs='?')
        parser.add_argument("k", default=1, nargs='?')
        parser.add_argument("device", default=0, nargs='?')
        parser.add_argument("d", default=50, nargs='?')
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
