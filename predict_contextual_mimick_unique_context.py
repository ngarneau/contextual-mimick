import logging
import math
import argparse
import os

from contextual_mimick_unique_context import get_contextual_mimick_unique_context
from utils import ngrams

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import numpy
import torch
from utils import ngrams
from pytoune.framework import torch_to_numpy

from evaluation import evaluate_embeddings
from contextual_mimick import ContextualMimick, get_contextual_mimick
from utils import load_embeddings, pad_sequences, parse_conll_file, make_vocab, WordsInContextVectorizer, load_vocab
from sklearn.metrics.pairwise import cosine_similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", default=7, nargs='?')
    parser.add_argument("k", default=1, nargs='?')
    parser.add_argument("d", default=100, nargs='?')
    parser.add_argument("model_path", default='models/best_comick_n7_k1_d100_unique_context.torch', nargs='?')
    parser.add_argument("path_words_to_predict", default='./embeddings_settings/setting2/all_oov_setting2.txt', nargs='?')
    args = parser.parse_args()
    n = int(args.n)
    k = int(args.k)
    d = int(args.d)
    model_path = args.model_path
    path_words_to_predict = args.path_words_to_predict

    # Prepare our examples
    path_sentences = './conll/train.txt'
    all_sentences = parse_conll_file(path_sentences)
    all_sentences += parse_conll_file('./conll/valid.txt')
    all_sentences += parse_conll_file('./conll/test.txt')
    word_to_idx, char_to_idx = make_vocab(all_sentences)

    vectorizer = WordsInContextVectorizer(word_to_idx, char_to_idx)

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        map_location = lambda storage, loc: storage.cuda(0)
    else:
        map_location = lambda storage, loc: storage

    net = get_contextual_mimick_unique_context(char_to_idx, word_to_idx, word_embedding_dim=d)

    net.eval()
    net.load_state_dict(torch.load(model_path, map_location))

    test_sentences = parse_conll_file('./conll/valid.txt')
    test_sentences += parse_conll_file('./conll/train.txt')
    test_sentences += parse_conll_file('./conll/test.txt')
    test_vocab = load_vocab(path_words_to_predict)

    examples = [ngram for sentence in test_sentences for ngram in ngrams(sentence, n)]
    filtered_examples = [e for e in examples if e[1].lower() in test_vocab]  # Target word is in test vocab

    examples_by_target_word = dict()
    for e in filtered_examples:
        if e[1].lower() not in examples_by_target_word:
            examples_by_target_word[e[1].lower()] = [e]
        else:
            examples_by_target_word[e[1].lower()].append(e)

    filtered_examples_splitted = [(e[1].lower(), vectorizer.vectorize_unknown_example_merged_context(e)) for e in filtered_examples]

    my_embeddings = dict()
    for word, x in filtered_examples_splitted:
        c, w = x
        c = torch.LongTensor([c])
        w = torch.LongTensor([w])
        if use_gpu:
            c.cuda()
            w.cuda()
        prediction = net((
            torch.autograd.Variable(c),
            torch.autograd.Variable(w),
        ))
        if word in my_embeddings:  # Compute the average
            my_embeddings[word].append(torch_to_numpy(prediction[0]))
        else:
            my_embeddings[word] = [torch_to_numpy(prediction[0])]

    averaged_embeddings = dict()
    for word, embeddings in my_embeddings.items():
        averaged_embeddings[word] = numpy.mean(embeddings, axis=0)

    filepath = './predicted_embeddings/'
    os.makedirs(filepath, exist_ok=True)
    filename = 'comick_pred_unique_context_n{}_k{}_d{}_dropout.txt'.format(n, k, d)
    with open(filepath + filename, 'w') as fhandle:
        for word, embedding in averaged_embeddings.items():
            str_embedding = ' '.join([str(i) for i in embedding])
            s = "{} {}\n".format(word, str_embedding)
            fhandle.write(s)

    evaluate_embeddings(averaged_embeddings, d)

if __name__ == '__main__':
    main()
