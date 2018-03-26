import logging
import math
import argparse
import os

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", default=41, nargs='?')
    parser.add_argument("k", default=2, nargs='?')
    parser.add_argument("d", default=100, nargs='?')
    parser.add_argument("model_path", default='best_comick_n41_k2_d100.torch', nargs='?')
    parser.add_argument("path_words_to_predict", default='./embeddings_settings/setting2/all_oov_setting2.txt', nargs='?')
    args = parser.parse_args()
    n = int(args.n)
    k = int(args.k)
    d = int(args.d)
    model_path = args.model_path
    path_words_to_predict = args.path_words_to_predict

    # Prepare our examples
    sentences = parse_conll_file('./conll/train.txt')

    word_to_idx, char_to_idx = make_vocab(sentences)

    vectorizer = WordsInContextVectorizer(word_to_idx, char_to_idx)

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        map_location = lambda storage, loc: storage.cuda(0)
    else:
        map_location = lambda storage, loc: storage

    net = get_contextual_mimick(char_to_idx, word_to_idx, word_embedding_dim=d)

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

    filtered_examples_splitted = [(e[1].lower(), vectorizer.vectorize_unknown_example(e)) for e in filtered_examples]

    my_embeddings = dict()
    for word, x in filtered_examples_splitted:
        l, w, r = x
        l = torch.LongTensor([l])
        w = torch.LongTensor([w])
        r = torch.LongTensor([r])
        if use_gpu:
            l.cuda()
            w.cuda()
            r.cuda()
        prediction = net((
            torch.autograd.Variable(l),
            torch.autograd.Variable(w),
            torch.autograd.Variable(r)
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
    filename = 'comick_pred_n{}_k{}_d{}_dropout.txt'.format(n, k, d)
    with open(filepath + filename, 'w') as fhandle:
        for word, embedding in averaged_embeddings.items():
            str_embedding = ' '.join([str(i) for i in embedding])
            s = "{} {}\n".format(word, str_embedding)
            fhandle.write(s)

    evaluate_embeddings(averaged_embeddings, d)

if __name__ == '__main__':
    main()
