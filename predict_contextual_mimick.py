import logging
import math
import argparse

from train_contextual_mimick import my_ngrams

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import numpy
import torch
from nltk.util import ngrams
from pytoune.framework import torch_to_numpy

from evaluation import evaluate_embeddings
from contextual_mimick import ContextualMimick, get_contextual_mimick
from utils import load_embeddings, pad_sequences, parse_conll_file, make_vocab, WordsInContextVectorizer, load_vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n")
    parser.add_argument("model_path")
    parser.add_argument("path_words_to_predict")
    args = parser.parse_args()
    n = int(args.n)
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

    net = get_contextual_mimick(char_to_idx, word_to_idx)

    net.eval()
    net.load_state_dict(torch.load(model_path, map_location))

    test_sentences = parse_conll_file('./conll/valid.txt')
    test_sentences += parse_conll_file('./conll/train.txt')
    test_sentences += parse_conll_file('./conll/test.txt')
    test_vocab = load_vocab(path_words_to_predict)

    examples = [ngram for sentence in test_sentences for ngram in my_ngrams(sentence, n)]
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
        l = torch.autograd.Variable(torch.LongTensor([l]))
        w = torch.autograd.Variable(torch.LongTensor([w]))
        r = torch.autograd.Variable(torch.LongTensor([r]))
        if use_gpu:
            l.cuda()
            w.cuda()
            r.cuda()
        prediction = net((l, w, r))
        if word in my_embeddings:  # Compute the average
            my_embeddings[word] = (my_embeddings[word] + torch_to_numpy(prediction[0])) / 2
        else:
            my_embeddings[word] = torch_to_numpy(prediction[0])

    with open('./predicted_embeddings/contextual_mimick_validation_embeddings.txt', 'w') as fhandle:
        for word, embedding in my_embeddings.items():
            str_embedding = ' '.join([str(i) for i in embedding])
            s = "{} {}\n".format(word, str_embedding)
            fhandle.write(s)

    evaluate_embeddings(my_embeddings)

if __name__ == '__main__':
    main()