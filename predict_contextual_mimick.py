import logging
import math

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import numpy
import torch
from nltk.util import ngrams
from pytoune.framework import torch_to_numpy

from evaluation import evaluate_embeddings
from contextual_mimick import ContextualMimick
from utils import load_embeddings, pad_sequences, parse_conll_file, make_vocab, WordsInContextVectorizer, load_vocab


def main():
    # Prepare our examples
    sentences = parse_conll_file('./conll/train.txt')
    n = 31

    word_to_idx, char_to_idx = make_vocab(sentences)

    vectorizer = WordsInContextVectorizer(word_to_idx, char_to_idx)

    net = ContextualMimick(
        characters_vocabulary=char_to_idx,
        words_vocabulary=word_to_idx,
        characters_embedding_dimension=20,
        characters_hidden_state_dimension=50,
        words_hidden_state_dimension=50,
        word_embeddings_dimension=50,
        fully_connected_layer_hidden_dimension=50
    )

    net.eval()
    net.load_state_dict(torch.load('./results/contextual_mimick_n31/contextual_mimick_n31.torch'))
    test_sentences = parse_conll_file('./conll/valid.txt')
    test_sentences += parse_conll_file('./conll/train.txt')
    test_vocab = load_vocab('./conll/oov_vocab_wo_embeds.txt')
    raw_examples = [
        ngram  for sentence in test_sentences for ngram in ngrams(sentence, n, pad_left=True, pad_right=True, left_pad_symbol='<BOS>', right_pad_symbol='EOS')
    ]
    filtered_examples = [e for e in raw_examples if e[math.floor(n/2)].lower() in test_vocab]  # Target word is in test vocab

    examples_by_target_word = dict()
    for e in filtered_examples:
        if e[int(n/2)].lower() not in examples_by_target_word:
            examples_by_target_word[e[int(n/2)].lower()] = [e]
        else:
            examples_by_target_word[e[int(n/2)].lower()].append(e)

    more_than_one_examples = [e for e in examples_by_target_word.items() if len(e) > 1]

    filtered_examples_splitted = [(e[int(n/2)].lower(), vectorizer.vectorize_unknown_example((e[:int(n/2)], e[int(n/2)], e[int(n/2)+1:]))) for e in filtered_examples]

    my_embeddings = dict()
    for word, x in filtered_examples_splitted:
        l, w, r = x
        l = torch.autograd.Variable(torch.LongTensor([l]))
        w = torch.autograd.Variable(torch.LongTensor([w]))
        r = torch.autograd.Variable(torch.LongTensor([r]))
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