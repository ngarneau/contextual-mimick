import logging

import torch
from pytoune import torch_to_numpy

from evaluation import evaluate_embeddings
from mimick import Mimick
from train_mimick import build_vocab, WordsVectorizer
from utils import load_embeddings, load_vocab

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def main():
    train_embeddings = load_embeddings('./embeddings/train_embeddings.txt')
    test = load_vocab('./conll/all_oov.txt')

    vocab = build_vocab(train_embeddings.keys())
    corpus_vectorizer = WordsVectorizer(vocab)

    x_test_tensor = [(word, torch.LongTensor([corpus_vectorizer.vectorize_unknown_example(word)])) for word in test]

    net = Mimick(
        characters_vocabulary=vocab,
        characters_embedding_dimension=20,
        characters_hidden_state_dimension=50,
        word_embeddings_dimension=50,
        fully_connected_layer_hidden_dimension=50
    )
    net.load_state_dict(torch.load('./models/mimick.torch'))
    net.eval()

    my_embeddings = dict()
    logging.info('Predicting embeddings for test set...')
    with open('./predicted_embeddings/current_mimick_validation_embeddings.txt', 'w') as fhandle:
        for word, tensor in x_test_tensor:
            prediction = net(torch.autograd.Variable(tensor))
            str_embedding = ' '.join([str(i) for i in prediction[0].data])
            s = "{} {}\n".format(word, str_embedding)
            fhandle.write(s)
            my_embeddings[word] = torch_to_numpy(prediction[0])

    evaluate_embeddings(my_embeddings)

if __name__ == '__main__':
    main()