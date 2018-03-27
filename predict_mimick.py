import logging

import torch
from pytoune import torch_to_numpy

from evaluation import evaluate_embeddings
from mimick import Mimick
from train_mimick import build_vocab, WordsVectorizer, prepare_data
from utils import load_embeddings, load_vocab, parse_conll_file

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def main():
    train_embeddings = load_embeddings('./embeddings_settings/setting2/1_glove_embeddings/glove.6B.100d.txt')
    test = load_vocab('./embeddings_settings/setting2/all_oov_setting2.txt')

    path_sentences = './conll/train.txt'
    sentences = parse_conll_file(path_sentences)

    _, _, _, char_to_idx = prepare_data(
        embeddings=train_embeddings,
        sentences=sentences,
        n=1,
        ratio=.8,
        use_gpu=False,
        k=1)

    # vocab = build_vocab(train_embeddings.keys())
    corpus_vectorizer = WordsVectorizer(char_to_idx)

    x_test_tensor = [(word, torch.LongTensor([corpus_vectorizer.vectorize_unknown_example(word)])) for word in test]

    net = Mimick(
        characters_vocabulary=char_to_idx,
        characters_embedding_dimension=20,
        characters_hidden_state_dimension=50,
        word_embeddings_dimension=100,
        fully_connected_layer_hidden_dimension=50
    )
    net.load_state_dict(torch.load('./models/mimick100.torch'))
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

    evaluate_embeddings(my_embeddings, 100)

if __name__ == '__main__':
    main()