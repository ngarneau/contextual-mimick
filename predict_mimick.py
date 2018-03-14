import numpy
import torch
from pytoune import torch_to_numpy

from model import Mimick
from train_mimick import load_embeddings, load_vocab, build_vocab, WordsVectorizer


def main():
    train_embeddings = load_embeddings('./embeddings/train_embeddings.txt')
    test = load_vocab('./validation_vocab.txt')

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
    with open('./predicted_embeddings/current_mimick_validation_embeddings.txt', 'w') as fhandle:
        for word, tensor in x_test_tensor:
            prediction = net(torch.autograd.Variable(tensor))
            str_embedding = ' '.join([str(i) for i in prediction[0].data])
            s = "{} {}\n".format(word, str_embedding)
            fhandle.write(s)
            my_embeddings[word] = torch_to_numpy(prediction[0])


    # Load glove embeddings for comparisons
    glove_embeddings = load_embeddings('./embeddings/glove_valid_embeddings.txt')

    # Compare distance of mimick with the glove embeddings
    mimick_distances = list()
    mimick_embeddings = load_embeddings('./embeddings/previous_mimick_validation_embeddings.txt')
    for word, embedding in mimick_embeddings.items():
        if word in glove_embeddings:
            target_embedding = glove_embeddings[word]
            mimick_distances.append(numpy.linalg.norm(embedding - target_embedding))
    print("Mimick distance: {}, {} ({})".format(numpy.mean(mimick_distances), numpy.std(mimick_distances), len(mimick_distances)))

    # Compare distance of our implementation with the glove embeddings
    our_distances = list()
    for word, embedding in my_embeddings.items():
        if word in glove_embeddings:
            target_embedding = glove_embeddings[word]
            our_distances.append(numpy.linalg.norm(embedding - target_embedding))
    print("Our distance: {}, {} ({})".format(numpy.mean(our_distances), numpy.std(our_distances), len(our_distances)))

    # Compare distance of our embeds from mimick's
    our_distances_with_mimick = list()
    for word, embedding in my_embeddings.items():
        if word in mimick_embeddings:
            target_embedding = mimick_embeddings[word]
            our_distances_with_mimick.append(numpy.linalg.norm(embedding - target_embedding))
    print("Our distance: {}, {} ({})".format(numpy.mean(our_distances_with_mimick), numpy.std(our_distances_with_mimick), len(our_distances_with_mimick)))

if __name__ == '__main__':
    main()