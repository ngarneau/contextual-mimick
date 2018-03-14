import math

import numpy
import torch
from nltk.util import ngrams
from pytoune.framework import Model, torch_to_numpy
from pytoune.framework.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader

from model import ContextualMimick
from train_mimick import pad_sequences, load_embeddings, random_split, square_distance, euclidean_distance, load_vocab


def parse_conll_file(filename):
    sentences = list()
    with open(filename) as fhandler:
        sentence = list()
        for line in fhandler:
            if not (line.startswith('-DOCSTART-') or line.startswith('\n')):
                token, _, _, e = line[:-1].split(' ')
                sentence.append(token)
            else:
                if len(sentence) > 0:
                    sentences.append(sentence)
                sentence = list()
    return sentences


def make_vocab(sentences):
    vocab = set()
    char_vocab = set()
    for s in sentences:
        for w in s:
            vocab.add(w)
            for c in w:
                char_vocab.add(c)
    word_to_idx = {
        'PAD': 0,
        'UNK': 1,
        '<BOS>': 2,
        '<EOS>': 3,
    }
    char_to_idx = {
        'PAD': 0,
        'UNK': 1,
    }
    for w in sorted(vocab):
        word_to_idx[w] = len(word_to_idx)
    for w in sorted(char_vocab):
        char_to_idx[w] = len(char_to_idx)
    return word_to_idx, char_to_idx


class WordsInContextVectorizer:
    def __init__(self, words_to_idx, chars_to_idx):
        self.words_to_idx = words_to_idx
        self.chars_to_idx = chars_to_idx

    def vectorize_sequence(self, word, to_idx):
        if 'UNK' in to_idx:
            unknown_index = to_idx['UNK']
            return [to_idx.get(char, unknown_index) for char in word]
        else:
            return [to_idx[char] for char in word]

    def vectorize_example(self, example):
        x, y = example
        left_context, word, right_context = x
        vectorized_left_context = self.vectorize_sequence(left_context, self.words_to_idx)
        vectorized_word = self.vectorize_sequence(word, self.chars_to_idx)
        vectorized_right_context = self.vectorize_sequence(right_context, self.words_to_idx)
        return (
            vectorized_left_context,
            vectorized_word,
            vectorized_right_context,
            y
        )

    def vectorize_unknown_example(self, x):
        left_context, word, right_context = x
        vectorized_left_context = self.vectorize_sequence(left_context, self.words_to_idx)
        vectorized_word = self.vectorize_sequence(word, self.chars_to_idx)
        vectorized_right_context = self.vectorize_sequence(right_context, self.words_to_idx)
        return (
            vectorized_left_context,
            vectorized_word,
            vectorized_right_context
        )


def collate_examples(samples):
    left_contexts, words, right_contexts, labels = list(zip(*samples))

    left_contexts_lengths = torch.LongTensor([len(s) for s in left_contexts])
    padded_left_contexts = pad_sequences(left_contexts, left_contexts_lengths)

    seq_lengths = torch.LongTensor([len(s) for s in words])
    padded_words = pad_sequences(words, seq_lengths)

    right_contexts_lengths = torch.LongTensor([len(s) for s in right_contexts])
    padded_right_contexts = pad_sequences(right_contexts, right_contexts_lengths)

    labels = torch.FloatTensor(numpy.array(labels))

    # seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    # _, rev_perm_idx = perm_idx.sort(0)
    #
    # padded_left_contexts = padded_left_contexts[perm_idx]
    # padded_words = padded_words[perm_idx]
    # padded_right_contexts = padded_right_contexts[perm_idx]
    # labels = labels[perm_idx]

    return (
        (
            padded_left_contexts,
            padded_words,
            padded_right_contexts
        ),
        labels
    )


class Corpus:
    def __init__(self, examples, name, f=lambda x: x):
        self.examples = examples
        self.name = name
        self.transform = f

    def __iter__(self):
        for e in self.examples:
            yield self.transform(e)

    def __getitem__(self, item):
        return self.transform(self.examples[item])

    def __len__(self):
        return len(self.examples)


def main():
    # Prepare our examples
    train_embeddings = load_embeddings('./embeddings/train_embeddings.txt')
    sentences = parse_conll_file('./conll/train.txt')
    n = 5
    raw_examples = [
        ngram  for sentence in sentences for ngram in ngrams(sentence, n, pad_left=True, pad_right=True, left_pad_symbol='<BOS>', right_pad_symbol='EOS')
    ]
    filtered_examples = [e for e in raw_examples if 'OS>' not in e[math.floor(n/2)]]
    filtered_examples_splitted = [(e[:int(n/2)], e[int(n/2)], e[int(n/2)+1:]) for e in filtered_examples]
    training_data = [(x, train_embeddings[x[1].lower()]) for x in filtered_examples_splitted if x[1].lower() in train_embeddings]

    # Vectorize our examples
    word_to_idx, char_to_idx = make_vocab(sentences)
    # x_tensor, y_tensor = collate_examples([vectorizer.vectorize_example(x, y) for x, y in training_data])
    # dataset = TensorDataset(x_tensor, y_tensor)

    train_valid_ratio = 0.8
    m = int(len(training_data) * train_valid_ratio)
    train_dataset, valid_dataset = random_split(training_data, [m, len(training_data) - m])

    print(len(train_dataset), len(valid_dataset))

    vectorizer = WordsInContextVectorizer(word_to_idx, char_to_idx)
    train_loader = DataLoader(
        Corpus(train_dataset, 'train', vectorizer.vectorize_example),
        batch_size=16,
        collate_fn=collate_examples,
        shuffle=True
    )

    valid_loader = DataLoader(
        Corpus(valid_dataset, 'valid', vectorizer.vectorize_example),
        batch_size=16,
        collate_fn=collate_examples,
        shuffle=True
    )

    net = ContextualMimick(
        characters_vocabulary=char_to_idx,
        words_vocabulary=word_to_idx,
        characters_embedding_dimension=20,
        characters_hidden_state_dimension=50,
        words_hidden_state_dimension=50,
        word_embeddings_dimension=50,
        fully_connected_layer_hidden_dimension=50
    )
    net.load_words_embeddings(train_embeddings)

    # lrscheduler = ReduceLROnPlateau(patience=5, factor=.1)
    # early_stopping = EarlyStopping(patience=10)
    # checkpoint = ModelCheckpoint('contextual_mimick.torch')
    # model = Model(net, SGD(net.parameters(), lr=0.01), square_distance, metrics=[euclidean_distance])
    # model.fit_generator(train_loader, valid_loader, n_epochs=100, callbacks=[lrscheduler, checkpoint, early_stopping])

    net.eval()
    net.load_state_dict(torch.load('./models/contextual_mimick.torch'))
    test_sentences = parse_conll_file('./conll/valid.txt')
    test_vocab = load_vocab('./validation_vocab.txt')
    words_in_testing_instances = set()
    raw_examples = [
        ngram  for sentence in test_sentences for ngram in ngrams(sentence, n, pad_left=True, pad_right=True, left_pad_symbol='<BOS>', right_pad_symbol='EOS')
    ]
    filtered_examples = [e for e in raw_examples if e[math.floor(n/2)] in test_vocab]  # Target word is in test vocab
    for e in filtered_examples:
        words_in_testing_instances.add(e[int(n/2)])
    filtered_examples_splitted = [(e[int(n/2)], vectorizer.vectorize_unknown_example((e[:int(n/2)], e[int(n/2)], e[int(n/2)+1:]))) for e in filtered_examples]

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


    # Load glove embeddings for comparisons
    glove_embeddings = load_embeddings('./embeddings/glove_valid_embeddings.txt')

    # Compare distance of mimick with the glove embeddings
    mimick_distances = list()
    mimick_embeddings = load_embeddings('./embeddings/previous_mimick_validation_embeddings.txt')
    for word, embedding in mimick_embeddings.items():
        if word in glove_embeddings and word in words_in_testing_instances:
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