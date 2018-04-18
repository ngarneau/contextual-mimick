import logging

from pytoune.framework import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger, Model
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader

from downstream_task.models import LSTMTagger
from downstream_task.sequence_tagging import sequence_cross_entropy, acc, collate_examples, make_vocab_and_idx
from utils import load_embeddings


def parse_pos_file(filename):
    sentences = list()
    targets = list()
    with open(filename, encoding='utf-8') as fhandler:
        sentence = list()
        tags = list()
        for line in fhandler:
            if not (line.startswith('-DOCSTART-') or line.startswith('\n')):
                token, pos, chunk, e = line[:-1].split(' ')
                sentence.append(token)
                tags.append(pos)
            else:
                if len(sentence) > 0:
                    sentences.append(sentence)
                    targets.append(tags)
                sentence = list()
                tags = list()
    return sentences, targets


def train(embeddings, model_name='vanilla'):
    train_sentences, train_tags = parse_pos_file('./data/conll/train.txt')
    valid_sentences, valid_tags = parse_pos_file('./data/conll/valid.txt')
    test_sentences, test_tags = parse_pos_file('./data/conll/test.txt')

    words_vocab, words_to_idx = make_vocab_and_idx(train_sentences + valid_sentences + test_sentences)
    tags_vocab, tags_to_idx = make_vocab_and_idx(train_tags + valid_tags + test_tags)

    train_sentences = [[words_to_idx[word] for word in sentence] for sentence in train_sentences]
    train_tags = [[tags_to_idx[word] for word in sentence] for sentence in train_tags]

    valid_sentences = [[words_to_idx[word] for word in sentence] for sentence in valid_sentences]
    valid_tags = [[tags_to_idx[word] for word in sentence] for sentence in valid_tags]

    test_sentences = [[words_to_idx[word] for word in sentence] for sentence in test_sentences]
    test_tags = [[tags_to_idx[word] for word in sentence] for sentence in test_tags]

    train_dataset = list(zip(train_sentences, train_tags))
    valid_dataset = list(zip(valid_sentences, valid_tags))
    test_dataset = list(zip(test_sentences, test_tags))

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_examples
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_examples
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_examples
    )

    net = LSTMTagger(
        100,
        50,
        words_to_idx,
        len(tags_to_idx)
    )
    net.load_words_embeddings(embeddings)

    lrscheduler = ReduceLROnPlateau(patience=5)
    early_stopping = EarlyStopping(patience=10)
    checkpoint = ModelCheckpoint('./models/pos_{}.torch'.format(model_name), save_best_only=True, restore_best=True)
    csv_logger = CSVLogger('./train_logs/pos_{}.csv'.format(model_name))
    model = Model(net, Adam(net.parameters(), lr=0.001), sequence_cross_entropy, metrics=[acc])
    model.fit_generator(train_loader, valid_loader, epochs=40, callbacks=[lrscheduler, checkpoint, early_stopping, csv_logger])
    loss, metric = model.evaluate_generator(test_loader)
    logging.info("Test loss: {}".format(loss))
    logging.info("Test metric: {}".format(metric))


if __name__ == '__main__':
    embeddings = load_embeddings('./data/glove_embeddings/glove.6B.100d.txt')
    train(embeddings)
