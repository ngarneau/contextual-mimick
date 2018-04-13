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
    with open(filename) as fhandler:
        sentence = list()
        tags = list()
        for line in fhandler:
            if not (line.startswith('-DOCSTART-') or line.startswith('\n')):
                token, pos, chunk = line[:-1].split(' ')
                sentence.append(token)
                tags.append(pos)
            else:
                if len(sentence) > 0:
                    sentences.append(sentence)
                    targets.append(tags)
                sentence = list()
                tags = list()
    return sentences, targets


def train(embeddings_path):
    train_sentences, train_tags = parse_pos_file('./data/pos/train.txt/data')
    test_sentences, test_tags = parse_pos_file('./data/pos/test.txt/data')

    words_vocab, words_to_idx = make_vocab_and_idx(train_sentences + test_sentences)
    tags_vocab, tags_to_idx = make_vocab_and_idx(train_tags + test_tags)

    train_sentences = [[words_to_idx[word] for word in sentence] for sentence in train_sentences]
    train_tags = [[tags_to_idx[word] for word in sentence] for sentence in train_tags]

    test_sentences = [[words_to_idx[word] for word in sentence] for sentence in test_sentences]
    test_tags = [[tags_to_idx[word] for word in sentence] for sentence in test_tags]

    train_sentences, valid_sentences, train_tags, valid_tags = train_test_split(train_sentences, train_tags)

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

    train_embeddings = load_embeddings(embeddings_path)

    net = LSTMTagger(
        100,
        50,
        words_to_idx,
        len(tags_to_idx)
    )
    net.load_words_embeddings(train_embeddings)

    lrscheduler = ReduceLROnPlateau(patience=5)
    early_stopping = EarlyStopping(patience=10)
    checkpoint = ModelCheckpoint('./models/pos.torch', save_best_only=True)
    csv_logger = CSVLogger('./train_logs/pos.csv')
    model = Model(net, Adam(net.parameters(), lr=0.001), sequence_cross_entropy, metrics=[acc])
    model.fit_generator(train_loader, valid_loader, epochs=40, callbacks=[lrscheduler, checkpoint, early_stopping, csv_logger])
    print(model.evaluate_generator(test_loader))


if __name__ == '__main__':
    train('./data/glove_embeddings/glove.6B.100d.txt')
