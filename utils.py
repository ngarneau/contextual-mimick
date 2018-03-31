import numpy
import torch
from pytoune import torch_to_numpy
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import functional as F
import re

def load_embeddings(path):
    embeddings = {}
    # First we read the embeddings from the file, only keeping vectors for the words we need.
    i = 0
    with open(path, 'r', encoding='utf8') as embeddings_file:
        for line in embeddings_file:
            fields = line.strip().split(' ')
            word = fields[0]
            vector = numpy.asarray(fields[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def pad_sequences(vectorized_seqs, seq_lengths):
    """
    Pads vectorized ngrams so that they occupy the same space in a LongTensor.
    """
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    return seq_tensor


def euclidean_distance(y_pred_tensor, y_true_tensor):
    y_pred = torch_to_numpy(y_pred_tensor)
    y_true = torch_to_numpy(y_true_tensor)
    dist = numpy.linalg.norm((y_true - y_pred), axis=1).mean()
    return torch.FloatTensor([dist.tolist()])


def cosine_sim(y_pred_tensor, y_true_tensor):
    y_pred = torch_to_numpy(y_pred_tensor)
    y_true = torch_to_numpy(y_true_tensor)
    dist = cosine_similarity(y_true, y_pred).mean()
    return torch.FloatTensor([dist.tolist()])


def square_distance(input, target):
    return F.pairwise_distance(input, target).mean()

def cosine_distance(input, target):
    return 1.0 - F.cosine_similarity(input, target).mean()


def parse_conll_file(filename):
    sentences = list()
    with open(filename) as fhandler:
        sentence = list()
        for line in fhandler:
            if not (line.startswith('-DOCSTART-') or line.startswith('\n')):
                token, _, _, e = line[:-1].split(' ')
                sentence.append(preprocess_token(token))
            else:
                if len(sentence) > 0:
                    sentences.append(sentence)
                sentence = list()
    return sentences


def preprocess_token(token):
    """
    Modifies a token in a particular format to a unique predefined format.
    """
    date_re = re.compile(r'\d{2}(\d{2})?[/-]\d{2}[/-]\d{2}')
    float_re = re.compile(r'(\d+,)*\d+\.\d*')
    int_re = re.compile(r'(\d+,)*\d{3,}')
    time_re = re.compile(r'\d{1,2}:\d{2}(\.\d*)?')
    code_re = re.compile(r'\d+(-\d+){3,}')

    if date_re.fullmatch(token):
        token = "<DATE>"
    elif float_re.fullmatch(token):
        token = "<FLOAT>"
    elif int_re.fullmatch(token):
        token = "<INT>"
    elif time_re.fullmatch(token):
        token = "<TIME>"
    elif code_re.fullmatch(token):
        token = "<CODE>"
    else:
        token = token.lower()
    return token

def test_preprocessing():
    for token in ['1998/08/01',
                '1998-08-01',
                '98/08/01',
                '98-08-01',
                '10.',
                '10.3232',
                '10,10.',
                '10,10.3232',
                '10,100,100.3232',
                '10',
                '10,002',
                '10:10',
                '10:10.3232',
                '10-10-10-10']:
        t = preprocess_token(token)
        # print(t)

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
            v = list()
            for char in word:
                if char not in to_idx:
                    # print("Unknown word: {}".format(char))
                    v.append(to_idx['UNK'])
                else:
                    v.append(to_idx[char])
            return v
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

    def vectorize_unknown_example_merged_context(self, x):
        left_context, word, right_context = x
        context = left_context + right_context
        vectorized_context = self.vectorize_sequence(context, self.words_to_idx)
        vectorized_word = self.vectorize_sequence(word, self.chars_to_idx)
        return (
            vectorized_context,
            vectorized_word,
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

    return (
        (
            padded_left_contexts,
            padded_words,
            padded_right_contexts
        ),
        labels
    )


def collate_examples_unique_context(samples):
    contexts, words, labels = list(zip(*samples))

    contexts_lengths = torch.LongTensor([len(s) for s in contexts])
    padded_contexts = pad_sequences(contexts, contexts_lengths)

    seq_lengths = torch.LongTensor([len(s) for s in words])
    padded_words = pad_sequences(words, seq_lengths)

    labels = torch.FloatTensor(numpy.array(labels))

    return (
        (
            padded_contexts,
            padded_words,
        ),
        labels
    )

def collate_fn(batch):
    batch = [(*x, y) for x, y in batch] # Unwraps the batch
    *x, y = list(zip(*batch))
    
    padded_x = []
    for x_part in x:
        x_lengths = torch.LongTensor([len(item) for item in x_part])
        padded_x.append(pad_sequences(x_part, x_lengths))
    
    y = torch.FloatTensor(numpy.array(y))

    return (tuple(padded_x), y)


def load_vocab(path):
    vocab = set()
    with open(path) as fhandle:
        for line in fhandle:
            vocab.add(line[:-1])
    return vocab


def ngrams(sequence, n, pad_left=1, pad_right=1, left_pad_symbol='<BOS>', right_pad_symbol='<EOS>'):
    sequence = [left_pad_symbol]*pad_left + sequence + [right_pad_symbol]*pad_right

    L = len(sequence)
    m = n//2
    for i, item in enumerate(sequence[pad_left:-pad_right]):
        left_idx = max(0, i-m+pad_left)
        left_side = tuple(sequence[left_idx:i+pad_left])
        right_idx = min(L, i+m+pad_left+pad_right)
        right_side = tuple(sequence[i+pad_left+pad_right:right_idx])
        yield (left_side, item, right_side)


if __name__ == '__main__':
    test_preprocessing()