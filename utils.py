import numpy as np
import torch
from pytoune import torch_to_numpy
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import functional as F
import re
import os
import pickle as pkl


def load_embeddings(path):
    embeddings = {}
    # First we read the embeddings from the file, only keeping vectors for the words we need.
    i = 0
    with open(path, 'r', encoding='utf8') as embeddings_file:
        for line in embeddings_file:
            if len(line) > 50:
                fields = line.strip().split(' ')
                word = fields[0]
                vector = np.asarray(fields[1:], dtype='float32')
                embeddings[word] = vector
    return embeddings


def save_embeddings(embeddings, filename, path='./predicted_embeddings/'):
    os.makedirs(path, exist_ok=True)
    with open(path + filename, 'w', encoding='utf-8') as fhandle:
        for word, embedding in embeddings.items():
            str_embedding = ' '.join([str(i) for i in embedding])
            s = "{} {}\n".format(word, str_embedding)
            fhandle.write(s)


def load_examples(pathfile):
    with open(pathfile, 'rb') as file:
        examples = pkl.load(file)
    return examples


def save_examples(examples, path, filename):
    os.makedirs(path, exist_ok=True)
    with open(path + filename + '.pkl', 'wb') as file:
        pkl.dump(examples, file)
        

def parse_conll_file(filename):
    sentences = list()
    with open(filename) as fhandler:
        sentence = list()
        for line in fhandler:
            if not (line.startswith('-DOCSTART-') or line.startswith('\n')):
                token, _, _, e = line[:-1].split(' ')
                sentence.append(token.lower())
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


def load_vocab(path):
    vocab = set()
    with open(path, 'rb') as fhandle:
        for line in fhandle:
            vocab.add(line[:-1])
    return vocab


class WordsInContextVectorizer:
    def __init__(self, words_to_idx, chars_to_idx):
        self.words_to_idx = words_to_idx
        self.chars_to_idx = chars_to_idx

    def vectorize_sequence(self, sequence, to_idx):
        if 'UNK' in to_idx:
            unknown_index = to_idx['UNK']
            v = list()
            for item in sequence:
                if item in to_idx:
                    v.append(to_idx[item])
                elif item.capitalize() in to_idx:
                    v.append(to_idx[item.capitalize()])
                elif item.upper() in to_idx:
                    v.append(to_idx[item.upper()])
                elif item.lower() in to_idx:
                    v.append(to_idx[item.lower()])
                else:
                    v.append(to_idx['UNK'])
            return v
        else:
            return [to_idx[item] for item in sequence]

    def vectorize_example(self, example):
        x, y = example
        x = self.vectorize_unknown_example(x)
        return x + (y,)

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
        token = "2000-01-01"
    elif float_re.fullmatch(token):
        token = "0.0"
    elif int_re.fullmatch(token):
        token = "0"
    elif time_re.fullmatch(token):
        token = "00:00"
    elif code_re.fullmatch(token):
        token = "00-00-00-00"
    return token


def collate_fn(batch):
    print(batch)
    x, y = collate_x(batch)
    return (x, torch.FloatTensor(np.array(y)))


def collate_x(batch):
    batch = [(*x, y) for x, y in batch]  # Unwraps the batch
    *x, y = list(zip(*batch))

    padded_x = []
    for x_part in x:
        x_lengths = torch.LongTensor([len(item) for item in x_part])
        padded_x.append(pad_sequences(x_part, x_lengths))

    return (tuple(padded_x), y)


def pad_sequences(vectorized_seqs, seq_lengths):
    """
    Pads vectorized ngrams so that they occupy the same space in a LongTensor.
    """
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

    return seq_tensor


def ngrams(sequence, n=-1, pad_left=1, pad_right=1, left_pad_symbol='<BOS>', right_pad_symbol='<EOS>'):
    sequence = [left_pad_symbol] * pad_left + sequence + [right_pad_symbol] * pad_right

    L = len(sequence)
    m = n // 2
    if n == -1:
        m = L
    for i, item in enumerate(sequence[pad_left:-pad_right]):
        left_idx = max(0, i - m + pad_left)
        left_side = tuple(sequence[left_idx:i + pad_left])
        right_idx = min(L, i + m + pad_left + 1)
        right_side = tuple(sequence[i + pad_left + 1:right_idx])
        yield (left_side, item, right_side)


def euclidean_distance(y_pred_tensor, y_true_tensor):
    y_pred = torch_to_numpy(y_pred_tensor)
    y_true = torch_to_numpy(y_true_tensor)
    dist = np.linalg.norm((y_true - y_pred), axis=1).mean()
    return torch.FloatTensor([dist.tolist()])


def cosine_sim(y_pred, y_true):
    return F.cosine_similarity(y_true, y_pred).mean()


def square_distance(input, target):
    return F.pairwise_distance(input, target).mean()


if __name__ == '__main__':
    # test_preprocessing()
    ex = 'My name is JS'.split(' ')
    a = [ngram for ngram in ngrams(ex, -1, pad_left=2, pad_right=4)]
    for b in a:
        print(b)
