import numpy
import torch
from pytoune import torch_to_numpy
from torch._utils import _accumulate
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


def load_embeddings(path):
    embeddings = {}
    # First we read the embeddings from the file, only keeping vectors for the words we need.
    i = 0
    with open(path, 'r') as embeddings_file:
        for line in embeddings_file:
            if i > 100000:
                break
            i += 1
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


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    indices = torch.randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def euclidean_distance(y_pred_tensor, y_true_tensor):
    y_pred = torch_to_numpy(y_pred_tensor)
    y_true = torch_to_numpy(y_true_tensor)
    dist = numpy.linalg.norm((y_true - y_pred), axis=1).mean()
    return torch.FloatTensor([dist.tolist()])


def square_distance(input, target):
    return F.pairwise_distance(input, target).mean()


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


def load_vocab(path):
    vocab = set()
    with open(path) as fhandle:
        for line in fhandle:
            vocab.add(line[:-1])
    return vocab


class DataLoader(DataLoader):
    """
    Overloads the DataLoader Class of PyTorch so that it can copy the data and target to the GPU if desired.
    """

    def __init__(self, *args, use_gpu=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gpu = use_gpu and torch.cuda.is_available()

    def to_cuda(self, obj):
        if self.use_gpu:
            if isinstance(obj, tuple):
                return [o.cuda() for o in obj]
            else:
                return obj.cuda()
        else:
            return obj

    def __iter__(self):
        for x, y in super().__iter__():
            yield self.to_cuda(x), self.to_cuda(y)


def ngrams(sequence, n, pad_left=1, pad_right=1, left_pad_symbol='<BOS>', right_pad_symbol='<EOS>'):
    sequence = [left_pad_symbol]*pad_left + sequence + [right_pad_symbol]*pad_right

    L = len(sequence)
    m = n//2
    for i, item in enumerate(sequence[1:-1]):
        left_idx = max(0, i-m+1)
        left_side = tuple(sequence[left_idx:i+1])
        right_idx = min(L, i+m+2)
        right_side = tuple(sequence[i+2:right_idx])
        yield (left_side, item, right_side)