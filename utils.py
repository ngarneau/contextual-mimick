import logging
import numpy as np
import collections
import torch
from pytoune import torch_to_numpy
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import functional as F
import re
import os
import pickle as pkl

from torch.utils.data import DataLoader
from torch.nn.init import kaiming_uniform, kaiming_normal, constant
from pytoune.framework import Experiment as PytouneExperiment
from pytoune.framework.callbacks import ClipNorm, ReduceLROnPlateau, Callback, EarlyStopping
from pytoune.utils import torch_to_numpy

import collections
from collections import defaultdict
from polyglot import text

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

import numpy as np

import torch
from torch import nn

UNK_TAG = "<UNK>"
NONE_TAG = "<NONE>"
START_TAG = "<START>"
END_TAG = "<STOP>"
PADDING_WORD = "<PAD>"
PADDING_CHAR = "<*>"
POS_KEY = "POS"

Instance = collections.namedtuple("Instance", ["source", "sentence", "chars", "substrings", "tags"])

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


class KLWeightingSigmoidDecay(Callback):
    def __init__(self, k, batches_per_epoch):
        super().__init__()
        self.k = k
        self.batches_per_epoch = batches_per_epoch

    def on_epoch_begin(self, epoch, logs):
        self.i = epoch * self.batches_per_epoch

    def on_batch_begin(self, batch, logs):
        self.i += 1
        ratio = self.k / (self.k + math.exp(self.i / self.k))
        words_to_drop_ratio = max(0, ratio - 0.6)
        self.model.model.oov_rate_to_drop = words_to_drop_ratio


class MetricsCallback(Callback):
    def __init__(self, logger):
        super(MetricsCallback, self).__init__()
        self.logger = logger
        self.stats = defaultdict(list)

    def on_backward_end(self, batch):
        for parameter, values in self.model.model.named_parameters():
            self.stats["{}.grad.mean".format(parameter)].append(float(values.mean()))
            self.stats["{}.grad.std".format(parameter)].append(float(values.std()))

    def on_epoch_end(self, epoch, logs):
        # Log gradient stats
        for stat, values in self.stats.items():
            self.logger.log_scalar(stat, np.mean(values))
        self.stats = defaultdict(list)

        self.logger.log_scalar("epochs.train.loss", logs['loss'])
        self.logger.log_scalar("epochs.val.loss", logs['val_loss'])
        if 'acc' in logs:
            self.logger.log_scalar("epochs.train.acc", logs['acc'])
            self.logger.log_scalar("epochs.val.acc", logs['val_acc'])


class MyEmbeddings(nn.Embedding):
    def __init__(self, word_to_idx, embedding_dim):
        super(MyEmbeddings, self).__init__(len(word_to_idx), embedding_dim, padding_idx=0)
        self.embedding_dim = embedding_dim
        self.vocab_size = len(word_to_idx)
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}

    def set_item_embedding(self, idx, embedding):
        self.weight.data[idx] = torch.FloatTensor(embedding)

    def load_words_embeddings(self, vec_model):
        for word in vec_model:
            if word in self.word_to_idx:
                idx = self.word_to_idx[word]
                embedding = vec_model[word]
                self.set_item_embedding(idx, embedding)

class LanguageDataset:

    BASE_PATH = "./data/Universal Dependencies 1.4/ud-treebanks-v1.4/{}/{}-"

    def __init__(self, polyglot_abbreviation, ud_path, ud_filename_prefix):
        logging.info("Creating dataset for {} - {}".format(polyglot_abbreviation, ud_path))
        self.base_path = self.BASE_PATH.format(ud_path, ud_filename_prefix)
        self.train_path = self.base_path + '{}.conllu'.format('train')
        self.dev_path = self.base_path + '{}.conllu'.format('dev')
        self.test_path = self.base_path + '{}.conllu'.format('test')

        self.polyglot_abbreviation = polyglot_abbreviation
        self.ud_path = ud_path
        self.ud_filename_prefix = ud_filename_prefix

        self.options = {
            'ud_tags': True,
            'no_morphotags': False
        }

        self.word_to_index = {} # mapping from word to index
        self.tags_to_index = {} # mapping from attribute name to mapping from tag to index
        self.char_to_index = {} # mapping from character to index, for char-RNN concatenations
        self.bos_to_index = {} # mapping from character to index, for char-RNN concatenations

        # Add special tokens / tags / chars to dicts
        self.word_to_index[PADDING_WORD] = len(self.word_to_index) # Pad is 0
        self.word_to_index[UNK_TAG] = len(self.word_to_index) # Unk is 1
        self.word_to_index[START_TAG] = len(self.word_to_index) # Start is 2
        self.word_to_index[END_TAG] = len(self.word_to_index) # End is 3

        self.char_to_index[PADDING_CHAR] = len(self.char_to_index)
        self.bos_to_index[PADDING_CHAR] = len(self.bos_to_index)

        self.embedding_dim = None
        self.__get_embeddings()
        self.__parse_dataset()

        self.idx_to_word = {v: k for k, v in self.word_to_index.items()}

    def __get_embeddings(self):
        embeddings = text.load_embeddings(lang=self.polyglot_abbreviation)
        self.embeddings = dict()
        for word in embeddings.words:
            self.embeddings[word] = embeddings.get(word)
            if self.embedding_dim is None:
                self.embedding_dim = len(embeddings.get(word))

    def __parse_dataset(self):
        self.training_instances, self.training_vocab = read_file(
            self.train_path,
            self.word_to_index,
            self.tags_to_index,
            self.char_to_index,
            self.bos_to_index,
            self.options
        )

        self.dev_instances, self.dev_vocab = read_file(
            self.dev_path,
            self.word_to_index,
            self.tags_to_index,
            self.char_to_index,
            self.bos_to_index,
            self.options
        )

        self.test_instances, self.test_vocab = read_file(
            self.test_path,
            self.word_to_index,
            self.tags_to_index,
            self.char_to_index,
            self.bos_to_index,
            self.options
        )


def cos_matrix_multiplication(matrix, vector):
    """
    Calculating pairwise cosine distance using matrix vector multiplication.
    """
    dotted = matrix.dot(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    matrix_vector_norms = np.multiply(matrix_norms, vector_norm)
    neighbors = np.divide(dotted, matrix_vector_norms)
    return neighbors


def get_source_directory(directory_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), directory_name)


def get_experiment_directory(directory_name):
    default_dir = get_source_directory('./results')
    dest_directory = os.environ.get('RESULTS_DIR', default_dir)
    return os.path.join(dest_directory, directory_name)


def split_tagstring(s, uni_key=False, has_pos=False):
    '''
    Returns attribute-value mapping from UD-type CONLL field
    :param uni_key: if toggled, returns attribute-value pairs as joined strings (with the '=')
    :param has_pos: input line segment includes POS tag label
    '''
    if has_pos:
        s = s.split("\t")[1]
    ret = [] if uni_key else {}
    if "=" not in s: # incorrect format
        return ret
    for attval in s.split('|'):
        attval = attval.strip()
        if not uni_key:
            a,v = attval.split('=')
            ret[a] = v
        else:
            ret.append(attval)
    return ret

def make_substrings(s, lmin=3, lmax=6) :
    s = '<' + s + '>'
    for i in range(len(s)) :
        s0 = s[i:]
        for j in range(lmin, 1 + min(lmax, len(s0))) :
            yield s0[:j]


def read_file(filename, w2i, t2is, c2i, b2i, options):
    """
    Read in a dataset and turn it into a list of instances.
    Modifies the w2i, t2is and c2i dicts, adding new words/attributes/tags/chars
    as it sees them.
    """

    # populate mandatory t2i tables
    if POS_KEY not in t2is:
        t2is[POS_KEY] = {}
        t2is[POS_KEY][PADDING_WORD] = len(t2is[POS_KEY])

    # build dataset
    instances = []
    vocab_counter = collections.Counter()
    with open(filename, "r", encoding="utf-8") as f:

        # running sentence buffers (lines are tokens)
        sentence = []
        chars = []
        substrings = []
        source = []
        tags = defaultdict(list)

        # main file reading loop
        for i, line in enumerate(f):

            # discard comments
            if line.startswith("#"):
                continue

            # parse sentence end
            elif line.isspace():

                # pad tag lists to sentence end
                slen = len(sentence)
                for seq in tags.values():
                    if len(seq) < slen:
                        seq.extend([1] * (slen - len(seq))) # 0 guaranteed below to represent NONE_TAG

                # add sentence to dataset
                instances.append(Instance(source, sentence, chars, substrings, tags))
                source = []
                chars = []
                substrings = []
                sentence = []
                tags = defaultdict(list)

            else:

                # parse token information in line
                data = line.split("\t")
                if '-' in data[0]: # Some UD languages have contractions on a separate line, we don't want to include them also
                    continue
                try:
                    idx = int(data[0])
                except:
                    continue
                word = data[1]
                source.append(word)
                postag = data[3] if options['ud_tags'] else data[4]
                morphotags = {} if options['no_morphotags'] else split_tagstring(data[5], uni_key=False)

                # ensure counts and dictionary population
                vocab_counter[word] += 1
                if word not in w2i:
                    w2i[word] = len(w2i)
                pt2i = t2is[POS_KEY]
                if postag not in pt2i:
                    pt2i[postag] = len(pt2i)

                chars_for_word = list()
                for c in word:
                    if c not in c2i:
                        c2i[c] = len(c2i)
                    chars_for_word.append(c2i[c])
                chars.append(chars_for_word)

                # BoS data prep
                bos = make_substrings(word)
                bos_for_word = list()
                for b in bos:
                    if b not in b2i:
                        b2i[b] = len(b2i)
                    bos_for_word.append(b2i[b])
                substrings.append(bos_for_word)

                for key, val in morphotags.items():
                    if key not in t2is:
                        t2is[key] = {PADDING_WORD: 0, NONE_TAG: 1}
                    mt2i = t2is[key]
                    if val not in mt2i:
                        mt2i[val] = len(mt2i)

                # add data to sentence buffer
                sentence.append(w2i[word])
                tags[POS_KEY].append(t2is[POS_KEY][postag])
                for k,v in morphotags.items():
                    mtags = tags[k]
                    # pad backwards to latest seen
                    missing_tags = idx - len(mtags) - 1
                    mtags.extend([1] * missing_tags) # 0 guaranteed above to represent NONE_TAG
                    mtags.append(t2is[k][v])

    return instances, vocab_counter


def oov_appearance_rate_in_contexts(sentences, oov_ids, n=20):
    oovs_app = list()
    lengths = list()
    for sentence in sentences:
        for i, word_id in enumerate(sentence):
            if word_id in oov_ids:
                left_context = sentence[max(0, i-n):max(0, i)]
                right_context = sentence[min(len(sentence)-1, i+1):min(len(sentence), i+n)]
                context = left_context + right_context
                num_oovs = len([w for w in context if w in oov_ids])
                ratio = num_oovs / len(context)
                oovs_app.append(ratio)
                lengths.append(len(context))
    return oovs_app, lengths
