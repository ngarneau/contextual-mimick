import math
import logging
import os
import collections
from collections import defaultdict
from polyglot import text

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

import numpy as np

import torch
from torch import nn

import sacred
from sacred import Experiment
from sacred.observers import MongoObserver

from torch.utils.data import DataLoader
from torch.nn.init import kaiming_uniform, kaiming_normal, constant
from downstream_task.sequence_tagging import collate_examples_multiple_tags
from downstream_task.models import SimpleLSTMTagger, CharRNN

from pytoune.framework import Experiment as PytouneExperiment
from pytoune.framework.callbacks import ClipNorm, ReduceLROnPlateau, Callback, EarlyStopping
from pytoune.utils import torch_to_numpy

from pymongo import MongoClient

from comick import TheFinalComick, TheFinalComickBoS, BoS, Mimick

from utils import load_embeddings


UNK_TAG = "<UNK>"
NONE_TAG = "<NONE>"
START_TAG = "<START>"
END_TAG = "<STOP>"
PADDING_WORD = "<PAD>"
PADDING_CHAR = "<*>"
POS_KEY = "POS"

Instance = collections.namedtuple("Instance", ["source", "sentence", "chars", "substrings", "tags"])

# Logging thangs
base_config_file = './configs/base.json'
experiment = Experiment('ud_tagging_model_multiple_tags')
experiment.add_config(base_config_file)
experiment.observers.append(MongoObserver.create(
    url=os.environ['DB_URL'],
    db_name=os.environ['DB_NAME']))

client = MongoClient(os.environ['DB_URL'])
database = client[os.environ['DB_NAME']]
collection = database['logs']



languages = {
    'kk': ('kk', 'UD_Kazakh', 'kk-ud'),
    'ta': ('ta', 'UD_Tamil', 'ta-ud'),
    'lv': ('lv', 'UD_Latvian', 'lv-ud'),
    'vi': ('vi', 'UD_Vietnamese', 'vi-ud'),
    'hu': ('hu', 'UD_Hungarian', 'hu-ud'),
    'tr': ('tr', 'UD_Turkish', 'tr-ud'),
    'el': ('el', 'UD_Greek', 'el-ud'),
    'bg': ('bg', 'UD_Bulgarian', 'bg-ud'),
    'sv': ('sv', 'UD_Swedish', 'sv-ud'),
    'eu': ('eu', 'UD_Basque', 'eu-ud'),
    'ru': ('ru', 'UD_Russian', 'ru-ud'),
    'da': ('da', 'UD_Danish', 'da-ud'),
    'id': ('id', 'UD_Indonesian', 'id-ud'),
    'zh': ('zh', 'UD_Chinese', 'zh-ud'),
    'fa': ('fa', 'UD_Persian', 'fa-ud'),
    'he': ('he', 'UD_Hebrew', 'he-ud'),
    'ro': ('ro', 'UD_Romanian', 'ro-ud'),
    'en': ('en', 'UD_English', 'en-ud'),
    'ar': ('ar', 'UD_Arabic', 'ar-ud'),
    'hi': ('hi', 'UD_Hindi', 'hi-ud'),
    'it': ('it', 'UD_Italian', 'it-ud'),
    'es': ('es', 'UD_Spanish', 'es-ud'),
    'cs': ('cs', 'UD_Czech', 'cs-ud'),
    'fr': ('fr', 'UD_French', 'fr-ud'),
}


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


@experiment.command
def train(_run, _config, seed, batch_size, lstm_hidden_layer, language, epochs):
    print(_config)
    np.random.seed(seed)
    torch.manual_seed(seed)

    language = LanguageDataset(*languages[language])

    train_sentences = [(instance.sentence, instance.chars, instance.substrings) for instance in language.training_instances]
    train_tags = [instance.tags for instance in language.training_instances]

    dev_sentences = [(instance.sentence, instance.chars, instance.substrings) for instance in language.dev_instances]
    dev_tags = [instance.tags for instance in language.dev_instances]

    test_sentences = [(instance.sentence, instance.chars, instance.substrings) for instance in language.test_instances]
    test_tags = [instance.tags for instance in language.test_instances]

    train_dataset = list(zip(train_sentences, train_tags))
    dev_dataset = list(zip(dev_sentences, dev_tags))
    test_dataset = list(zip(test_sentences, test_tags))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_examples_multiple_tags
    )

    valid_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        collate_fn=collate_examples_multiple_tags
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_examples_multiple_tags
    )

    oovs = language.word_to_index.keys() - language.embeddings.keys()
    t_oovs = language.test_vocab.keys() & oovs
    print("Ratio OOVs: {} ({}/{})".format(len(oovs)/len(language.word_to_index), len(oovs), len(language.word_to_index)))

    # Compute the number of occurences of OOVs in the test set as well as the ratio
    oovs_ids = {language.word_to_index[o] for o in oovs}
    print(len([word_id for s, _, _ in train_sentences for word_id in s]))
    print(len([word_id for s, _, _ in dev_sentences for word_id in s]))
    print(len([word_id for s, _, _ in test_sentences for word_id in s]))
    num_words = len([word_id for s, _, _ in test_sentences for word_id in s])
    num_oovs = len([word_id for s, _, _ in test_sentences for word_id in s if word_id in oovs_ids])
    print("Ratio of occurrences of OOVs: {} ({}/{})".format(num_oovs/num_words, num_oovs, num_words))

    embedding_layer = MyEmbeddings(language.word_to_index, language.embedding_dim)
    embedding_layer.load_words_embeddings(language.embeddings)

    if _config["embeddings_mode"] == "random":
        # Leave OOV random embeddings
        comick = None

    elif _config["embeddings_mode"] == "mimick":
        # Fill OOV embeddings with Mimick's
        embed_path = "./data/mimick-embs/{}-lstm-est-embs.txt".format(language.polyglot_abbreviation)
        mimick_embeds = load_embeddings(embed_path)
        embedding_layer.load_words_embeddings(mimick_embeds)
        comick = None

    elif _config["embeddings_mode"] == "comick":
        embedding_layer_comick = MyEmbeddings(language.word_to_index, language.embedding_dim)
        embedding_layer_comick.load_words_embeddings(language.embeddings)

        if _config["oov_word_model"] == "mimick":
            oov_word_model = Mimick(
                characters_vocabulary=language.char_to_index,
                word_embeddings_dimension=language.embedding_dim
            )
        elif _config["oov_word_model"] == "bos":
            oov_word_model = BoS(
                language.bos_to_index,
                embedding_dim=language.embedding_dim,
            )

        comick = TheFinalComickBoS(
            embedding_layer_comick,
            oov_word_model,
            word_hidden_state_dimension=128,
            freeze_word_embeddings=False,
            attention=_config['attention']
        )


    char_model = CharRNN(
        language.char_to_index,
        _config["char_embedding_size"],
        lstm_hidden_layer,
    )

    model = SimpleLSTMTagger(
        char_model,
        embedding_layer,
        lstm_hidden_layer,
        {label: len(tags) for label, tags in language.tags_to_index.items()},
        oovs,
        comick,
        n=41
    )

    for name, parameter in model.named_parameters():
        if 'embedding_layer' not in name:
            if 'bias' in name:
                constant(parameter, 0)
            elif 'weight' in name:
                kaiming_normal(parameter)

    model_name = "{}".format(language.polyglot_abbreviation)
    expt_name = './expt_{}_{}_{}'.format(model_name, _config["embeddings_mode"], os.environ['DB_NAME'])
    expt_dir = get_experiment_directory(expt_name)

    device_id = _config["device"]
    device = None
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id) # Fix bug where memory is allocated on GPU0 when ask to take GPU1.
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:%d' % device_id)
        logging.info("Training on GPU %d" % device_id)
    else:
        logging.info("Training on CPU")


    optimizer = torch.optim.Adam(model.parameters(), lr=_config["learning_rate"])
    expt = PytouneExperiment(
        expt_dir,
        model,
        device=device,
        optimizer=optimizer,
        monitor_metric='val_acc',
        monitor_mode='max'
    )

    callbacks = [
        ClipNorm(model.parameters(), _config["gradient_clipping"]),
        ReduceLROnPlateau(monitor='val_acc', mode='max', patience=_config["reduce_lr_on_plateau"]["patience"], factor=_config["reduce_lr_on_plateau"]["factor"], threshold_mode='abs', threshold=1e-3, verbose=True),
        EarlyStopping(patience=_config["early_stopping"]["patience"], min_delta=1e-4, monitor='val_acc', mode='max'),
        MetricsCallback(_run)
    ]

    try:
        expt.train(train_loader, valid_loader, callbacks=callbacks, seed=42, epochs=epochs)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    print("Testing on test set...")
    metrics = expt.test(test_loader)

    vectors = torch_to_numpy(model.embedding_layer.weight)

    all_preds = list()
    all_trues = list()
    for x, ys in test_loader:
        preds, attention = expt.model.predict_on_batch(x)
        for tag, y in preds.items():
            if tag is not 'POS':
                all_pred = np.argmax(y, axis=2).reshape(-1)
                all_true = ys[tag].view(-1)
                for y_pred, y_true in zip(all_pred, all_true):
                    if y_true != 0 and y_true != 1:
                        all_preds.append(y_pred)
                        all_trues.append(y_true.item())
    if len(all_preds) > 0:
        f1 = f1_score(all_preds, all_trues, average='micro')
        metrics['f1'] = f1
        print("Precision score: {}".format(precision_score(all_preds, all_trues, average='micro')))
        print("Recall score: {}".format(recall_score(all_preds, all_trues, average='micro')))
        print("F1 score: {}".format(f1))
        print(classification_report(all_trues, all_preds, digits=4))

    pred_morph_per_oovs = defaultdict(list)
    true_morph_per_oovs = defaultdict(list)
    stats_pos_per_oovs = defaultdict(list)
    attention_analysis = defaultdict(list)
    for x, ys in test_loader:
        sentence = x[0]
        preds, attentions = expt.model.predict_on_batch(x)
        for tag, y in preds.items():
            if tag is 'POS':
                all_pred = np.argmax(y, axis=2).reshape(-1)
                all_true = ys[tag].view(-1)
                all_sentence = sentence.view(-1)
                attention_pred_tag = np.argmax(y, axis=2)
                attention_true_tag = ys[tag]
                for token, y_pred, y_true in zip(all_sentence, all_pred, all_true):
                    token_value = language.idx_to_word[token.item()]
                    if token_value in oovs:
                        if y_pred == y_true.item():
                            stats_pos_per_oovs[token_value].append(1)
                        else:
                            stats_pos_per_oovs[token_value].append(0)
            else:
                all_pred = np.argmax(y, axis=2).reshape(-1)
                all_true = ys[tag].view(-1)
                all_sentence = sentence.view(-1)
                for token, y_pred, y_true in zip(all_sentence, all_pred, all_true):
                    if y_true != 0 and y_true != 1:
                        token_value = language.idx_to_word[token.item()]
                        if token_value in oovs:
                            true_morph_per_oovs[token_value].append(y_true.item())
                            pred_morph_per_oovs[token_value].append(y_pred)
        for sent_idx, _, word_idx, embedding, attention in attentions:
            s = sentence[sent_idx]
            target_word = language.idx_to_word[s[word_idx].item()]
            sims = cos_matrix_multiplication(vectors, embedding)
            most_similar_word = language.idx_to_word[np.argmax(sims)]
            most_similar_word_sim = np.max(sims)
            s_to_words = " ".join([language.idx_to_word[w.item()] for w in s if w.item() > 0])
            result = attention_pred_tag[sent_idx][word_idx] == attention_true_tag[sent_idx][word_idx]

            formatted_attention = []
            l, w, r = attention
            l1 = l.reshape(-1).tolist()
            w1 = w.reshape(-1).tolist()
            r1 = r.reshape(-1).tolist()
            formatted_attention += [l1, w1, r1]

            attention_analysis[target_word.replace('.', '<DOT>')].append((
                target_word, most_similar_word, float(most_similar_word_sim), int(word_idx), formatted_attention, s_to_words, int(result.item())
            ))

    metrics['attention'] = attention_analysis
    metrics['pos_per_oov'] = dict()

    # for target_word, occurrences in attention_analysis.items():
    #     print("="*80)
    #     print("TARGET WORD: {}".format(target_word))
    #     for target_word, sim_word, sim_word_sim, word_idx, attention, sentence, result in occurrences:
    #         print("{}\t({})\t{}\t{}\n{}\n{}\t({})".format(target_word, word_idx, "\t".join([str(a) for a in attention]), result, sentence, sim_word, sim_word_sim))
    #         print()
    #     print("="*80)

    all_occurrences = list()
    for oov, occurrences in stats_pos_per_oovs.items():
        oov = oov.replace('.', '<DOT>') # For mongodb
        all_occurrences += occurrences
        metrics['pos_per_oov'][oov] = dict()
        metrics['pos_per_oov'][oov]['percent'] = sum(occurrences)/float(len(occurrences))
        metrics['pos_per_oov'][oov]['num'] = len(occurrences)
    metrics['pos_per_oov']['total'] = dict()
    metrics['pos_per_oov']['total']['percent'] = sum(all_occurrences)/float(len(all_occurrences))
    metrics['pos_per_oov']['total']['num'] = len(all_occurrences)
    print("OOV acc rate: {}".format(metrics['pos_per_oov']['total']['percent'], metrics['pos_per_oov']['total']['num']))

    all_occurrences = list()
    all_true_occurrences = list()
    for oov, preds in pred_morph_per_oovs.items():
        all_occurrences += preds
        all_true_occurrences += true_morph_per_oovs[oov]
    if len(all_occurrences) > 0:
        print("OOV Precision rate: {}".format(precision_score(all_occurrences, all_true_occurrences, average='micro')))
        print("OOV Recall rate: {}".format(recall_score(all_occurrences, all_true_occurrences, average='micro')))
        print("OOV F1 rate: {}".format(f1_score(all_occurrences, all_true_occurrences, average='micro')))

    all_stats = {
        'model': model_name,
        'metrics': metrics
    }
    collection.insert_one(all_stats)


@experiment.automain
def main(_config):
    for language in languages:
        _config['language'] = language
        run = experiment.run_command('train', config_updates=_config)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
