import logging
import os
import collections
from collections import defaultdict
from polyglot import text

import torch
from torch import nn

from torch.utils.data import DataLoader
from downstream_task.sequence_tagging import collate_examples
from downstream_task.models import SimpleLSTMTagger

from pytoune.framework import Experiment as PytouneExperiment
from pytoune.framework.callbacks import ClipNorm, ReduceLROnPlateau


UNK_TAG = "<UNK>"
NONE_TAG = "<NONE>"
START_TAG = "<START>"
END_TAG = "<STOP>"
PADDING_WORD = "<PAD>"
PADDING_CHAR = "<*>"
POS_KEY = "POS"

Instance = collections.namedtuple("Instance", ["source", "sentence", "tags"])

class MyEmbeddings(nn.Embedding):
    def __init__(self, word_to_idx, embedding_dim):
        super(MyEmbeddings, self).__init__(len(word_to_idx), embedding_dim, padding_idx=0)
        self.embedding_dim = embedding_dim
        self.vocab_size = len(word_to_idx)
        self.word_to_idx = word_to_idx

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
            'ud_tags': False,
            'no_morphotags': False
        }

        self.word_to_index = {} # mapping from word to index
        self.tags_to_index = {} # mapping from attribute name to mapping from tag to index
        self.char_to_index = {} # mapping from character to index, for char-RNN concatenations

        # Add special tokens / tags / chars to dicts
        self.word_to_index[PADDING_WORD] = len(self.word_to_index)
        self.word_to_index[UNK_TAG] = len(self.word_to_index)
        # for t2i in t2is.values():
        #     t2i[START_TAG] = len(t2i)
        #     t2i[END_TAG] = len(t2i)
        self.char_to_index[PADDING_CHAR] = len(self.char_to_index)

        self.embedding_dim = None
        self.__get_embeddings()
        self.__parse_dataset()

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
            self.options
        )

        self.dev_instances, self.dev_vocab = read_file(
            self.dev_path,
            self.word_to_index,
            self.tags_to_index,
            self.char_to_index,
            self.options
        )

        self.test_instances, self.test_vocab = read_file(
            self.test_path,
            self.word_to_index,
            self.tags_to_index,
            self.char_to_index,
            self.options
        )


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



def read_file(filename, w2i, t2is, c2i, options):
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
                        seq.extend([0] * (slen - len(seq))) # 0 guaranteed below to represent NONE_TAG

                # add sentence to dataset
                instances.append(Instance(source, sentence, tags))
                source = []
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
                for c in word:
                    if c not in c2i:
                        c2i[c] = len(c2i)
                for key, val in morphotags.items():
                    if key not in t2is:
                        t2is[key] = {NONE_TAG:0}
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
                    mtags.extend([0] * missing_tags) # 0 guaranteed above to represent NONE_TAG
                    mtags.append(t2is[k][v])

    return instances, vocab_counter


def main():
    languages = [
        # LanguageDataset('kk', 'UD_Kazakh', 'kk-ud'),
        # LanguageDataset('ta', 'UD_Tamil', 'ta-ud'),
        # LanguageDataset('lv', 'UD_Latvian', 'lv-ud'),
        # LanguageDataset('vi', 'UD_Vietnamese', 'vi-ud'),
        # LanguageDataset('hu', 'UD_Hungarian', 'hu-ud'),
        # LanguageDataset('tr', 'UD_Turkish', 'tr-ud'),
        # LanguageDataset('el', 'UD_Greek', 'el-ud'),
        # LanguageDataset('bg', 'UD_Bulgarian', 'bg-ud'),
        # LanguageDataset('sv', 'UD_Swedish', 'sv-ud'),
        # LanguageDataset('eu', 'UD_Basque', 'eu-ud'),
        # LanguageDataset('ru', 'UD_Russian', 'ru-ud'),
        # LanguageDataset('da', 'UD_Danish', 'da-ud'),
        # LanguageDataset('id', 'UD_Indonesian', 'id-ud'),
        # LanguageDataset('zh', 'UD_Chinese', 'zh-ud'),
        # LanguageDataset('fa', 'UD_Persian', 'fa-ud'),
        # LanguageDataset('he', 'UD_Hebrew', 'he-ud'),
        # LanguageDataset('ro', 'UD_Romanian', 'ro-ud'),
        # LanguageDataset('en', 'UD_English', 'en-ud'),
        LanguageDataset('ar', 'UD_Arabic', 'ar-ud'),
        # LanguageDataset('hi', 'UD_Hindi', 'hi-ud'),
        # LanguageDataset('it', 'UD_Italian', 'it-ud'),
        # LanguageDataset('es', 'UD_Spanish', 'es-ud'),
        # LanguageDataset('cs', 'UD_Czech', 'cs-ud'),
    ]

    for language in languages:

        train_sentences = [instance.sentence for instance in language.training_instances]
        train_tags = [instance.tags['POS'] for instance in language.training_instances]

        dev_sentences = [instance.sentence for instance in language.dev_instances]
        dev_tags = [instance.tags['POS'] for instance in language.dev_instances]

        test_sentences = [instance.sentence for instance in language.test_instances]
        test_tags = [instance.tags['POS'] for instance in language.test_instances]

        train_dataset = list(zip(train_sentences, train_tags))
        dev_dataset = list(zip(dev_sentences, dev_tags))
        test_dataset = list(zip(test_sentences, test_tags))

        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_examples
        )

        valid_loader = DataLoader(
            dev_dataset,
            batch_size=32,
            collate_fn=collate_examples
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            collate_fn=collate_examples
        )

        embedding_layer = MyEmbeddings(language.word_to_index, language.embedding_dim)
        embedding_layer.load_words_embeddings(language.embeddings)

        model = SimpleLSTMTagger(
            embedding_layer,
            128,
            len(language.tags_to_index['POS'])
        )

        model_name = "{}".format(language.polyglot_abbreviation)
        expt_name = './expt_{}'.format(model_name)
        expt_dir = get_experiment_directory(expt_name)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
        expt = PytouneExperiment(
            expt_dir,
            model,
            device=None,
            optimizer=optimizer,
            monitor_metric='val_loss',
            monitor_mode='min'
        )

        callbacks = [
            ClipNorm(model.parameters(), 0.25),
            ReduceLROnPlateau(monitor='val_loss', mode='min', patience=20, factor=0.5, threshold_mode='abs', threshold=1e-3, verbose=True),
        ]

        try:
            expt.train(train_loader, valid_loader, callbacks=callbacks, seed=42, epochs=1000)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        print("Testing on test set...")
        expt.test(test_loader)





if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
