import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from downstream_task.sequence_tagging import sequence_cross_entropy, acc

from utils import pad_sequences


class Module(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parameters(self, requires_grad_only=True):
        """
        Overloads the parameters iterator function so only variable 'requires_grad' set to True are iterated over.
        """
        filter_cond = lambda param: param.requires_grad if requires_grad_only else True
        return (param for param in super().parameters() if filter_cond(param))

    def named_parameters(self, memo=None, prefix='', requires_grad_only=True):
        """
        Overloads the parameters iterator function so only variable 'requires_grad' set to True are iterated over.
        """
        filter_cond = lambda param: param.requires_grad if requires_grad_only else True
        return ((name, param) for (name, param) in super().named_parameters() if filter_cond(param))

    def reset_requires_grad_to_true(self):
        for param in super().parameters():
            param.requires_grad = True


class LSTMSequence(Module):
    def __init__(
            self,
            words_embedding_dimension,
            words_hidden_dimension,
            words_vocabulary,
            tagset_size,
            comick,
            oov_words,
            n,
            use_cuda=False,
            freeze_comick=False
    ):
        super(LSTMSequence, self).__init__()
        self.n_gram = n
        self.use_cuda = use_cuda
        self.words_embedding_dimension = words_embedding_dimension
        self.words_hidden_dimension = words_hidden_dimension
        self.words_vocabulary_size = len(words_vocabulary)
        self.words_vocabulary = words_vocabulary
        self.idx_to_word = {i: word for word, i in self.words_vocabulary.items()}
        self.oov_words = oov_words
        self.comick = comick
        if freeze_comick:
            for p in self.comick.parameters(requires_grad_only=False):
                p.requires_grad = False

        self.word_embeddings = nn.Embedding(self.words_vocabulary_size, words_embedding_dimension)
        self.word_lstm = nn.LSTM(
            words_embedding_dimension,
            words_hidden_dimension,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        self.hidden2tag = nn.Linear(words_hidden_dimension*2, tagset_size)

    def set_item_embedding(self, idx, embedding):
        t = torch.FloatTensor(embedding)
        if self.use_cuda:
            t = t.cuda()
        self.word_embeddings.weight.data[idx] = t

    def load_words_embeddings(self, words_embeddings):
        for word, embedding in words_embeddings.items():
            if word in self.words_vocabulary:
                idx = self.words_vocabulary[word]
                self.set_item_embedding(idx, embedding)

    def make_ngram(self, sequence, i):
        L = len(sequence)
        m = self.n_gram // 2
        left_idx = max(0, i - m)
        left_side = tuple(sequence[left_idx:i]) if i != 0 else tuple(Variable(torch.LongTensor([2])))
        right_idx = min(L, i + m + 1)
        right_side = tuple(sequence[i+1:right_idx]) if i != L-1 else tuple(Variable(torch.LongTensor([3])))
        return left_side, right_side

    def get_oov(self, sentences):
        """
        Returns for each batch for each sentences oov words to predict
        :param sentences:
        :return:
        """
        words_to_drop = list()
        for si, sentence in enumerate(sentences):
            sent_length = sentence.data.ne(0).long().sum()
            for i, idx in enumerate(sentence):
                word = self.idx_to_word[int(idx.data[0])]
                if word in self.oov_words:
                    left_context, right_context = self.make_ngram(sentence[:sent_length], i)
                    left_context = [c.view(1) for c in left_context]
                    right_context = [c.view(1) for c in right_context]
                    words_to_drop.append((si, i, word, torch.cat(left_context), torch.cat(right_context)))
        return words_to_drop

    def predict_embeddings(self, words_to_drop):
        batches_i, sents_i, words, left_contexts, right_contexts = list(zip(*words_to_drop))

        vectorized_words = [[self.comick.characters_vocabulary[c] for c in w] for w in words]
        words_lengths = torch.LongTensor([len(w) for w in words])
        padded_words = pad_sequences(vectorized_words, words_lengths)

        vectorized_left_contexts = [l.data for l in left_contexts]
        left_contexts_length = torch.LongTensor([len(c) for c in left_contexts])
        padded_left = pad_sequences(vectorized_left_contexts, left_contexts_length)

        vectorized_right_contexts = [l.data for l in right_contexts]
        right_contexts_length = torch.LongTensor([len(c) for c in right_contexts])
        padded_right = pad_sequences(vectorized_right_contexts, right_contexts_length)

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            padded_left = padded_left.cuda()
            padded_words = padded_words.cuda()
            padded_right = padded_right.cuda()

        embeddings = self.comick((Variable(padded_left), Variable(padded_words), Variable(padded_right)))
        attentions = self.comick.get_attentions()

        for si, i, embedding, attention in zip(batches_i, sents_i, embeddings, attentions):
            yield (si, i, embedding, attention)

    def forward(self, sentence):
        raise NotImplementedError()


class LSTMTagger(LSTMSequence):

    def __init__(
            self,
            words_embedding_dimension,
            words_hidden_dimension,
            words_vocabulary,
            tagset_size,
            comick,
            oov_words,
            n,
            use_cuda=False,
            freeze_comick=False
    ):
        super(LSTMTagger, self).__init__(
            words_embedding_dimension,
            words_hidden_dimension,
            words_vocabulary,
            tagset_size,
            comick,
            oov_words,
            n,
            use_cuda,
            freeze_comick
        )
        self.attentions = list()

    def forward(self, sentence):
        # Sort sentences in decreasing order
        lengths = sentence.data.ne(0).long().sum(dim=1)
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        _, rev_perm_idx = perm_idx.sort(0)
        sentence_sorted = sentence[perm_idx]

        oov_to_predict = self.get_oov(sentence_sorted)
        if len(oov_to_predict) > 0:
            embeddings_to_replace = list(self.predict_embeddings(oov_to_predict))

        embeds = self.word_embeddings(sentence_sorted)

        self.attentions = list()
        if len(oov_to_predict) > 0:
            for si, i, embed, attention in embeddings_to_replace:
                embeds[si, i] = embed
                self.attentions.append([perm_idx[si], i, attention])

        packed_input = pack_padded_sequence(embeds, list(seq_lengths), batch_first=True)
        packed_output, (hidden_states, cell_states) = self.word_lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        lstm_out = lstm_out[rev_perm_idx]
        tag_space = self.hidden2tag(lstm_out)

        return tag_space


class LSTMClassifier(LSTMSequence):

    def __init__(
            self,
            words_embedding_dimension,
            words_hidden_dimension,
            words_vocabulary,
            tagset_size,
            comick,
            oov_words,
            n,
            use_cuda=False,
            freeze_comick=False
    ):
        super(LSTMClassifier, self).__init__(
            words_embedding_dimension,
            words_hidden_dimension,
            words_vocabulary,
            tagset_size,
            comick,
            oov_words,
            n,
            use_cuda,
            freeze_comick
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, sentence):
        # Sort sentences in decreasing order
        lengths = sentence.data.ne(0).long().sum(dim=1)
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        _, rev_perm_idx = perm_idx.sort(0)
        sentence = sentence[perm_idx]

        oov_to_predict = self.get_oov(sentence)
        if len(oov_to_predict) > 0:
            embeddings_to_replace = list(self.predict_embeddings(oov_to_predict))

        embeds = self.word_embeddings(sentence)

        if len(oov_to_predict) > 0:
            for si, i, embed in embeddings_to_replace:
                embeds[si, i] = embed
        # embeds = self.dropout(embeds)

        packed_input = pack_padded_sequence(embeds, list(seq_lengths), batch_first=True)
        _, (hidden_states, cell_states) = self.word_lstm(packed_input)
        lstm_out = torch.cat([hidden_states[0], hidden_states[1]], dim=1)
        lstm_out = lstm_out[rev_perm_idx]
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out)

        return tag_space


class BOWClassifier(nn.Module):

    def __init__(
            self,
            words_embedding_dimension,
            words_vocabulary,
            tagset_size
    ):
        super(BOWClassifier, self).__init__()
        self.words_embedding_dimension = words_embedding_dimension
        self.words_vocabulary_size = len(words_vocabulary)
        self.words_vocabulary = words_vocabulary
        self.word_embeddings = nn.Embedding(self.words_vocabulary_size, words_embedding_dimension)
        self.dropout = nn.Dropout(0.5)
        self.hidden2tag = nn.Linear(words_embedding_dimension, tagset_size)

    def set_item_embedding(self, idx, embedding):
        self.word_embeddings.weight.data[idx] = torch.FloatTensor(embedding)

    def load_words_embeddings(self, words_embeddings):
        for word, embedding in words_embeddings.items():
            if word in self.words_vocabulary:
                idx = self.words_vocabulary[word]
                self.set_item_embedding(idx, embedding)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = self.dropout(embeds)
        bow = embeds.sum(dim=1)
        bow = F.sigmoid(bow)
        tag_space = self.hidden2tag(bow)
        return tag_space


class SimpleLSTMTagger(nn.Module):

    def __init__(self, embedding_layer, hidden_dim, tags):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.hidden_dim = hidden_dim
        self.num_layers = 2
        self.tags = tags
        # self.word_lstm = nn.LSTM(
        #     self.embedding_layer.embedding_dim,
        #     self.hidden_dim,
        #     batch_first=True,
        #     bidirectional=True,
        #     num_layers=self.num_layers,
        #     dropout=0.5
        # )

        self.lstms = nn.ModuleDict()
        self.hidden2tags = nn.ModuleDict()
        for tag, tagset_size in tags.items():
            self.lstms.add_module(
                tag,
                nn.LSTM(
                    self.embedding_layer.embedding_dim,
                    self.hidden_dim,
                    batch_first=True,
                    bidirectional=True,
                    num_layers=self.num_layers,
                    dropout=0.5
                )
            )
            self.hidden2tags.add_module(
                tag,
                nn.Linear(self.hidden_dim*2, tagset_size)
            )

        self.metrics = [self.acc]


    def loss_function(self, y_pred, y):
        losses = list()
        for label, output in y_pred.items():
            loss = sequence_cross_entropy(output, y[label])
            losses.append(loss)
        return sum(losses)

    def acc(self, y_pred, y):
        return acc(y_pred['POS'], y['POS'])

    def forward(self, input):
        sentence, tags_to_produce = input
        # Sort sentences in decreasing order
        lengths = sentence.data.ne(0).long().sum(dim=1)
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        _, rev_perm_idx = perm_idx.sort(0)
        sentence_sorted = sentence[perm_idx]

        embeds = self.embedding_layer(sentence_sorted)

        outputs = dict()
        for label in tags_to_produce:
            packed_input = pack_padded_sequence(embeds, list(seq_lengths), batch_first=True)
            packed_output, (hidden_states, cell_states) = self.lstms[label](packed_input)
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
            lstm_out = lstm_out[rev_perm_idx]
            tag_space = self.hidden2tags[label](lstm_out)
            outputs[label] = tag_space

        return outputs
