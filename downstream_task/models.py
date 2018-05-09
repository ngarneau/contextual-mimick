import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from comick import ComickDev
from utils import pad_sequences


class LSTMSequence(nn.Module):
    def __init__(
            self,
            words_embedding_dimension,
            words_hidden_dimension,
            words_vocabulary,
            words_wo_embedding,
            tagset_size,
            use_cuda=False
    ):
        super(LSTMSequence, self).__init__()
        self.use_cuda = use_cuda
        self.words_embedding_dimension = words_embedding_dimension
        self.words_hidden_dimension = words_hidden_dimension
        self.words_wo_embedding = words_wo_embedding
        self.words_vocabulary_size = len(words_vocabulary)
        self.words_vocabulary = words_vocabulary
        self.idx_to_word = {i: word for word, i in self.words_vocabulary.items()}
        self.word_dropout_num = 2
        self.n_gram = 21

        self.word_embeddings = nn.Embedding(self.words_vocabulary_size, words_embedding_dimension)
        self.word_lstm = nn.LSTM(
            words_embedding_dimension,
            words_hidden_dimension,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        self.hidden2tag = nn.Linear(words_hidden_dimension * 2, tagset_size)

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

    def forward(self, sentence):
        raise NotImplementedError()


class LSTMTagger(LSTMSequence):

    def __init__(
            self,
            words_embedding_dimension,
            words_hidden_dimension,
            words_vocabulary,
            words_wo_embedding,
            tagset_size,
            comick: ComickDev,
            use_cuda=False
    ):
        super(LSTMTagger, self).__init__(words_embedding_dimension, words_hidden_dimension, words_vocabulary,
                                         words_wo_embedding, tagset_size, use_cuda)
        self.comick = comick

    def make_ngram(self, sequence, i):
        pad_left = 0
        pad_right = 0
        L = len(sequence)
        m = self.n_gram // 2
        left_idx = max(0, i - m)
        left_side = tuple(sequence[left_idx:i]) if i != 0 else tuple(Variable(torch.LongTensor([1])))
        right_idx = min(L, i + m + 1)
        right_side = tuple(sequence[i+1:right_idx]) if i != L-1 else tuple(Variable(torch.LongTensor([2])))
        return (left_side, right_side)

    def word_dropout(self, sentences):
        """
        Returns for each batch for each sentences i words that have been
        dropped out and its embedding predicted. Return value is batch_num, i, embedding
        :param sentences:
        :return:
        """
        words_to_drop = list()
        for si, sentence in enumerate(sentences):
            sent_length = sentence.data.ne(0).long().sum()
            random_dropout_candidates = np.random.choice(sent_length, sent_length, replace=False)
            num_to_choose = self.word_dropout_num
            for i in random_dropout_candidates:
                if num_to_choose == 0:
                    break
                word = self.idx_to_word[sentence[i].data[0]]
                if word not in self.words_wo_embedding:
                    left_context, right_context = self.make_ngram(sentence[:sent_length], i)
                    words_to_drop.append((si, i, word, torch.cat(left_context), torch.cat(right_context)))
                    num_to_choose -= 1
        return words_to_drop

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
                word = self.idx_to_word[idx.data[0]]
                if word in self.words_wo_embedding:
                    left_context, right_context = self.make_ngram(sentence[:sent_length], i)
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

        embeddings = self.comick((Variable(padded_left), Variable(padded_words), Variable(padded_right)))

        for si, i, embedding in zip(batches_i, sents_i, embeddings):
            yield (si, i, embedding)

    def forward(self, sentence):
        # Sort sentences in decreasing order
        lengths = sentence.data.ne(0).long().sum(dim=1)
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        _, rev_perm_idx = perm_idx.sort(0)
        sentence = sentence[perm_idx]

        if self.training:
            words_to_drop = self.word_dropout(sentence)
            embeddings_to_replace = list(self.predict_embeddings(words_to_drop))
        else:
            # Predict embeddings for OOV
            oov_to_predict = self.get_oov(sentence)
            embeddings_to_replace = list(self.predict_embeddings(oov_to_predict))
            pass

        embeds = self.word_embeddings(sentence)
        # For each batch, indices, embedding pair, replace the word with its predicted embedding
        for si, i, embed in embeddings_to_replace:
            embeds[si, i] = embed

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
            use_cuda=False
    ):
        super(LSTMClassifier, self).__init__(words_embedding_dimension, words_hidden_dimension, words_vocabulary,
                                             tagset_size, use_cuda)
        self.dropout = nn.Dropout(0.5)

    def forward(self, sentence):
        # Sort sentences in decreasing order
        lengths = sentence.data.ne(0).long().sum(dim=1)
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        _, rev_perm_idx = perm_idx.sort(0)
        sentence = sentence[perm_idx]

        embeds = self.word_embeddings(sentence)
        embeds = self.dropout(embeds)

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
