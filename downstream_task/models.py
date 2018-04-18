import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMSequence(nn.Module):
    def __init__(
            self,
            words_embedding_dimension,
            words_hidden_dimension,
            words_vocabulary,
            tagset_size
    ):
        super(LSTMSequence, self).__init__()
        self.words_embedding_dimension = words_embedding_dimension
        self.words_hidden_dimension = words_hidden_dimension
        self.words_vocabulary_size = len(words_vocabulary)
        self.words_vocabulary = words_vocabulary

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
        self.word_embeddings.weight.data[idx] = torch.FloatTensor(embedding)

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
            tagset_size
    ):
        super(LSTMTagger, self).__init__(words_embedding_dimension, words_hidden_dimension, words_vocabulary, tagset_size)

    def forward(self, sentence):
        # Sort sentences in decreasing order
        lengths = sentence.data.ne(0).sum(dim=1).long()
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        _, rev_perm_idx = perm_idx.sort(0)
        sentence = sentence[perm_idx]

        embeds = self.word_embeddings(sentence)

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
            tagset_size
    ):
        super(LSTMClassifier, self).__init__(words_embedding_dimension, words_hidden_dimension, words_vocabulary, tagset_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, sentence):
        # Sort sentences in decreasing order
        lengths = sentence.data.ne(0).sum(dim=1).long()
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
