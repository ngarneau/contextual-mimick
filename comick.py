import torch
from torch import nn, autograd
from torch.nn import functional as F
from torch.nn.init import kaiming_uniform, constant
from torch.nn.utils.rnn import pack_padded_sequence

from typing import Dict


class Module(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parameters(self):
        """
        Overloads the parameters iterator function so only variable 'requires_grad' set to True are iterated over.
        """
        return (param for param in super().parameters() if param.requires_grad)


class MultiLSTM(Module):
    """
    Module that converts multiple sequences of items into their common embeddings, then applies a bidirectional LSTM on each sequence. The outputs are the concatenations of the two final hidden states.
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 hidden_state_dim,
                 n_lstms=1,
                 padding_idx=0,
                 freeze_embeddings=False):
        super().__init__()

        self.embeddings = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=0)
        kaiming_uniform(self.embeddings.weight)
        if freeze_embeddings:
            for param in self.embeddings.parameters():
                print('Freezing embeddings')
                param.requires_grad = False

        self.lstms = []
        for i in range(n_lstms):
            lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_state_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True)
            setattr(self, 'lstm' + str(i), lstm)  # To support 'parameters()'
            self.lstms.append(lstm)

    def forward(self, *xs):
        """
        xs is a tuple of sequences of items. Must be the same length as 'n_lstms'. Returns a list of outputs if there is more than one stream
        """
        outputs = []
        for x, lstm in zip(xs, self.lstms):
            lengths = x.data.ne(0).sum(dim=1).long()
            seq_lengths, perm_idx = lengths.sort(0, descending=True)
            _, rev_perm_idx = perm_idx.sort(0)

            # Embed
            embeddings = self.embeddings(x)

            # Initialize hidden to zero
            packed_input = pack_padded_sequence(
                embeddings, list(seq_lengths), batch_first=True)
            packed_output, (hidden_states, cell_states) = lstm(packed_input)
            output = torch.cat([hidden_states[0], hidden_states[1]], dim=1)
            output = output[rev_perm_idx]
            outputs.append(output)

        return outputs if len(outputs) > 1 else outputs[0]

    def set_item_embedding(self, idx, embedding):
        self.embeddings.weight.data[idx] = torch.FloatTensor(embedding)


class Context(MultiLSTM):
    """
    This Context module adds dropout and a fully connected layer to a MultiStream class.
    """

    def __init__(self, *args, hidden_state_dim, output_dim, n_contexts=1, dropout_p=0.5, **kwargs):
        super().__init__(*args, hidden_state_dim=hidden_state_dim, n_lstms=n_contexts, **kwargs)

        self.fcs = []
        for i in range(n_contexts):
            fc = nn.Linear(in_features=2 * hidden_state_dim,
                           out_features=output_dim)
            setattr(self, 'fc' + str(i), fc)
            self.fcs.append(fc)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, *xs):
        xs = super().forward(*xs)
        outputs = []
        for x, fc in zip(xs, self.fcs):
            outputs.append(fc(self.dropout(x)))
        return outputs if len(outputs) > 1 else outputs[0]


class LRComick(Module):
    """
    This is a re-implementation of our original Comick with right and left context.
    """

    def __init__(self,
                 characters_vocabulary: Dict[str, int],
                 words_vocabulary: Dict[str, int],
                 characters_embedding_dimension=20,
                 characters_hidden_state_dimension=50,
                 word_embeddings_dimension=50,
                 words_hidden_state_dimension=50,
                 words_embeddings=None,
                 fully_connected_layer_hidden_dimension=50,
                 freeze_word_embeddings=False,
                 ):
        super().__init__()

        self.words_vocabulary = words_vocabulary
        self.characters_vocabulary = characters_vocabulary

        self.contexts = Context(hidden_state_dim=words_hidden_state_dimension,
                                output_dim=2 * characters_hidden_state_dimension,
                                num_embeddings=len(self.words_vocabulary),
                                embedding_dim=word_embeddings_dimension,
                                n_contexts=2,
                                freeze_embeddings=freeze_word_embeddings)
        if words_embeddings != None:
            self.load_words_embeddings(words_embeddings)

        self.mimick = MultiLSTM(num_embeddings=len(self.characters_vocabulary),
                                embedding_dim=characters_embedding_dimension,
                                hidden_state_dim=characters_hidden_state_dimension)

        self.fc1 = nn.Linear(in_features=2 * characters_hidden_state_dimension,
                             out_features=fully_connected_layer_hidden_dimension)
        kaiming_uniform(self.fc1.weight)

        self.fc2 = nn.Linear(in_features=fully_connected_layer_hidden_dimension,
                             out_features=word_embeddings_dimension)
        kaiming_uniform(self.fc2.weight)

    def load_words_embeddings(self, words_embeddings):
        for word, embedding in words_embeddings.items():
            if word in self.words_vocabulary:
                idx = self.words_vocabulary[word]
                self.contexts.set_item_embedding(idx, embedding)

    def forward(self, x):
        left_context, word, right_context = x

        left_rep, right_rep = self.contexts(left_context, right_context)
        word_hidden_rep = self.mimick(word)
        hidden_rep = left_rep + word_hidden_rep + right_rep

        output = self.fc1(F.tanh(hidden_rep))
        output = self.fc2(F.tanh(output))

        return output


class ComickUniqueContext(Module):
    """
    This is the architecture with only one context.
    """

    def __init__(self,
                 characters_vocabulary: Dict[str, int],
                 words_vocabulary: Dict[str, int],
                 characters_embedding_dimension=20,
                 characters_hidden_state_dimension=50,
                 word_embeddings_dimension=50,
                 words_hidden_state_dimension=50,
                 words_embeddings=None,
                 freeze_word_embeddings=False,
                 ):
        super().__init__()

        self.words_vocabulary = words_vocabulary
        self.characters_vocabulary = characters_vocabulary

        self.context = Context(hidden_state_dim=words_hidden_state_dimension,
                               output_dim=2 * characters_hidden_state_dimension,
                               num_embeddings=len(self.words_vocabulary),
                               embedding_dim=word_embeddings_dimension,
                               freeze_embeddings=freeze_word_embeddings)
        if words_embeddings != None:
            self.load_words_embeddings(words_embeddings)

        self.mimick = MultiLSTM(num_embeddings=len(self.characters_vocabulary),
                                embedding_dim=characters_embedding_dimension,
                                hidden_state_dim=characters_hidden_state_dimension)

        self.fc = nn.Linear(in_features=2 * characters_hidden_state_dimension,
                            out_features=word_embeddings_dimension)
        kaiming_uniform(self.fc.weight)

    def load_words_embeddings(self, words_embeddings):
        for word, embedding in words_embeddings.items():
            if word in self.words_vocabulary:
                idx = self.words_vocabulary[word]
                self.context.set_item_embedding(idx, embedding)

    def forward(self, x):
        context, word = x

        context_rep = self.context(context)
        word_hidden_rep = self.mimick(word)
        hidden_rep = context_rep + hidden_word_rep

        output = self.fc(F.tanh(hidden_rep))

        return output


class ComickDev(Module):
    """
    This is the architecture in development.
    """
    def __init__(self,
                 characters_vocabulary: Dict[str, int],
                 words_vocabulary: Dict[str, int],
                 characters_embedding_dimension=20,
                 word_embeddings_dimension=50,
                 words_embeddings=None,
                 freeze_word_embeddings=True,
                 ):
        super().__init__()

        self.words_vocabulary = words_vocabulary
        self.characters_vocabulary = characters_vocabulary

        self.contexts = MultiLSTM(num_embeddings=len(self.words_vocabulary),
                                  embedding_dim=word_embeddings_dimension,
                                  hidden_state_dim=word_embeddings_dimension,
                                  n_lstms=2,
                                  freeze_embeddings=freeze_word_embeddings)

        if words_embeddings != None:
            self.load_words_embeddings(words_embeddings)

        self.mimick_lstm = MultiLSTM(num_embeddings=len(self.characters_vocabulary),
                                embedding_dim=characters_embedding_dimension,
                                hidden_state_dim=word_embeddings_dimension)

        self.fc1 = nn.Linear(in_features=4*word_embeddings_dimension,
                             out_features=word_embeddings_dimension)
        kaiming_uniform(self.fc1.weight)

        self.fc2 = nn.Linear(in_features=word_embeddings_dimension,
                             out_features=word_embeddings_dimension)
        kaiming_uniform(self.fc2.weight)

        self.dropout = nn.Dropout(p=0.3)

    def load_words_embeddings(self, words_embeddings):
        for word, embedding in words_embeddings.items():
            if word in self.words_vocabulary:
                idx = self.words_vocabulary[word]
                self.contexts.set_item_embedding(idx, embedding)

    def forward(self, x):
        left_context, word, right_context = x

        left_rep, right_rep = self.contexts(left_context, right_context)
        context_rep = left_rep + right_rep
        word_hidden_rep = self.mimick_lstm(word)
        output = word_hidden_rep
        # output = torch.cat((context_rep, word_hidden_rep), dim=1)
        # output = self.dropout(output)
        # output = F.tanh(output)
        output = self.fc1(output)
        # output = self.dropout(output)
        output = F.tanh(output)
        output = self.fc2(output)

        return output
