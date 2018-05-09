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
                 freeze_embeddings=False,
                 dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

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
                bidirectional=True,
                dropout=dropout)
            setattr(self, 'lstm' + str(i), lstm)  # To support 'parameters()'
            self.lstms.append(lstm)

    def forward(self, *xs):
        """
        xs is a tuple of sequences of items. Must be the same length as 'n_lstms'. Returns a list of outputs if there is more than one sequence.
        """
        outputs = []
        for x, lstm in zip(xs, self.lstms):
            lengths = x.data.ne(0).long().sum(dim=1)
            seq_lengths, perm_idx = lengths.sort(0, descending=True)
            _, rev_perm_idx = perm_idx.sort(0)

            # Embed
            embeddings = self.embeddings(x[perm_idx])
            embeddings = self.dropout(embeddings)

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


class MirrorLSTM(Module):
    """
    Module that converts two sequences of items into their common embeddings, then applies a bidirectional LSTM on each sequence. The outputs are the concatenations of the two final hidden states.
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 hidden_state_dim,
                 padding_idx=0,
                 freeze_embeddings=True,
                 dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.embeddings = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=0)
        kaiming_uniform(self.embeddings.weight)
        if freeze_embeddings:
            for param in self.embeddings.parameters():
                print('Freezing embeddings')
                param.requires_grad = False

        self.lstm_left = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_state_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.lstm_right = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_state_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.lstms = {'left': self.lstm_left,
                      'right': self.lstm_right}

    def forward(self, x_left, x_right):

        x = {'left': x_left,
             'right': x_right}

        outputs = []
        for side in ['left', 'right']:
            lengths = x[side].data.ne(0).long().sum(dim=1)
            seq_lengths, perm_idx = lengths.sort(0, descending=True)
            _, rev_perm_idx = perm_idx.sort(0)

            # Embed
            embeddings = self.embeddings(x[side][perm_idx])
            embeddings = self.dropout(embeddings)

            # Initialize hidden to zero
            packed_input = pack_padded_sequence(
                embeddings, list(seq_lengths), batch_first=True)
            packed_output, (hidden_states, cell_states) = self.lstms[side](
                packed_input)

            if side == 'left':
                # Concatenate [forward, backward]
                output = torch.cat([hidden_states[0], hidden_states[1]], dim=1)
            else:
            # Concatenate [backward, forward]
                output = torch.cat([hidden_states[1], hidden_states[0]], dim=1)
            output = output[rev_perm_idx]
            outputs.append(output)

        return outputs

    def set_item_embedding(self, idx, embedding):
        self.embeddings.weight.data[idx] = torch.FloatTensor(embedding)


class Comick(Module):
    """
    This is the current architecture.
    """

    def __init__(self,
                 characters_vocabulary: Dict[str, int],
                 words_vocabulary: Dict[str, int],
                 characters_embedding_dimension=20,
                 word_embeddings_dimension=50,
                 words_embeddings=None,
                 freeze_word_embeddings=True,
                 context_dropout_p=0,
                 fc_dropout_p=0.5,
                 ):
        super().__init__()

        self.words_vocabulary = words_vocabulary
        self.characters_vocabulary = characters_vocabulary

        self.contexts = MirrorLSTM(num_embeddings=len(self.words_vocabulary),
                                   embedding_dim=word_embeddings_dimension,
                                   hidden_state_dim=word_embeddings_dimension,
                                   freeze_embeddings=freeze_word_embeddings,
                                   dropout=context_dropout_p)

        if words_embeddings != None:
            self.load_words_embeddings(words_embeddings)

        self.mimick_lstm = MultiLSTM(num_embeddings=len(self.characters_vocabulary),
                                     embedding_dim=characters_embedding_dimension,
                                     hidden_state_dim=word_embeddings_dimension)

        self.fc_context = nn.Linear(in_features=2 * word_embeddings_dimension,
                                    out_features=word_embeddings_dimension)
        kaiming_uniform(self.fc_context.weight)

        self.fc_word = nn.Linear(in_features=2 * word_embeddings_dimension,
                                 out_features=word_embeddings_dimension)
        kaiming_uniform(self.fc_word.weight)

        self.fc_output = nn.Linear(in_features=2 * word_embeddings_dimension,
                                   out_features=word_embeddings_dimension)
        kaiming_uniform(self.fc_output.weight)

        self.dropout = nn.Dropout(p=fc_dropout_p)

    def load_words_embeddings(self, words_embeddings):
        for word, embedding in words_embeddings.items():
            if word in self.words_vocabulary:
                idx = self.words_vocabulary[word]
                self.contexts.set_item_embedding(idx, embedding)

    def forward(self, x):
        left_context, word, right_context = x
        left_rep, right_rep = self.contexts(left_context, right_context)
        context_rep = self.fc_context(left_rep + right_rep)
        word_hidden_rep = self.fc_word(self.mimick_lstm(word))
        output = torch.cat((context_rep, word_hidden_rep), dim=1)
        output = self.dropout(output)
        output = F.tanh(output)
        output = self.fc_output(output)

        return output


class Mimick(Module):
    """
    This is the Mimick architecture.
    """

    def __init__(self,
                 characters_vocabulary: Dict[str, int],
                 characters_embedding_dimension=20,
                 word_embeddings_dimension=50,
                 fc_dropout_p=0.5,
                 comick_compatibility=True
                 ):
        super().__init__()
        self.characters_vocabulary = characters_vocabulary
        self.comick_compatibility = comick_compatibility

        self.mimick_lstm = MultiLSTM(
            num_embeddings=len(self.characters_vocabulary),
            embedding_dim=characters_embedding_dimension,
            hidden_state_dim=word_embeddings_dimension
        )

        self.fc_word = nn.Linear(
            in_features=2 * word_embeddings_dimension,
            out_features=word_embeddings_dimension
        )
        kaiming_uniform(self.fc_word.weight)

        self.fc_output = nn.Linear(
            in_features=word_embeddings_dimension,
            out_features=word_embeddings_dimension
        )
        kaiming_uniform(self.fc_output.weight)

        self.dropout = nn.Dropout(p=fc_dropout_p)

    def forward(self, x):
        if self.comick_compatibility:
            _CL, x, _CR = x
        word_hidden_rep = self.fc_word(self.mimick_lstm(x))
        output = self.dropout(word_hidden_rep)
        output = F.tanh(output)
        output = self.fc_output(output)
        return output
