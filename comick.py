import torch
from torch import nn, autograd
from torch.nn import functional as F
from torch.nn.init import kaiming_uniform, kaiming_normal, constant
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from typing import Dict


def make_substrings(s, lmin=3, lmax=6) :
    s = '<' + s + '>'
    for i in range(len(s)) :
        s0 = s[i:]
        for j in range(lmin, 1 + min(lmax, len(s0))) :
            yield s0[:j]

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
            packed_input = pack_padded_sequence(embeddings, list(seq_lengths), batch_first=True)
            packed_output, (hidden_states, cell_states) = lstm(packed_input)
            padded_output, lengths = pad_packed_sequence(packed_output, batch_first=True)
            # output = torch.cat([hidden_states[0], hidden_states[1]], dim=1)
            last_output = torch.cat([hidden_states[0], hidden_states[1]], dim=1)
            output = padded_output[rev_perm_idx]
            last_output = last_output[rev_perm_idx]
            outputs.append((output, last_output))

        return outputs if len(outputs) > 1 else outputs[0]

    def set_item_embedding(self, idx, embedding):
        self.embeddings.weight.data[idx] = torch.FloatTensor(embedding)


class MirrorLSTM(Module):
    """
    Module that converts two sequences of items into their common embeddings, then applies a bidirectional LSTM on each sequence. The outputs are the concatenations of the two final hidden states.
    """

    def __init__(self,
                 embedding_layer,
                 hidden_state_dim,
                 padding_idx=0,
                 freeze_embeddings=True,
                 dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.embeddings = embedding_layer
        kaiming_uniform(self.embeddings.weight)
        if freeze_embeddings:
            for param in self.embeddings.parameters():
                print('Freezing embeddings')
                param.requires_grad = False

        self.lstm_left = nn.LSTM(
            input_size=embedding_layer.embedding_dim,
            hidden_size=hidden_state_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.lstm_right = nn.LSTM(
            input_size=embedding_layer.embedding_dim,
            hidden_size=hidden_state_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.lstms = nn.ModuleDict()
        self.lstms.add_module('left', self.lstm_left)
        self.lstms.add_module('right', self.lstm_right)

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
            packed_input = pack_padded_sequence(embeddings, list(seq_lengths), batch_first=True)
            packed_output, (hidden_states, cell_states) = self.lstms[side](packed_input)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)

            # if side == 'left':
            #     # Concatenate [forward, backward]
            #     output = torch.cat([hidden_states[0], hidden_states[1]], dim=1)
            # else:
            # # Concatenate [backward, forward]
            #     output = torch.cat([hidden_states[1], hidden_states[0]], dim=1)
            last_output = torch.cat([hidden_states[0], hidden_states[1]], dim=1)
            output = output[rev_perm_idx]
            outputs.append((output, last_output))

        return outputs

    def set_item_embedding(self, idx, embedding):
        self.embeddings.weight.data[idx] = torch.FloatTensor(embedding)


class Context(MultiLSTM):
    """
    This Context module adds dropout and a fully connected layer to a MultiLSTM class.
    """

    def __init__(self, *args, hidden_state_dim, output_dim, n_contexts=1, dropout_p=0.5, **kwargs):
        super().__init__(*args, hidden_state_dim=hidden_state_dim, **kwargs)

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
                 characters_embeddings=None,
                 word_embeddings_dimension=50,
                 words_hidden_state_dimension=50,
                 words_embeddings=None,
                 fully_connected_layer_hidden_dimension=50,
                 freeze_word_embeddings=False,
                 context_dropout_p=0,
                 ):
        super().__init__()
        self.version = 1.2
        self.words_vocabulary = words_vocabulary
        self.characters_vocabulary = characters_vocabulary

        self.contexts = MirrorLSTM(num_embeddings=len(self.words_vocabulary),
                                   embedding_dim=word_embeddings_dimension,
                                   hidden_state_dim=words_hidden_state_dimension,
                                   freeze_embeddings=freeze_word_embeddings,
                                   dropout=context_dropout_p)

        if words_embeddings is not None:
            self.load_words_embeddings(words_embeddings)

        self.mimick = MultiLSTM(num_embeddings=len(self.characters_vocabulary),
                                embedding_dim=characters_embedding_dimension,
                                hidden_state_dim=characters_hidden_state_dimension)

        if characters_embeddings is not None:
            self.load_chars_embeddings(characters_embeddings)

        self.fc1 = nn.Linear(in_features=2 * words_hidden_state_dimension,
                             out_features=word_embeddings_dimension)
        kaiming_uniform(self.fc1.weight)

        self.fc2 = nn.Linear(in_features=2 * characters_hidden_state_dimension,
                             out_features=word_embeddings_dimension)
        kaiming_uniform(self.fc2.weight)

        self.fc3 = nn.Linear(in_features=word_embeddings_dimension,
                             out_features=word_embeddings_dimension)
        kaiming_uniform(self.fc3.weight)

    def load_words_embeddings(self, words_embeddings):
        for word, embedding in words_embeddings.items():
            if word in self.words_vocabulary:
                idx = self.words_vocabulary[word]
                self.contexts.set_item_embedding(idx, embedding)

    def load_chars_embeddings(self, chars_embeddings):
        for word, embedding in chars_embeddings.items():
            if word in self.characters_vocabulary:
                idx = self.characters_vocabulary[word]
                self.mimick.set_item_embedding(idx, embedding)

    def forward(self, x):
        left_context, word, right_context = x

        left_rep, right_rep = self.contexts(left_context, right_context)
        context_rep = left_rep + right_rep
        context_rep = self.fc1(context_rep)

        word_hidden_rep = self.mimick(word)
        word_rep = self.fc2(F.tanh(word_hidden_rep))

        hidden_rep = context_rep + word_rep
        output = self.fc3(F.tanh(hidden_rep))

        return output


class LRComickContextOnly(LRComick):
    def forward(self, x):
        left_context, word, right_context = x
        left_rep, right_rep = self.contexts(left_context, right_context)
        context_rep = left_rep + right_rep
        context_rep = self.fc1(F.tanh(context_rep))
        # word_hidden_rep = self.mimick(word)
        # word_rep = self.fc2(F.tanh(word_hidden_rep))
        # hidden_rep = context_rep + word_rep
        output = self.fc3(F.tanh(context_rep))
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
        left_context, word, right_context = x
        context = torch.cat([left_context, right_context], dim=1)
        context_rep = self.context(context)
        word_hidden_rep = self.mimick(word)
        hidden_rep = context_rep + word_hidden_rep
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
                 context_dropout_p=0,
                 fc_dropout_p=0.5,
                 ):
        super().__init__()
        self.version = 2.1
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

        self.fc1 = nn.Linear(in_features=2 * word_embeddings_dimension, out_features=word_embeddings_dimension)
        kaiming_uniform(self.fc1.weight)

        self.fc2 = nn.Linear(in_features=word_embeddings_dimension, out_features=word_embeddings_dimension)
        kaiming_uniform(self.fc2.weight)

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
        output = self.fc1(output)
        # output = self.dropout(output)
        # output = F.tanh(output)
        # output = self.fc2(output)
        return output


class Mimick(Module):
    """
    This is Pinter's Mimick architecture.
    """

    def __init__(self,
                 characters_vocabulary: Dict[str, int],
                 characters_embedding_dimension=20,
                 word_embeddings_dimension=100,
                 hidden_state_dimension=128,
                 fc_dropout_p=0,
                 lstm_dropout=0,
                 comick_compatibility=False
                 ):
        super().__init__()
        self.version = 1.0
        self.characters_vocabulary = characters_vocabulary
        self.comick_compatibility = comick_compatibility

        self.mimick_lstm = MultiLSTM(
            num_embeddings=len(self.characters_vocabulary),
            embedding_dim=characters_embedding_dimension,
            hidden_state_dim=hidden_state_dimension,
            dropout=lstm_dropout,
        )

        self.fc_word = nn.Linear(
            in_features=2 * hidden_state_dimension,
            out_features=word_embeddings_dimension
        )

        self.fc_output = nn.Linear(
            in_features=word_embeddings_dimension,
            out_features=word_embeddings_dimension
        )

        self.dropout = nn.Dropout(p=fc_dropout_p)

    def vectorize_words(self, words):
        return [[self.characters_vocabulary[c] for c in w] for w in words]

    def forward(self, x):
        if self.comick_compatibility:
            _CL, x, _CR = x
        output = self.mimick_lstm(x)
        return output


class MimickV2(Module):
    """
    This is new Mimick architecture
    """

    def __init__(self,
                 characters_vocabulary: Dict[str, int],
                 characters_embedding_dimension=100,
                 context_size=128,
                 word_embeddings_dimension=100,
                 hidden_state_dimension=128,
                 fc_dropout_p=0,
                 lstm_dropout=0,
                 comick_compatibility=False
                 ):
        super().__init__()
        self.version = 2.0
        self.characters_vocabulary = characters_vocabulary
        self.context_size = context_size

        self.embeddings = nn.Embedding(
            num_embeddings=len(self.characters_vocabulary),
            embedding_dim=characters_embedding_dimension,
            padding_idx=0)
        kaiming_uniform(self.embeddings.weight)

        self.lstm = nn.LSTM(
                input_size=characters_embedding_dimension + self.context_size,
                hidden_size=hidden_state_dimension,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                )

    def vectorize_words(self, words):
        return [[self.characters_vocabulary[c] for c in w] for w in words]

    def forward(self, x, contexts):
        lengths = x.data.ne(0).long().sum(dim=1)
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        _, rev_perm_idx = perm_idx.sort(0)
        sorted_contexts = contexts[perm_idx]
        expanded_contexts = sorted_contexts.expand(
            x.shape[1], x.shape[0], -1
        ).transpose(0, 1)

        # Embed
        embeddings = self.embeddings(x[perm_idx])

        lstm_input = torch.cat([embeddings, expanded_contexts], dim=2)

        # Initialize hidden to zero
        packed_input = pack_padded_sequence(lstm_input, list(seq_lengths), batch_first=True)
        packed_output, (hidden_states, cell_states) = self.lstm(packed_input)
        padded_output, lengths = pad_packed_sequence(packed_output, batch_first=True)
        # output = torch.cat([hidden_states[0], hidden_states[1]], dim=1)
        output = padded_output[rev_perm_idx]
        return output


class BoS(Module):
    def __init__(self, bos_vocabulary, embedding_dim):
        super().__init__()
        self.bos_vocabulary = bos_vocabulary
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(len(self.bos_vocabulary), self.embedding_dim, padding_idx=0)

    def vectorize_words(self, words):
        return [[self.bos_vocabulary[b] for b in make_substrings(w)] for w in words]

    def forward(self, bos):
        lengths = bos.data.ne(0).long().sum(dim=1)
        return self.embeddings(bos).sum(dim=1)/lengths.view(-1, 1).float()


class TheFinalComick(Module):
    def __init__(self,
                 characters_vocabulary: Dict[str, int],
                 words_vocabulary: Dict[str, int],
                 characters_embedding_dimension=20,
                 word_embeddings_dimension=100,
                 char_hidden_state_dimension=128,
                 word_hidden_state_dimension=128,
                 chars_embeddings=None,
                 words_embeddings=None,
                 lstm_dropout=.3,
                 freeze_word_embeddings=True,
                 freeze_mimick=False,
                 mimick_model_path='',
                 stats=None,
                 use_gpu=False):
        super().__init__()
        self.stats = stats
        self.words_vocabulary = words_vocabulary
        self.word_embeddings_dimension = word_embeddings_dimension
        self.characters_vocabulary = characters_vocabulary
        self.version = 3.0

        self.mimick = Mimick(characters_vocabulary=characters_vocabulary,
                             characters_embedding_dimension=characters_embedding_dimension,
                             word_embeddings_dimension=word_embeddings_dimension,
                             hidden_state_dimension=char_hidden_state_dimension,
                             lstm_dropout=0.5)
        if mimick_model_path != '':
            self.load_mimick(mimick_model_path, use_gpu)
        if chars_embeddings != None:
            self.load_chars_embeddings(chars_embeddings)
        if freeze_mimick:
            self.freeze_mimick()
            logging.info('Freezing Mimick')

        self.contexts = MirrorLSTM(num_embeddings=len(self.words_vocabulary),
                                   embedding_dim=word_embeddings_dimension,
                                   hidden_state_dim=word_embeddings_dimension,
                                   freeze_embeddings=freeze_word_embeddings,
                                   dropout=0.3)

        if words_embeddings != None:
            self.load_words_embeddings(words_embeddings)

        self.fc_context_left = nn.Linear(in_features=2 * word_embeddings_dimension, out_features=word_embeddings_dimension)
        kaiming_uniform(self.fc_context_left.weight)

        self.fc_context_right = nn.Linear(in_features=2 * word_embeddings_dimension, out_features=word_embeddings_dimension)
        kaiming_uniform(self.fc_context_right.weight)

        self.left_ponderation = nn.Linear(in_features=word_embeddings_dimension, out_features=1)
        constant(self.left_ponderation.weight, 0.25)

        self.right_ponderation = nn.Linear(in_features=word_embeddings_dimension, out_features=1)
        constant(self.right_ponderation.weight, 0.25)

        self.middle_ponderation = nn.Linear(in_features=word_embeddings_dimension, out_features=1)
        constant(self.middle_ponderation.weight, 0.5)

        self.attention_layer = nn.Linear(word_embeddings_dimension * 3, 3)


    def load_mimick(self, model_path, use_gpu):
        if use_gpu:
            map_location = lambda storage, loc: storage.cuda(0)
        else:
            map_location = lambda storage, loc: storage
        state_dict = torch.load(model_path, map_location)
        # Make sure the dimensions fit
        state_dict['mimick_lstm.embeddings.weight'] = self.mimick.mimick_lstm.embeddings.weight
        self.mimick.load_state_dict(state_dict)

    def load_chars_embeddings(self, chars_embeddings):
        for word, embedding in chars_embeddings.items():
            if word in self.characters_vocabulary:
                idx = self.characters_vocabulary[word]
                self.mimick.mimick_lstm.set_item_embedding(idx, embedding)

    def freeze_mimick(self):
        for param in self.mimick.parameters():
            param.requires_grad = False
        self.mimick.fc_output.weight.requires_grad = True
        self.mimick.fc_output.bias.requires_grad = True

    def load_words_embeddings(self, words_embeddings):
        for word, embedding in words_embeddings.items():
            if word in self.words_vocabulary:
                idx = self.words_vocabulary[word]
                self.contexts.set_item_embedding(idx, embedding)

    def log_stats(self, left_context, word, right_context, attention):
        if self.stats:
            self.stats.update(left_context, word, right_context, attention)

    def get_stats(self):
        return self.stats

    def vectorize_words(self, words):
        return [[self.characters_vocabulary[c] for c in w] for w in words]

    def forward(self, x):
        left_context, word, right_context = x

        word_hidden_rep = self.mimick.mimick_lstm(word)
        word_rep = F.tanh(self.mimick.fc_word(word_hidden_rep))

        left_context_hidden_rep, right_context_hidden_rep = self.contexts(left_context, right_context)
        left_context_rep = F.tanh(self.fc_context_left(left_context_hidden_rep))
        right_context_rep = F.tanh(self.fc_context_right(right_context_hidden_rep))

        attn_input = torch.cat([word_rep, left_context_rep, right_context_rep], dim=1)
        attn_logits = self.attention_layer(attn_input.view(-1, self.word_embeddings_dimension * 3))
        attn_pond = F.softmax(attn_logits)

        self.log_stats(left_context, word, right_context, attn_pond)

        # output = self.middle_ponderation.weight * word_rep + self.left_ponderation.weight * left_context_rep + self.right_ponderation.weight * right_context_rep
        output = word_rep * attn_pond[:, 0].view(-1, 1) + left_context_rep * attn_pond[:, 1].view(-1, 1) + right_context_rep * attn_pond[:, 2].view(-1, 1)
        # output = word_rep + left_context_rep + right_context_rep
        # output = self.mimick.fc_output(output)

        return output


class TheFinalComickBoS(Module):
    def __init__(self,
                 embedding_layer,
                 oov_word_model,
                 char_hidden_state_dimension=128,
                 word_hidden_state_dimension=128,
                 chars_embeddings=None,
                 lstm_dropout=.3,
                 freeze_word_embeddings=True,
                 stats=None,
                 attention=False
                 ):
        super().__init__()
        self.oov_word_model = oov_word_model
        self.stats = stats
        self.embedding_layer = embedding_layer
        self.words_vocabulary = embedding_layer.word_to_idx
        self.word_embeddings_dimension = embedding_layer.embedding_dim
        self.version = 3.0
        self.attention = attention

        # self.contexts = MirrorLSTM(
        #     embedding_layer,
        #     hidden_state_dim=word_hidden_state_dimension,
        #     freeze_embeddings=freeze_word_embeddings,
        #     dropout=0.3)

        self.context_lstm = nn.LSTM(
            input_size=embedding_layer.embedding_dim,
            hidden_size=word_hidden_state_dimension,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.word_attention_layer = nn.Linear(word_hidden_state_dimension * 2, 1)
        self.char_attention_layer = nn.Linear(word_hidden_state_dimension * 2, 1)

        self.fc1 = nn.Linear(word_hidden_state_dimension * 4, self.word_embeddings_dimension)

        self.representations_mapping_to_ouput = nn.Linear(self.word_embeddings_dimension, self.word_embeddings_dimension)

    def log_stats(self, left_context, word, right_context, attention):
        if self.stats:
            self.stats.update(left_context, word, right_context, attention)

    def get_stats(self):
        return self.stats

    def vectorize_words(self, words):
        return self.oov_word_model.vectorize_words(words)

    def get_context_rep(self, context):
        lengths = context.data.ne(0).long().sum(dim=1)
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        _, rev_perm_idx = perm_idx.sort(0)
        sorted_contexts = context[perm_idx]

        # Embed
        embeddings = self.embedding_layer(sorted_contexts)

        # Initialize hidden to zero
        packed_input = pack_padded_sequence(embeddings, list(seq_lengths), batch_first=True)
        packed_output, (hidden_states, cell_states) = self.context_lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        last_output = torch.cat([hidden_states[0], hidden_states[1]], dim=1)
        output = output[rev_perm_idx]
        last_output = last_output[rev_perm_idx]
        return (output, last_output)

    def forward(self, x):
        context, word = x

        context_hiddens, context_last = self.get_context_rep(context)

        c_lengths = context.data.ne(0).long().sum(dim=1)
        w_lengths = word.data.ne(0).long().sum(dim=1)

        word_hiddens, word_last = self.oov_word_model(word)

        # attn_input = torch.cat([left_context_hidden_rep, word_rep, right_context_hidden_rep], dim=1)

        if self.attention:
            output = list()
            real_attentions = list()
            for i, word_rep in enumerate(word_last): # Loop over the last hidden states of each words
                # First, compute attention over context condition with word representation
                context_hidden = context_hiddens[i]
                # expanded_word_rep = word_rep.expand_as(context_hidden)
                # attention_input = torch.cat([context_hidden, expanded_word_rep], dim=1)
                words_attn_logits = self.word_attention_layer(context_hidden)
                words_attn_pond = F.softmax(words_attn_logits, dim=0)
                words_attended_output = context_hidden.transpose(0, 1).matmul(words_attn_pond).view(1, -1)

                word_hidden = word_hiddens[i]
                context_rep = context_last[i]
                # expanded_context_rep = context_rep.expand_as(word_hidden)
                # attention_input = torch.cat([word_hidden, expanded_context_rep], dim=1)
                chars_attn_logits = self.char_attention_layer(word_hidden)
                chars_attn_pond = F.softmax(chars_attn_logits, dim=0)
                chars_attended_output = word_hidden.transpose(0, 1).matmul(chars_attn_pond).view(1, -1)
                output.append(torch.cat([words_attended_output, chars_attended_output], dim=1))
                real_attentions.append((words_attn_pond, chars_attn_pond))

        # if self.attention:
        #     output = list()
        #     real_attentions = list()
        #     for i, example in enumerate(attn_input):
        #         left_input = example[:l_lengths[i]]
        #         word_input = example[l_lengths.max():l_lengths.max()+w_lengths[i]]
        #         right_input = example[l_lengths.max() + w_lengths.max():l_lengths.max() + w_lengths.max()+r_lengths[i]]

        #         # Words attention
        #         all_words_input = torch.cat([left_input, right_input], dim=0)
        #         words_attn_logits = self.word_attention_layer(all_words_input)
        #         words_attn_pond = F.softmax(words_attn_logits, dim=0)
        #         words_attended_output = all_words_input.transpose(0, 1).matmul(words_attn_pond).view(1, -1)

        #         # Chars attention
        #         all_chars_input = word_input
        #         chars_attn_logits = self.char_attention_layer(all_chars_input)
        #         chars_attn_pond = F.softmax(chars_attn_logits, dim=0)
        #         chars_attended_output = all_chars_input.transpose(0, 1).matmul(chars_attn_pond).view(1, -1)

        #         output.append(torch.cat([words_attended_output, chars_attended_output], dim=1))

        #         left_attention = words_attn_pond[:l_lengths[i]].view(-1)
        #         right_attention = words_attn_pond[l_lengths[i]:l_lengths[i]+r_lengths[i]].view(-1)
        #         word_attention = chars_attn_pond.view(-1)
        #         real_attentions.append((left_attention, word_attention, right_attention))
            # self.log_stats(left_context, word, right_context, attn_pond)
            # output = word_rep * attn_pond[:, 0].view(-1, 1) + left_context_rep * attn_pond[:, 1].view(-1, 1) + right_context_rep * attn_pond[:, 2].view(-1, 1)
            output = F.tanh(self.fc1(torch.cat(output)))
            output = self.representations_mapping_to_ouput(output)
            return output, real_attentions
        else:
            output = self.representations_mapping_to_ouput(attn_input)
            return output, []


