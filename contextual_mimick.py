from typing import Dict

import torch
from torch import nn, autograd
from torch.nn import functional as F
from torch.nn.init import kaiming_uniform
from torch.nn.utils.rnn import pack_padded_sequence


class ContextualMimick(nn.Module):
    def __init__(self, characters_vocabulary: Dict[str, int], characters_embedding_dimension: int,
                 characters_hidden_state_dimension: int, word_embeddings_dimension: int,
                 words_vocabulary: Dict[str, int], words_hidden_state_dimension: int,
                 fully_connected_layer_hidden_dimension: int):
        super().__init__()
        self.words_hidden_state_dimension = words_hidden_state_dimension
        self.words_vocabulary = words_vocabulary
        self.fully_connected_layer_hidden_dimension = fully_connected_layer_hidden_dimension
        self.characters_vocabulary = characters_vocabulary
        self.characters_embedding_dimension = characters_embedding_dimension
        self.characters_hidden_state_dimension = characters_hidden_state_dimension
        self.word_embeddings_dimension = word_embeddings_dimension
        self.num_layers = 1
        self.bidirectional = True
        self.dropout = nn.Dropout(0.5)

        self.characters_embeddings = nn.Embedding(
            num_embeddings=len(self.characters_vocabulary),
            embedding_dim=self.characters_embedding_dimension,
            padding_idx=0
        )
        kaiming_uniform(self.characters_embeddings.weight)

        self.words_embeddings = nn.Embedding(
            num_embeddings=len(self.words_vocabulary),
            embedding_dim=self.word_embeddings_dimension,
            padding_idx=0
        )

        self.left_to_right_lstm = nn.LSTM(
            input_size=self.word_embeddings_dimension,
            hidden_size=self.words_hidden_state_dimension,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.left_to_right_fc = nn.Linear(
            in_features=self.words_hidden_state_dimension*2,
            out_features=self.characters_hidden_state_dimension*2
        )

        self.lstm = nn.LSTM(
            input_size=self.characters_embedding_dimension,
            hidden_size=self.characters_hidden_state_dimension,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.right_to_left_lstm = nn.LSTM(
            input_size=self.word_embeddings_dimension,
            hidden_size=self.words_hidden_state_dimension,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.right_to_left_fc = nn.Linear(
            in_features=self.words_hidden_state_dimension*2,
            out_features=self.characters_hidden_state_dimension*2
        )

        self.fully_connected_1 = nn.Linear(
            # self.characters_hidden_state_dimension * 2 + self.words_hidden_state_dimension * 2,  # Two hidden states concatenated
            self.characters_hidden_state_dimension * 2,
            # self.characters_hidden_state_dimension,
            self.fully_connected_layer_hidden_dimension
        )
        kaiming_uniform(self.fully_connected_1.weight)

        self.output = nn.Linear(
            self.fully_connected_layer_hidden_dimension,
            self.word_embeddings_dimension
        )
        kaiming_uniform(self.output.weight)

    def init_hidden(self, batch, dim):
        if self.bidirectional:
            first_dim = 2 * self.num_layers
        else:
            first_dim = self.num_layers
        hidden1, hidden2 = torch.zeros(first_dim, batch, dim), torch.zeros(first_dim, batch, dim)
        if torch.cuda.is_available():
            hidden1.cuda()
            hidden2.cuda()
        return (autograd.Variable(hidden1), autograd.Variable(hidden2))

    def load_words_embeddings(self, word_to_embed):
        for word, embed in word_to_embed.items():
            target_word = word
            if target_word not in self.words_vocabulary:
                target_word = word.upper()
                if target_word not in self.words_vocabulary:
                    pass
            else:
                word_idx = self.words_vocabulary[target_word]
                self.words_embeddings.weight[word_idx].data = torch.FloatTensor(embed)

    def forward(self, x):
        # Pre processing
        left_contexts, words, right_contexts = x

        ### LEFT THING HERE
        lengths = left_contexts.data.ne(0).sum(dim=1).long()
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        _, rev_perm_idx = perm_idx.sort(0)

        # Embed
        embeds = self.words_embeddings(left_contexts)

        # LSTM thing
        # Initialize hidden to zero
        packed_input = pack_padded_sequence(embeds, list(seq_lengths), batch_first=True)
        packed_output, (ht, ct) = self.left_to_right_lstm(packed_input)
        output = torch.cat([ht[0], ht[1]], dim=1)
        output = self.dropout(output)
        output_left = output[rev_perm_idx]
        output_left = self.left_to_right_fc(output_left)
        # output_left = F.tanh(output_left)
        # output_left = self.dropout(output_left)
        # output_left = F.relu(output_left)


        ### WORDS THING HERE
        lengths = words.data.ne(0).sum(dim=1).long()
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        _, rev_perm_idx = perm_idx.sort(0)

        # Embed
        embeds = self.characters_embeddings(words)

        # LSTM thing
        packed_input = pack_padded_sequence(embeds, list(seq_lengths), batch_first=True)
        packed_output, (ht, ct) = self.lstm(packed_input)
        output = torch.cat([ht[0], ht[1]], dim=1)
        # output = ht[0] + ht[1]
        output_middle = output[rev_perm_idx]
        # output_middle = F.tanh(output_middle)

        ### RIGHT THING HERE
        lengths = right_contexts.data.ne(0).sum(dim=1).long()
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        _, rev_perm_idx = perm_idx.sort(0)

        # Embed
        embeds = self.words_embeddings(right_contexts)

        # LSTM thing
        packed_input = pack_padded_sequence(embeds, list(seq_lengths), batch_first=True)
        packed_output, (ht, ct) = self.right_to_left_lstm(packed_input)
        output = torch.cat([ht[0], ht[1]], dim=1)
        output = self.dropout(output)
        output_right = output[rev_perm_idx]
        output_right = self.right_to_left_fc(output_right)
        # output_right = F.tanh(output_right)
        # output_right = self.dropout(output_right)
        # output_right = F.relu(output_right)

        final_output = output_left + output_middle + output_right
        # final_output = output_left + output_right
        final_output = F.tanh(final_output)
        # final_output = torch.cat([output_left, output_middle, output_right], dim=1)
        # final_output = torch.cat([output_middle], dim=1)
        # final_output = output_middle


        # Map to word embedding dim
        x = self.fully_connected_1(final_output)
        x = F.tanh(x)
        x = self.output(x)
        return x


def get_contextual_mimick(char_to_idx, word_to_idx, word_embedding_dim=50):
    net = ContextualMimick(
        characters_vocabulary=char_to_idx,
        words_vocabulary=word_to_idx,
        characters_embedding_dimension=20,
        characters_hidden_state_dimension=50,
        words_hidden_state_dimension=50,
        word_embeddings_dimension=word_embedding_dim,
        fully_connected_layer_hidden_dimension=50
    )
    return net
