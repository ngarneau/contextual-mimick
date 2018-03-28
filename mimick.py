from typing import Dict

import torch
from torch import nn, autograd
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.init import kaiming_uniform


class Mimick(nn.Module):
    def __init__(self, characters_vocabulary: Dict[str, int], characters_embedding_dimension: int,
                 characters_hidden_state_dimension: int, word_embeddings_dimension: int,
                 fully_connected_layer_hidden_dimension: int):
        super().__init__()
        self.fully_connected_layer_hidden_dimension = fully_connected_layer_hidden_dimension
        self.characters_vocabulary = characters_vocabulary
        self.characters_embedding_dimension = characters_embedding_dimension
        self.characters_hidden_state_dimension = characters_hidden_state_dimension
        self.word_embeddings_dimension = word_embeddings_dimension
        self.num_layers = 1
        self.bidirectional = True

        self.characters_embeddings = nn.Embedding(
            num_embeddings=len(self.characters_vocabulary),
            embedding_dim=self.characters_embedding_dimension,
            padding_idx=0,
            max_norm=5
        )
        kaiming_uniform(self.characters_embeddings.weight)

        self.lstm = nn.LSTM(
            input_size=self.characters_embedding_dimension,
            hidden_size=self.characters_hidden_state_dimension,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.fully_connected_1 = nn.Linear(
            self.characters_hidden_state_dimension * 2,  # Two hidden states concatenated
            self.fully_connected_layer_hidden_dimension
        )
        kaiming_uniform(self.fully_connected_1.weight)

        self.fully_connected_2 = nn.Linear(
            self.fully_connected_layer_hidden_dimension,
            self.fully_connected_layer_hidden_dimension
        )
        kaiming_uniform(self.fully_connected_2.weight)

        self.output = nn.Linear(
            self.fully_connected_layer_hidden_dimension,
            self.word_embeddings_dimension
        )
        kaiming_uniform(self.output.weight)

    def init_hidden(self, batch):
        if self.bidirectional:
            first_dim = 2 * self.num_layers
        else:
            first_dim = self.num_layers
        return (autograd.Variable(torch.zeros(first_dim, batch, self.characters_hidden_state_dimension)),
                autograd.Variable(torch.zeros(first_dim, batch, self.characters_hidden_state_dimension)))

    def forward(self, x):
        # Pre processing
        lengths = x.data.ne(0).sum(dim=1).long()
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        _, rev_perm_idx = perm_idx.sort(0)

        # Embed
        embeds = self.characters_embeddings(x)

        # LSTM thing
        packed_input = pack_padded_sequence(embeds, list(seq_lengths), batch_first=True)
        packed_output, (ht, ct) = self.lstm(packed_input)
        output = torch.cat([ht[0], ht[1]], dim=1)
        output = output[rev_perm_idx]

        # Map to word embedding dim
        x = self.fully_connected_1(output)
        x = F.tanh(x)
        x = self.output(x)
        return x
