from typing import Dict

import torch
from torch import nn, autograd
from torch.nn import functional as F
from torch.nn.init import kaiming_uniform, constant
from torch.nn.utils.rnn import pack_padded_sequence


class ContextualMimickUniqueContext(nn.Module):
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
            padding_idx=0,
            max_norm=5
        )
        kaiming_uniform(self.characters_embeddings.weight)

        self.words_embeddings = nn.Embedding(
            num_embeddings=len(self.words_vocabulary),
            embedding_dim=self.word_embeddings_dimension,
            padding_idx=0
        )

        self.context_lstm = nn.LSTM(
            input_size=self.word_embeddings_dimension,
            hidden_size=self.words_hidden_state_dimension,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        # constant(self.context_lstm.bias_ih_l[0], 1)
        # constant(self.context_lstm.bias_hh_l[0], 1)

        self.context_fc = nn.Linear(
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
        # constant(self.lstm.bias_ih_l[0], 1)
        # constant(self.lstm.bias_hh_l[0], 1)

        self.output = nn.Linear(
            self.characters_hidden_state_dimension * 2,
            self.word_embeddings_dimension
        )
        kaiming_uniform(self.output.weight)

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
        contexts, words = x

        ### CONTEXT THING HERE
        lengths = contexts.data.ne(0).sum(dim=1).long()
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        _, rev_perm_idx = perm_idx.sort(0)

        # Embed
        embeds = self.words_embeddings(contexts)

        # LSTM thing
        # Initialize hidden to zero
        packed_input = pack_padded_sequence(embeds, list(seq_lengths), batch_first=True)
        packed_output, (ht, ct) = self.context_lstm(packed_input)
        output = torch.cat([ht[0], ht[1]], dim=1)
        output = self.dropout(output)
        output_context = output[rev_perm_idx]
        output_context = self.context_fc(output_context)


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
        output_middle = output[rev_perm_idx]

        final_output = output_middle + output_context
        final_output = F.tanh(final_output)

        # Map to word embedding dim
        x = self.output(final_output)
        return x


def get_contextual_mimick_unique_context(char_to_idx, word_to_idx, word_embedding_dim=50):
    net = ContextualMimickUniqueContext(
        characters_vocabulary=char_to_idx,
        words_vocabulary=word_to_idx,
        characters_embedding_dimension=20,
        characters_hidden_state_dimension=50,
        words_hidden_state_dimension=50,
        word_embeddings_dimension=word_embedding_dim,
        fully_connected_layer_hidden_dimension=50
    )
    return net
