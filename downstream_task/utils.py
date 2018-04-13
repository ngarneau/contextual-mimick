def make_idx(vocab: set):
    idx = dict()
    idx['PAD'] = 0
    for v in sorted(vocab):
        idx[v] = len(idx)
    return idx

def make_vocab_and_idx(sequences):
    words_vocab = {word for sentence in sequences for word in sentence}
    words_to_idx = make_idx(words_vocab)
    return words_vocab, words_to_idx
