import torch

class AttentionStats:
    def __init__(self, word_to_index, char_to_index):
        self.word_to_index = word_to_index
        self.char_to_index = char_to_index

        self.left_contexts = list()
        self.right_contexts = list()
        self.words = list()
        self.attentions = list()

    def update(self, left_contexts, words, right_contexts, attentions):
        self.left_contexts.append(left_contexts)
        self.right_contexts.append(right_contexts)
        self.words.append(words)
        self.attentions.append(attentions)


    def compute_stats(self):
        print(yoo)
