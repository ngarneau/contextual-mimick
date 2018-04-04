import pickle
from gensim.models.keyedvectors import KeyedVectors
from multiprocessing import Pool
from numpy import Infinity
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils import load_embeddings


def find_most_similars(embeddings, target_embedding, similar_words, n=1, excluded_words=None):
    if excluded_words is None:
        excluded_words = dict()
    similar_embeddings = list()
    for i in range(n):
        highest_sim = float(-Infinity)
        highest_word = None
        highest_embedding = None
        for word, embedding in embeddings.items():
            similarity = cosine_similarity(target_embedding.reshape(1, -1), embedding.reshape(1, -1))
            if similarity > highest_sim and word not in similar_words and word not in excluded_words:
                highest_sim = similarity
                highest_word = word
                highest_embedding = embedding
        similar_words.add(highest_word)
        similar_embeddings.append((highest_word, highest_sim, highest_embedding))
    return similar_embeddings


def compute_cartesian_similarities(all_embeddings):
    all_similarities = dict()  # Contains a word -> similarities which is a dict that contains word -> similarity
    for word1, embedding1 in tqdm(all_embeddings.items()):
        for word2, embedding2 in tqdm(all_embeddings.items()):
            if word1 not in all_similarities:
                all_similarities[word1] = dict()
            if word1 != word2:
                if word2 in all_similarities and word1 in all_similarities[word2]:  # Already computed
                    all_similarities[word1][word2] = all_similarities[word2][word1]
                else:  # We have to compute it and we are going to store it for both
                    if word2 not in all_similarities:
                        all_similarities[word2] = dict()
                    sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))
                    all_similarities[word1][word2] = sim
                    all_similarities[word2][word1] = sim


def main():
    print("Loading all embeddings from GloVe...")
    word_vectors = KeyedVectors.load_word2vec_format('./embeddings_settings/setting1/1_glove_embeddings/glove.6B.100d.txt')
    print("Done.")

    print("Loading all embeddings from training set...")
    train_embeddings = load_embeddings('./embeddings_settings/setting2/1_glove_embeddings/glove.6B.100d.txt')
    print("Done.")

    similar_embeddings = dict()
    for word, embedding in tqdm(train_embeddings.items()):
        similar_embeddings[word] = word_vectors.most_similar(word, topn=5)

    pickle.dump(similar_embeddings, open('similar_words.p', 'wb'))

    # all_embeddings = load_embeddings('./embeddings_settings/setting1/1_glove_embeddings/glove.6B.100d.txt')
    #
    # print("Computing similarities...")
    # all_similarities = compute_cartesian_similarities(all_embeddings)
    # print("Done.")

    # print("Loading all embeddings from training set...")
    # train_embeddings = load_embeddings('./embeddings_settings/setting2/1_glove_embeddings/glove.6B.100d.txt')
    # print("Done.")
    #
    # print("Fetching most similar embeddings for all words in training set...")
    # similar_embeddings = dict()
    # for word, embedding in tqdm(train_embeddings.items()):
    #     similar_embeddings[word] = find_most_similars(all_embeddings, embedding, {word}, 1, train_embeddings.keys())
    #
    # for word, similar in similar_embeddings.keys():
    #     print(word)
    #     print(similar_embeddings[0], [1])
    #     print()


if __name__ == '__main__':

    main()
