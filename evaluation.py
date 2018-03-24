import numpy
import logging

from utils import load_embeddings
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_embeddings(embeddings):

    # Load glove embeddings for comparisons
    logging.info('Loading GloVe embeddings for comparison...')
    glove_embeddings = load_embeddings('./embeddings/glove_valid_embeddings.txt')

    # Compare distance of mimick with the glove embeddings
    logging.info('Comparing the distance of Pinter\'s Mimick implementation...')
    mimick_distances = list()
    mimick_cosine_distances = list()
    mimick_embeddings = load_embeddings('./embeddings/previous_mimick_validation_embeddings.txt')
    for word, embedding in mimick_embeddings.items():
        if word in glove_embeddings:
            target_embedding = glove_embeddings[word]
            mimick_distances.append(numpy.linalg.norm(embedding - target_embedding))
            mimick_cosine_distances.append(cosine_similarity(embedding.reshape(1, -1), target_embedding.reshape(1, -1)))
    logging.info("Mimick distance: {}, var: {}, cosine: {}, ({})".format(numpy.mean(mimick_distances), numpy.std(mimick_distances), numpy.mean(mimick_cosine_distances), len(mimick_distances)))

    # Compare distance of our implementation with the glove embeddings
    logging.info('Comparing the distance of our embeddings with GloVe embeddings...')
    our_distances = list()
    our_cosine_distances = list()
    for word, embedding in embeddings.items():
        if word in glove_embeddings:
            target_embedding = glove_embeddings[word]
            our_distances.append(numpy.linalg.norm(embedding - target_embedding))
            our_cosine_distances.append(cosine_similarity(embedding.reshape(1, -1), target_embedding.reshape(1, -1)))
    logging.info("Our distance: {}, var: {}, cosine: {}, ({})".format(numpy.mean(our_distances), numpy.std(our_distances), numpy.mean(our_cosine_distances), len(our_distances)))

    # Compare distance of our embeds from mimick's
    logging.info('Comparing the distance of our embeddings with Yuval\'s Mimick...')
    our_distances_with_mimick = list()
    our_distances = list()
    for word, embedding in embeddings.items():
        if word in mimick_embeddings:
            target_embedding = mimick_embeddings[word]
            our_distances_with_mimick.append(numpy.linalg.norm(embedding - target_embedding))
    logging.info("Distance with mimick: {}, {} ({})".format(numpy.mean(our_distances_with_mimick), numpy.std(our_distances_with_mimick), len(our_distances_with_mimick)))
