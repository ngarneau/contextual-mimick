from utils import load_embeddings, parse_conll_file

def load_data(d, verbose=True):
    path_embeddings_train = './embeddings_settings/setting1/1_glove_embeddings/glove.6B.{}d.txt'.format(d)
    path_embeddings_test = './embeddings_settings/validation_embeddings/glove.6B.{}d.txt'.format(d)
    try:
        train_embeddings = load_embeddings(path_embeddings_train)
    except:
        if d == 50:
            path_embeddings_train = './embeddings/train_embeddings.txt'
            train_embeddings = load_embeddings(path_embeddings_train)
        else:
            raise
    if verbose:
        print('Loading ' + str(d) + 'd training embeddings from: "' + path_embeddings_train + '" and test embeddings from: "' + path_embeddings_test + '"')

    test_embeddings = load_embeddings(path_embeddings_test)

    train_sentences = parse_conll_file('./conll/train.txt')
    valid_sentences = parse_conll_file('./conll/valid.txt')
    test_sentences = parse_conll_file('./conll/test.txt')

    return (train_embeddings, test_embeddings), (train_sentences, valid_sentences, test_sentences)

if __name__ == '__main__':
    embeddings, sentences = load_data(50)
    train_embeddings, test_embeddings = embeddings
    train_sentences, valid_sentences, test_sentences = sentences
    # valid_sentences += test_sentences

    train_labels = set(token for sentence in train_sentences for token in sentence)
    print('number of distinct labels seen in training:', len(train_labels))

    test_labels = set(token for sentence in valid_sentences for token in sentence)
    print('number of distinct tokens in the test (valid of conll):', len(test_labels))

    test_inter_train = test_labels & train_labels
    print('number of labels both in train and test:', len(test_inter_train))

    test_wo_train = test_labels - test_inter_train
    print('number of labels in test NOT in train:', len(test_wo_train))

    tr_emb = set(train_embeddings.keys()) # This is the entirety of GloVe
    ts_emb = set(test_embeddings.keys())
    print('number of labels in the validation embeddings "embeddings_settings/validation_embeddings/glove.6B.50d" file:', len(ts_emb))

    print('number of labels in test for which we have an embedding:', len(test_labels & tr_emb))

    test_wo_train_inter_tr_emb = test_wo_train & tr_emb
    print('number of labels in test but not in train, for which we have an embedding:', len(test_wo_train_inter_tr_emb))

    test_wo_train_inter_ts_emb = test_wo_train & ts_emb
    print('number of labels in test but not in train, and also in the validation embeddings file:', len(test_wo_train_inter_ts_emb))
