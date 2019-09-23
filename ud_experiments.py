import math
import logging
import os

import sacred
from sacred import Experiment
from sacred.observers import MongoObserver


from pymongo import MongoClient

from downstream_task.sequence_tagging import collate_examples_multiple_tags
from downstream_task.models import CharRNN, SimpleLSTMTagger
from comick import TheFinalComick, TheFinalComickBoS, BoS, Mimick, MimickV2

from utils import *

exp_name = os.getenv('EXP_NAME', default='ud_tagging_model_multiple_tags')
db_url = os.getenv('DB_URL', default='localhost')
db_name = os.getenv('DB_NAME', default='chalet')

# Logging thangs
base_config_file = './configs/base.json'

# Experiment
experiment = Experiment(exp_name)
experiment.add_config(base_config_file)
experiment.observers.append(
    MongoObserver.create(
        url=db_url,
        db_name=db_name
    )
)

client = MongoClient(db_url)
database = client[db_name]
collection = database['logs']

languages = {
    'kk': ('kk', 'UD_Kazakh', 'kk-ud'),
    'ta': ('ta', 'UD_Tamil', 'ta-ud'),
    'lv': ('lv', 'UD_Latvian', 'lv-ud'),
    'vi': ('vi', 'UD_Vietnamese', 'vi-ud'),
    'hu': ('hu', 'UD_Hungarian', 'hu-ud'),
    'tr': ('tr', 'UD_Turkish', 'tr-ud'),
    'el': ('el', 'UD_Greek', 'el-ud'),
    'bg': ('bg', 'UD_Bulgarian', 'bg-ud'),
    'sv': ('sv', 'UD_Swedish', 'sv-ud'),
    'eu': ('eu', 'UD_Basque', 'eu-ud'),
    'ru': ('ru', 'UD_Russian', 'ru-ud'),
    'da': ('da', 'UD_Danish', 'da-ud'),
    'id': ('id', 'UD_Indonesian', 'id-ud'),
    'zh': ('zh', 'UD_Chinese', 'zh-ud'),
    'fa': ('fa', 'UD_Persian', 'fa-ud'),
    'he': ('he', 'UD_Hebrew', 'he-ud'),
    'ro': ('ro', 'UD_Romanian', 'ro-ud'),
    'en': ('en', 'UD_English', 'en-ud'),
    'ar': ('ar', 'UD_Arabic', 'ar-ud'),
    'hi': ('hi', 'UD_Hindi', 'hi-ud'),
    'it': ('it', 'UD_Italian', 'it-ud'),
    'es': ('es', 'UD_Spanish', 'es-ud'),
    'cs': ('cs', 'UD_Czech', 'cs-ud'),
    'fr': ('fr', 'UD_French', 'fr-ud'),
}


@experiment.command
def train(_run, _config, seed, batch_size, lstm_hidden_layer, language, epochs):
    if _config['tag_to_predict'] == 'MORPH' and language in {'vi', 'id'}:
        return None
    print(_config)
    np.random.seed(seed)
    torch.manual_seed(seed)

    language = LanguageDataset(*languages[language])

    train_sentences = [(instance.sentence, instance.chars, instance.substrings) for instance in language.training_instances]
    train_tags = [instance.tags for instance in language.training_instances]

    dev_sentences = [(instance.sentence, instance.chars, instance.substrings) for instance in language.dev_instances]
    dev_tags = [instance.tags for instance in language.dev_instances]

    test_sentences = [(instance.sentence, instance.chars, instance.substrings) for instance in language.test_instances]
    test_tags = [instance.tags for instance in language.test_instances]

    train_dataset = list(zip(train_sentences, train_tags))
    dev_dataset = list(zip(dev_sentences, dev_tags))
    test_dataset = list(zip(test_sentences, test_tags))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_examples_multiple_tags
    )

    valid_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        collate_fn=collate_examples_multiple_tags
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_examples_multiple_tags
    )

    oovs = language.word_to_index.keys() - language.embeddings.keys()
    t_oovs = language.test_vocab.keys() & oovs
    print("Ratio OOVs: {} ({}/{})".format(len(oovs)/len(language.word_to_index), len(oovs), len(language.word_to_index)))

    # Compute the number of occurences of OOVs in the test set as well as the ratio
    oovs_ids = {language.word_to_index[o] for o in oovs}
    print(len([word_id for s, _, _ in train_sentences for word_id in s]))
    print(len([word_id for s, _, _ in dev_sentences for word_id in s]))
    print(len([word_id for s, _, _ in test_sentences for word_id in s]))
    num_words = len([word_id for s, _, _ in test_sentences for word_id in s])
    num_oovs = len([word_id for s, _, _ in test_sentences for word_id in s if word_id in oovs_ids])
    print("Ratio of occurrences of OOVs: {} ({}/{})".format(num_oovs/num_words, num_oovs, num_words))

    # OOV appeareance rate in the contexts
    oov_app_test, lengths = oov_appearance_rate_in_contexts([i for i, _, _ in test_sentences], oovs_ids)
    print("Average length: {}".format(np.mean(lengths)))
    print("Average OOV appearance in context: {}".format(np.mean(oov_app_test)))
    print("Average OOV appearance 50th percentile in context: {}".format(np.percentile(oov_app_test, 50)))
    print("Average OOV appearance 80th percentile in context: {}".format(np.percentile(oov_app_test, 80)))

    embedding_layer = MyEmbeddings(language.word_to_index, language.embedding_dim)
    embedding_layer.load_words_embeddings(language.embeddings)

    if _config["embeddings_mode"] == "random":
        # Leave OOV random embeddings
        comick = None

    elif _config["embeddings_mode"] == "mimick":
        # Fill OOV embeddings with Mimick's
        embed_path = "./data/mimick-embs/{}-lstm-est-embs.txt".format(language.polyglot_abbreviation)
        mimick_embeds = load_embeddings(embed_path)
        embedding_layer.load_words_embeddings(mimick_embeds)
        comick = None

    elif _config["embeddings_mode"] == "comick":
        embedding_layer_comick = MyEmbeddings(language.word_to_index, language.embedding_dim)
        embedding_layer_comick.load_words_embeddings(language.embeddings)

        if _config["oov_word_model"] == "mimick":
            oov_word_model = Mimick(
                characters_vocabulary=language.char_to_index,
                word_embeddings_dimension=language.embedding_dim,
                characters_embedding_dimension=64
            )
        elif _config["oov_word_model"] == "bos":
            oov_word_model = BoS(
                language.bos_to_index,
                embedding_dim=language.embedding_dim,
            )

        comick = TheFinalComickBoS(
            embedding_layer_comick,
            oov_word_model,
            word_hidden_state_dimension=128,
            freeze_word_embeddings=False,
            attention=_config['attention']
        )


    char_model = CharRNN(
        language.char_to_index,
        _config["char_embedding_size"],
        lstm_hidden_layer,
    )

    model = SimpleLSTMTagger(
        char_model,
        embedding_layer,
        lstm_hidden_layer,
        {label: len(tags) for label, tags in language.tags_to_index.items()},
        oovs,
        comick,
        n=41,
        tag_to_predict=_config['tag_to_predict']
    )

    for name, parameter in model.named_parameters():
        if 'embedding_layer' not in name:
            if 'bias' in name:
                constant(parameter, 0)
            elif 'weight' in name:
                kaiming_normal(parameter)

    model_name = "{}".format(language.polyglot_abbreviation)
    expt_name = './expt_{}_{}_{}'.format(model_name, _config["embeddings_mode"], db_name)
    expt_dir = get_experiment_directory(expt_name)

    device_id = _config["device"]
    device = None
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id) # Fix bug where memory is allocated on GPU0 when ask to take GPU1.
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:%d' % device_id)
        logging.info("Training on GPU %d" % device_id)
    else:
        logging.info("Training on CPU")


    optimizer = torch.optim.Adam(model.parameters(), lr=_config["learning_rate"])
    monitor_metric = 'val_loss' if _config['tag_to_predict'] == 'MORPH' else 'val_acc'
    monitor_mode = 'min' if _config['tag_to_predict'] == 'MORPH' else 'max'
    expt = PytouneExperiment(
        expt_dir,
        model,
        device=device,
        optimizer=optimizer,
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode
    )

    callbacks = [
        ClipNorm(model.parameters(), _config["gradient_clipping"]),
        ReduceLROnPlateau(monitor=monitor_metric, mode=monitor_mode, patience=_config["reduce_lr_on_plateau"]["patience"], factor=_config["reduce_lr_on_plateau"]["factor"], threshold_mode='abs', threshold=1e-3, verbose=True),
        EarlyStopping(patience=_config["early_stopping"]["patience"], min_delta=1e-4, monitor=monitor_metric, mode=monitor_mode),
        MetricsCallback(_run)
    ]

    try:
        expt.train(train_loader, valid_loader, callbacks=callbacks, seed=42, epochs=epochs)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    print("Testing on test set...")
    metrics = expt.test(test_loader)

    vectors = torch_to_numpy(model.embedding_layer.weight)

    all_preds = list()
    all_trues = list()
    for x, ys in test_loader:
        preds, attention = expt.model.predict_on_batch(x)
        for tag, y in preds.items():
            if tag is not 'POS':
                all_pred = np.argmax(y, axis=2).reshape(-1)
                all_true = ys[tag].view(-1)
                for y_pred, y_true in zip(all_pred, all_true):
                    if y_true != 0 and y_true != 1:
                        all_preds.append(y_pred)
                        all_trues.append(y_true.item())
    if len(all_preds) > 0:
        f1 = f1_score(all_preds, all_trues, average='micro')
        metrics['f1'] = f1
        print("Precision score: {}".format(precision_score(all_preds, all_trues, average='micro')))
        print("Recall score: {}".format(recall_score(all_preds, all_trues, average='micro')))
        print("F1 score: {}".format(f1))
        print(classification_report(all_trues, all_preds, digits=4))

    pred_morph_per_oovs = defaultdict(list)
    true_morph_per_oovs = defaultdict(list)
    stats_pos_per_oovs = defaultdict(list)
    attention_analysis = defaultdict(list)
    for x, ys in test_loader:
        sentence = x[0]
        preds, attentions = expt.model.predict_on_batch(x)
        for tag, y in preds.items():
            if tag is 'POS':
                all_pred = np.argmax(y, axis=2).reshape(-1)
                all_true = ys[tag].view(-1)
                all_sentence = sentence.view(-1)
                attention_pred_tag = np.argmax(y, axis=2)
                attention_true_tag = ys[tag]
                for token, y_pred, y_true in zip(all_sentence, all_pred, all_true):
                    token_value = language.idx_to_word[token.item()]
                    if token_value in oovs:
                        if y_pred == y_true.item():
                            stats_pos_per_oovs[token_value].append(1)
                        else:
                            stats_pos_per_oovs[token_value].append(0)
            else:
                all_pred = np.argmax(y, axis=2).reshape(-1)
                all_true = ys[tag].view(-1)
                all_sentence = sentence.view(-1)
                for token, y_pred, y_true in zip(all_sentence, all_pred, all_true):
                    if y_true != 0 and y_true != 1:
                        token_value = language.idx_to_word[token.item()]
                        if token_value in oovs:
                            true_morph_per_oovs[token_value].append(y_true.item())
                            pred_morph_per_oovs[token_value].append(y_pred)
        for sent_idx, _, word_idx, embedding, attention in attentions:
            s = sentence[sent_idx]
            target_word = language.idx_to_word[s[word_idx].item()]
            sims = cos_matrix_multiplication(vectors, embedding)
            most_similar_word = language.idx_to_word[np.argmax(sims)]
            most_similar_word_sim = np.max(sims)
            s_to_words = " ".join([language.idx_to_word[w.item()] for w in s if w.item() > 0])
            result = attention_pred_tag[sent_idx][word_idx] == attention_true_tag[sent_idx][word_idx]

            formatted_attention = []
            c, w = attention
            c1 = c.reshape(-1).tolist()
            w1 = w.reshape(-1).tolist()
            formatted_attention += [c1, w1]

            attention_analysis[target_word.replace('.', '<DOT>')].append((
                target_word, most_similar_word, float(most_similar_word_sim), int(word_idx), formatted_attention, s_to_words, int(result.item())
            ))

    metrics['attention'] = attention_analysis
    metrics['pos_per_oov'] = dict()

    # for target_word, occurrences in attention_analysis.items():
    #     print("="*80)
    #     print("TARGET WORD: {}".format(target_word))
    #     for target_word, sim_word, sim_word_sim, word_idx, attention, sentence, result in occurrences:
    #         print("{}\t({})\t{}\t{}\n{}\n{}\t({})".format(target_word, word_idx, "\t".join([str(a) for a in attention]), result, sentence, sim_word, sim_word_sim))
    #         print()
    #     print("="*80)

    all_occurrences = list()
    for oov, occurrences in stats_pos_per_oovs.items():
        oov = oov.replace('.', '<DOT>') # For mongodb
        all_occurrences += occurrences
        metrics['pos_per_oov'][oov] = dict()
        metrics['pos_per_oov'][oov]['percent'] = sum(occurrences)/float(len(occurrences))
        metrics['pos_per_oov'][oov]['num'] = len(occurrences)
    metrics['pos_per_oov']['total'] = dict()
    metrics['pos_per_oov']['total']['percent'] = sum(all_occurrences)/float(len(all_occurrences))
    metrics['pos_per_oov']['total']['num'] = len(all_occurrences)
    print("OOV acc rate: {}".format(metrics['pos_per_oov']['total']['percent'], metrics['pos_per_oov']['total']['num']))

    all_occurrences = list()
    all_true_occurrences = list()
    for oov, preds in pred_morph_per_oovs.items():
        all_occurrences += preds
        all_true_occurrences += true_morph_per_oovs[oov]
    if len(all_occurrences) > 0:
        print("OOV Precision rate: {}".format(precision_score(all_occurrences, all_true_occurrences, average='micro')))
        print("OOV Recall rate: {}".format(recall_score(all_occurrences, all_true_occurrences, average='micro')))
        print("OOV F1 rate: {}".format(f1_score(all_occurrences, all_true_occurrences, average='micro')))

    all_stats = {
        'model': model_name,
        'metrics': metrics
    }
    collection.insert_one(all_stats)


@experiment.automain
def main(_config):
    for language in languages:
        run = experiment.run('train', config_updates={'language': language})


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()

