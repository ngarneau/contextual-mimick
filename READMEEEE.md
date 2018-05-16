First thing to do is to download the datasets by [...how do we do that?]
Place the dataset in your 'data' directory. It should look like this:

```
./data
    /conll
        /train.txt
        /valid.txt
        /test.txt
    /glove_embeddings
        /glove.6B.50d.txt
        /glove.6B.100d.txt
        /glove.6B.200d.txt
        /glove.6B.300d.txt
    /scienceie
        /train_spacy.txt
        /valid_spacy.txt
        /test_spacy.txt
    /sentiment
        /train.pickle
        /dev.pickle
        /test.pickle
    /__init__.py
    /data_preprocessing.py
    /dataset_manager.py
```

Run the main in the file 'data_preprocessing.py' to generate the examples and the statistics about the three datasets.