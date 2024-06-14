# HF Tabular Tokenizer

A Huggingface compatible tokenizer for tokenizing tabular data. Example files show usage.

Essentially, provide a pandas DataFrame and schema and then tokenize the data for ingestion into your `transformers` model.

It is seamless to use with FlashAttention, as well as Ray and Optuna too. 

# Examples
The first example shows how to train the tokenizer and create a dataset from a DataFrame
The second example shows how to train a HuggingFace model using the TabularTokenizer and the dataset created from the first example.
```
python example1_create_vocab.py --nrows 5000 --ncols 40 --seqlen 1024 --stride 5
python example2_train_model.py --vocab-file example_vocab.pkl --seqlen 1024 --train train.json --valid valid.json --test test.json
```

## License
Apache 2.0
