############################################################################
##
## Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
##
## NVIDIA Sample Code
##
## Please refer to the NVIDIA end user license agreement (EULA) associated
## with this source code for terms and conditions that govern your use of
## this software. Any use, reproduction, disclosure, or distribution of
## this software and related documentation outside the terms of the EULA
## is strictly prohibited.
##
###########################################################################

from argparse import ArgumentParser
from collections import namedtuple
import json
import math
import pickle
from typing import List, Dict

import pandas as pd
import torch
from datasets import load_dataset

from coder.hf_tabular_tokenizer import HFTabularTokenizer
from utils.utils import tab_structure, column_codes, make_data, make_multiple_docs
from utils.logger import create_logger

logger = create_logger(__name__)


def tokenize_function(text: str):
    """
    Function used by tokenized_datasets to create custom tabular tokenized dataset.
    Args:
        text: string to be tokenized by tabular_tokenizer

    Returns:
        encoded string
    """
    return tabular_tokenizer.encode(text)


def tokenized_datasets(example: dict) -> dict:
    """
    takes a list of token_ids and creates a item for causal LM.
    """
    token_ids = tokenize_function(example['text'][0])

    return {'input_ids': [token_ids],
            'attention_mask': [[1 for _ in range(len(token_ids))]],
            'labels': [token_ids],
            }


def get_col_dtypes_from_make_data(df):
    """
    Function is specific for the make_data function in this example.
    Args:
        df: pd.DataFrame

    Returns: column names of their corresponding dtypes.

    """
    categorical_cols = []
    float_cols = []
    integer_cols = []
    vector_cols = []
    for col in df.columns:
        if 'letter' in col:
            categorical_cols.append(col)
        elif 'float' in col:
            float_cols.append(col)
        elif 'integer' in col:
            integer_cols.append(col)
    return categorical_cols, float_cols, integer_cols, vector_cols


def decoded_str_to_df(single_item: str,
                      columns: List,
                      tab_struct: List[Dict],
                      delimiter: str,
                      endoftext: str,
                      newline: str) -> pd.DataFrame:
    """
    Takes a single tabular string and converts it back to a pandas DataFrame
    Args:
        single_item (str): string version of a table
        columns (list): list of column names for the DataFrame
        tab_struct (list(dict)): the structure used to compute the tabular structure. Needed to extract the column dtype
        delimiter (str): delimiter str in the single_item
        endoftext (str): endoftext str in the single_item
        newline (str): newline str in the single_item

    Returns: pd.DataFrame
    """

    string_rows = [row.split(delimiter) for row in
                   single_item.strip(endoftext).split(newline)]
    d = pd.DataFrame(string_rows, columns=columns)
    for strct in tab_struct:
        typ = strct['code_type'] if strct['code_type'] != 'category' else str
        d[strct['name']] = d[strct['name']].astype(typ)
    return d


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nrows', type=int, default=2000, help='number of synthetic rows to create')
    parser.add_argument('--ncols', type=int, default=10, help='number of synthetic cols to create')
    parser.add_argument('--seqlen', type=int, default=512, help='Sequence length of the tokenized documents and model')
    parser.add_argument('--stride', type=int, default=2, help='how many rows to stride in the dataframe when creating docs')
    args = parser.parse_args()
    assert args.nrows > 0
    assert args.ncols > 0
    assert args.seqlen > 0
    assert args.stride > 0

    Constants = namedtuple('constants', ['END_OF_TEXT', 'NEW_LINE', 'DELIMITER'])
    CONSTS = Constants(END_OF_TEXT='<|endoftext|>', NEW_LINE='\n', DELIMITER=',')

    VOCABULARY_PATH = 'example_vocab.pkl'
    n_rows = args.nrows
    n_cols = args.ncols
    logger.info(f'Creating data from inputs: nrows={n_rows} ncols={n_cols}')
    df = make_data(n_rows, n_cols)
    df = df.reset_index(names='time')

    categorical_cols, float_cols, integer_cols, vector_cols = get_col_dtypes_from_make_data(df)
    excluded = ['time']
    # create tabular structure per col
    tab_struct = tab_structure(df,
                               categorical_columns=categorical_cols,
                               float_columns=float_cols,
                               integer_columns=integer_cols,
                               vector_columns=vector_cols,
                               excluded=excluded)
    logger.info(f'Compute the vocabulary')
    col_codes = column_codes(tab_struct, df)
    logger.info(f'Each row uses {sum(col_codes.sizes) + 1} tokens')
    logger.info(f'Vocab size: {col_codes.vocab_size}')

    # save the vocab
    with open(VOCABULARY_PATH, 'wb') as handle:
        pickle.dump(col_codes, handle)
    logger.info(f'Instantiating the HF Tabular Tokenizer')
    tabular_tokenizer = HFTabularTokenizer(col_codes,
                                         special_tokens=[CONSTS.NEW_LINE, CONSTS.END_OF_TEXT],
                                         delimiter=CONSTS.DELIMITER
                                         )

    # logging some info about the data.
    for col, size in zip(col_codes.columns, col_codes.sizes):
        logger.info(f'{col}:\t {size}')
    logger.info('\nTransformations for float columns:\n')
    for col in df.columns:
        if col not in excluded and hasattr(tabular_tokenizer.code_column.column_codes[col], 'transform'):
            logger.info(f'{col}: {tabular_tokenizer.code_column.column_codes[col].transform}')

    logger.info('\nCreate Documents from tabular data\n')

    # Document parameters
    SEQ_LEN = args.seqlen  # number of tokens in a document
    TOKENS_PER_ROW = sum(tabular_tokenizer.code_column.sizes) + 1
    NROWS = math.floor(SEQ_LEN / TOKENS_PER_ROW)
    STRIDE = args.stride

    # train, valid, test split
    train = df.loc[df.time < math.ceil(len(df)*0.9)]
    valid = df.loc[(math.ceil(len(df)*0.9) < df.time) & (df.time < math.ceil(len(df)*0.95))]
    test = df.loc[df.time >= math.ceil(len(df)*0.95)]

    columns = [c for c in df.columns if c != 'time']
    train = train[columns]
    valid = valid[columns]
    test = test[columns]

    train_docs = make_multiple_docs(train, n_rows=NROWS, stride_rows=STRIDE, delimiter=CONSTS.DELIMITER)
    valid_docs = make_multiple_docs(valid, n_rows=NROWS, stride_rows=STRIDE, delimiter=CONSTS.DELIMITER)
    test_docs = make_multiple_docs(test, n_rows=NROWS, stride_rows=STRIDE, delimiter=CONSTS.DELIMITER)

    with open('train.json', 'w') as f:
        # json.dump(make_docs(train)[0], f)
        json.dump([i[0] for i in train_docs], f)
    # todo take part of the valid for the test set for continuity
    with open('valid.json', 'w') as f:
        # json.dump(make_docs(valid)[0], f)
        if isinstance(valid_docs[0], list):
            json.dump([i[0] for i in valid_docs], f)
        else:
            json.dump(valid_docs, f)
    with open('test.json', 'w') as f:
        # json.dump(make_docs(test)[0], f)
        if isinstance(test_docs[0], list):
            json.dump([i[0] for i in test_docs], f)
        else:
            json.dump(test_docs, f)

    ds = load_dataset('json', data_files={'train': 'train.json',
                                          'valid': 'valid.json',
                                          'test': 'test.json'},
                      num_proc=3)
    logger.info('Tokenizing the loaded dataset with the tabular tokenizer...')
    mapped_ds = ds.map(tokenized_datasets, batch_size=1, batched=True, remove_columns=["text"], num_proc=24)

    single_decoded = tabular_tokenizer.decode(mapped_ds['train']['input_ids'][0])
    batch_decoded = tabular_tokenizer.decode(mapped_ds['train']['input_ids'])

    single_df = decoded_str_to_df(single_item=single_decoded,
                                  columns=columns,
                                  tab_struct=tab_struct,
                                  delimiter=CONSTS.DELIMITER,
                                  endoftext=CONSTS.END_OF_TEXT,
                                  newline=CONSTS.NEW_LINE)
    logger.info('Successfully tokenized example data using the Tabular Tokenizer and successfully decoded the data and '
                'converted back to a DataFrame')
