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
############################################################################

import argparse
import json
import os
import pickle
import shutil
import sys
from typing import Optional, List, Dict
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from coder.column_code import ColumnCodes
from coder.tabular_tokenizer import TabularTokenizer
from utils.utils import create_logger


def read_data(path: str):
    """
    Reads in the CSV file and assigns the NaNs, columns, and DATE columns split up.
    Returns the cleaned up Pandas DataFrame
    """
    df = pd.read_csv(path, na_values=[-999999999, -999999])

    return df


def tab_structure(data: pd.DataFrame,
                  categorical_columns: List,
                  float_cols: Optional[List] = None,
                  fp16_cols: Optional[List] = None,
                  binned_categorical_cols: Optional[List] = None,
                  integer_columns: Optional[List] = None,
                  vector_columns: Optional[List] = None,
                  excluded: Optional[List] = None) -> List[Dict]:
    columns = data.columns
    tab_struct = []
    # picking a random float col to stay a float col, not vector type
    # date_cols = []
    # float_cols = [col for col in columns if 'float' in col]  # request.param
    # categorical_columns = [col for col in columns if 'letter' in col]
    # integer_columns = [col for col in columns if 'integer' in col]
    # vector_columns = []  # [col for col in data_cols if col not in float_col]
    vector_cols_structure = {'name': [],
                             'idx': [],
                             "code_type": "vector",
                             # all the vector args should be the same for all vector columns
                             "args": {"radius_code_len": 5,  # number of tokens used to code the column
                                      "degrees_precision": 'four_decimal',
                                      'custom_degrees_tokens': 0,
                                      "base": 100,  # the positional base number. ie. it uses 32 tokens for one digit
                                      "fill_all": True,
                                      # whether to use full base number for each token or derive it from the data.
                                      "has_nan": True,  # can it handles nan or not
                                      }
                             }

    for idx, c in enumerate([col for col in columns if col not in excluded]):
        item = {}

        if c in categorical_columns:
            item = {
                "name": c,
                'idx': idx,
                "code_type": "category",
            }
        elif c in binned_categorical_cols:
            item = {
                "name": c,
                'idx': idx,
                "code_type": "binned_category",
            }
        elif c in float_cols:

            uniques = df[c].nunique()
            if uniques < 1001:
                code_len = 2
            elif uniques < 8001:
                code_len = 3
            else:
                code_len = 4  # should cover the majority of cases
            base = int(np.ceil(uniques ** (1 / code_len)))

            item = {
                "name": c,
                'idx': idx,
                "code_type": "float",
                "args": {
                    "code_len": code_len,  # 3  number of tokens used to code the column
                    "base": base,  # 15 the positional base number. i.e. it uses 15 tokens for one digit
                    "fill_all": True,  # whether to use full base number for each token or derive it from the data.
                    "has_nan": True,  # can it handles nan or not
                    "transform": "best"
                    # can be ['yeo-johnson', 'quantile', 'robust', 'log1p', 'identity'], check https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
                }
            }
        elif c in fp16_cols:
            item = {
                "name": c,
                'idx': idx,
                "code_type": "fp16",
                "args": {
                    "code_len": 4,  # 4  number of tokens used to code the column
                    "base": None,  # 15 the positional base number. i.e. it uses 15 tokens for one digit
                    "fill_all": True,  # whether to use full base number for each token or derive it from the data.
                    "has_nan": True,  # can it handles nan or not
                }
            }
        elif c in integer_columns:
            item = {"name": c,
                    "code_type": "int",
                    "idx": idx,
                    "args": {
                        "code_len": 3,  # number of tokens used to code the column
                        "base": 100,  # the positional base number. ie. it uses 32 tokens for one digit
                        "fill_all": True,  # whether to use full base number for each token or derive it from the data.
                        "has_nan": True,  # can it handles nan or not
                    }}
        elif c in vector_columns:
            vector_cols_structure['name'].append(c)
            vector_cols_structure['idx'].append(idx)
        if item:

            if c == 'CO_TimeSpread':
                item['args']['code_len'] = 4

            tab_struct.append(item)
    if vector_columns:
        tab_struct.append(vector_cols_structure)
    return tab_struct


def column_codes(tab_struct, df):
    example_arrays = []
    for idx, col in enumerate(tab_struct):

        if col['code_type'] in ['category', 'binned_category']:
            example_arrays.append(df[col['name']].unique())
        elif col['code_type'] in ['int', 'float', 'fp16']:
            example_arrays.append(df[col['name']].dropna().unique())
        elif col['code_type'] == 'vector':
            example_arrays.append(df[col['name']])
        else:
            raise TypeError(f'Code_type for col: {col} must be one of "fp16", "float", "int", "category", or "vector"')

    cc = ColumnCodes.get_column_codes(tab_struct, example_arrays)
    return cc


def make_docs(frame, delimiter: str) -> List[Dict]:
    docs = []
    u = frame[frame.columns[0]].astype(str).copy()
    for col in frame.columns[1:]:
        u = u.str.cat(frame[col].astype(str), sep=delimiter)
    doc = {'text': u.str.cat(sep='\n')}
    docs.append(doc)
    return docs


def prep_dataframe(df):
    train = df.loc[df.year <= 2017].copy()
    valid = df.loc[df.year == 2018].copy()
    test = df.loc[df.year > 2018].copy()

    columns = [c for c in cols if c != 'year']
    train = train[columns]
    valid = valid[columns]
    test = test[columns]
    return train, valid, test


def print_args(args, logger=None):
    for arg in vars(args):
        if logger:
            logger.info(f"{arg:20} : {getattr(args, arg)}")
        else:
            print(f"{arg:20} : {getattr(args, arg)}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-file", nargs='+',
                   help='pass in single file with all train/val/test or separate train, val_and_test files. Max 2 files')
    p.add_argument("--with-binned-categorical", action="store_true")
    p.add_argument("--with-fp16", action="store_true")
    p.add_argument("--output-vocab-file", type=str)
    p.add_argument("--output-dir", type=str, help="output directory to vocab file and json files")

    args = p.parse_args()
    return args


if __name__ == '__main__':
    tqdm.pandas()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    warnings.filterwarnings("ignore")
    logger = create_logger(__name__)

    args = get_args()
    print_args(args, logger)

    # Path to the dataset
    assert len(args.input_file) < 3
    fpath = args.input_file[0]  # f'NVDARETURNS_20221121.CSV'
    VOCABULARY_PATH = args.output_dir + '/' + args.output_vocab_file  # 'finance_vocab_coder.pickle'
    assert len(VOCABULARY_PATH.rsplit('.', 1)) == 2 and VOCABULARY_PATH.rsplit('.', 1)[1] in ['pickle',
                                                                                              'pkl'], 'vocab path should be a pickle or pkl file format'

    END_OF_TEXT = '<|endoftext|>'
    NEW_LINE = '\n'
    DELIMITER = ','

    cat_columns = []
    float_cols = []
    fp16_cols = []
    binned_categorical_cols = []

    if args.with_binned_categorical:
        df = pd.read_csv(fpath)
        cat_columns = ['month', 'day', 'weekday']
        binned_categorical_cols = [col for col in df.columns if col not in ['year'] + cat_columns]
    elif args.with_fp16:
        df, cols, cat_columns, float_columns = read_data(fpath)
        logger.info('running tab_struct')
        fp16_cols = float_columns
    else:
        df, cols, cat_columns, float_columns = read_data(fpath)
        logger.info('running tab_struct')
        float_cols = float_columns

    tab_struct = tab_structure(df,
                               categorical_columns=cat_columns,
                               float_cols=float_cols,
                               fp16_cols=fp16_cols,
                               binned_categorical_cols=binned_categorical_cols,
                               excluded=['year']
                               )

    os.makedirs(args.output_dir, exist_ok=True)

    tab_struct_fp = args.output_dir + '/' + 'tab_struct.json'
    with open(tab_struct_fp, 'w') as fp:
        json.dump(tab_struct, fp, indent=4)

    logger.info('\ngetting column codes')
    cc = column_codes(tab_struct, df)
    logger.debug(cc)
    logger.info(f'each row uses {sum(cc.sizes) + 1} tokens\n')

    with open(VOCABULARY_PATH, 'wb') as handle:
        pickle.dump(cc, handle)
        logger.info("vocabulary written to disk")

    tabular_tokenizer = TabularTokenizer(cc,
                                         special_tokens=[NEW_LINE, END_OF_TEXT],
                                         delimiter=DELIMITER
                                         )

    # for col, size in zip(cc.columns, cc.sizes):
    #    print(f'{col}:\t {size}')

    # for col in df.columns:
    #    if col in cat_columns:
    #        continue
    #    print(col, tabular_tokenizer.code_column.column_codes[col].transform)

    logger.info(f"{sum(tabular_tokenizer.code_column.sizes)}\n" +
                f"{tabular_tokenizer.code_column.vocab_size}\n" +
                f"{len(tabular_tokenizer.code_column.columns)}")

    if len(args.input_file) == 1:
        train, val, test = prep_dataframe(df)
    else:
        train = df
        val_and_test = pd.read_csv(args.input_file[1])
        val = val_and_test.loc[val_and_test.year == 2018].copy()
        test = val_and_test.loc[val_and_test.year > 2018].copy()

        columns = [c for c in df.columns if c != 'year']
        train = train[columns]
        val = val[columns]
        test = test[columns]

    total: list = make_docs(train, DELIMITER) + make_docs(val, DELIMITER) + make_docs(test, DELIMITER)

    jsonlines_path = f'{args.output_dir}/tmp.jl'
    with open(jsonlines_path, 'w') as f:
        f.write('\n'.join([json.dumps(row) for row in total]))

    # test the encoding/decoding process
    with open(jsonlines_path, 'r') as f:
        for line in f:
            break
    os.remove(jsonlines_path)

    text = json.loads(line)['text']
    token_ids = tabular_tokenizer.encode(text)
    tex = tabular_tokenizer.decode(token_ids)

    # create separate train, test, val files for training dataset
    for filename, data in zip(['train.jl', 'val.jl', 'test.jl'], total):  # [train, val, test]):
        with open(os.path.join(args.output_dir, filename), 'w') as fp:
            fp.write(json.dumps(data))
            fp.write('\n')
    logger.info('complete')
