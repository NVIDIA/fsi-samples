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

from typing import Dict, List, Optional

import numpy as np
from numpy.random import default_rng
import pandas as pd

from coder.column_code import ColumnCodes
from tests.conftest import generate_categorical, generate_multimodal_data, generate_single_dist, get_distr_args


def make_data(nrows, ncols) -> pd.DataFrame:
    rng = default_rng(seed=0)
    assert isinstance(nrows, int) and nrows > 0
    assert isinstance(ncols, int) and ncols >= 4, 'we want to use one of each column type first, the rest will be ' \
                                                  'randomly chosen'
    column_options = ['categorical_letter', 'categorical_integer', 'float_single_dist', 'float_multi_dist']
    distribution_options = ['normal', 'lognormal', 'laplace', 'exponential', 'poisson', 'power', 'cauchy', 't']

    options = rng.choice(column_options, size=ncols - len(column_options))
    columns = np.concatenate([column_options, options])

    data = {str(i)+'_'+columns[i]: [] for i in range(len(columns))}

    # loop over columns and generate the rows

    for col_name, column_type in zip(data, columns):
        if column_type == 'categorical_letter':
            array = generate_categorical(rng, nrows, generate_letters=True)
        elif column_type == 'categorical_integer':
            array = generate_categorical(rng, nrows, generate_letters=False)
        elif column_type == 'float_single_dist':
            dist = rng.choice(distribution_options)
            args = get_distr_args(rng, dist)
            array = generate_single_dist(rng, nrows, dist, *args)
        elif column_type == 'float_multi_dist':
            dists = rng.choice(distribution_options, size=rng.integers(2, 10))
            array = generate_multimodal_data(rng, max(nrows, 30 * len(dists)), dists)
        else:
            raise TypeError()
        data[col_name] = array
    return pd.DataFrame(data)


def tab_structure(data: pd.DataFrame,
                  categorical_columns: List,
                  float_columns: Optional[List] = None,
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
        elif c in float_columns:
            item = {
                "name": c,
                'idx': idx,
                "code_type": "float",
                "args": {
                    "code_len": 3,  # number of tokens used to code the column
                    "base": 15,  # the positional base number. i.e. it uses 32 tokens for one digit
                    "fill_all": True,  # whether to use full base number for each token or derive it from the data.
                    "has_nan": True,  # can it handles nan or not
                    "transform": "best"
                    # can be ['yeo-johnson', 'quantile', 'robust', 'log1p', 'identity'], check https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
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
            tab_struct.append(item)
    if vector_columns:
        tab_struct.append(vector_cols_structure)
    return tab_struct


def column_codes(tab_struct, df):
    example_arrays = []
    for idx, col in enumerate(tab_struct):

        if col['code_type'] == 'category':
            example_arrays.append(df[col['name']].unique())
        elif col['code_type'] in ['int', 'float']:
            example_arrays.append(df[col['name']].dropna().unique())
        elif col['code_type'] == 'vector':
            example_arrays.append(df[col['name']])
        else:
            raise TypeError(f'Code_type for col: {col} must be one of "float", "int", "category", or "vector"')

    cc = ColumnCodes.get_column_codes(tab_struct, example_arrays)
    return cc


def make_docs(frame, DELIMITER) -> List[Dict]:
    docs = []
    u = frame[frame.columns[0]].astype(str).copy()
    for col in frame.columns[1:]:
        u = u.str.cat(frame[col].astype(str), sep=DELIMITER)
    doc = {'text': u.str.cat(sep='\n')}
    docs.append(doc)
    return docs


def make_multiple_docs(frame, n_rows: int, stride_rows: int, delimiter: str) -> List:
    docs = []
    if len(frame) > n_rows:
        for i in range(len(frame) - n_rows + 1):
            if (len(frame.iloc[i*stride_rows: i*stride_rows + n_rows]) == n_rows or
                    len(frame.iloc[i*stride_rows: i*stride_rows + n_rows]) == len(frame)):
                docs.append(make_docs(frame.iloc[i*stride_rows:i*stride_rows + n_rows], DELIMITER=delimiter))
    else:
        docs = make_docs(frame, DELIMITER=delimiter)
    return docs

