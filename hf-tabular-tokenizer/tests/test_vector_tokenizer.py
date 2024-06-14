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

import pytest

from coder.vector_tokenizer import VectorCode


@pytest.fixture()
def data_cols(data):
    return [col for col in data.columns if 'float' in col]


@pytest.fixture()
def vect_tokenizer(data, data_cols):
    vect_tokenizer = VectorCode(col_name=data_cols, start_id=0, base=100, hasnan=True, frac_below_min=0.1,
                                custom_degrees_tokens=4)
    vect_tokenizer.compute_code(data[data_cols])
    return vect_tokenizer


def test_compute_code(vect_tokenizer, data):
    # # df, columns, date_cols, data_cols = data
    # vect_tokenizer.compute_code(data[list(vect_tokenizer.names)])
    assert isinstance(vect_tokenizer.names, tuple)

    assert vect_tokenizer.start_id == 0
    assert all(hasattr(vect_tokenizer, attr) for attr in ('float_code', 'phi_degree_codes', 'theta_degree_code'))
    with pytest.raises(KeyError):
        _ = vect_tokenizer.theta_degree_code.id_to_decimal_tok[vect_tokenizer.end_id]
    if vect_tokenizer.precision == 'two_decimal':
        assert vect_tokenizer.theta_degree_code.id_to_decimal_tok[vect_tokenizer.end_id - 1]
    elif vect_tokenizer.precision == 'four_decimal':
        assert vect_tokenizer.theta_degree_code.id_to_four_decimal_tok[vect_tokenizer.end_id - 1]
    assert vect_tokenizer.code_len == vect_tokenizer.float_code_len + (
            len(vect_tokenizer.names) - 1) * vect_tokenizer.num_tokens_for_degrees


def test_encode(vect_tokenizer, data, data_cols):
    # df, columns, date_cols, data_cols = data
    # vect_tokenizer.compute_code(df[data_cols])
    tokens = vect_tokenizer.encode(data[data_cols])
    # code_range = vect_tokenizer.float_code.code_range + vect_tokenizer.phi_degree_code.code_range + \
    #              vect_tokenizer.theta_degree_code.code_range
    code_range = vect_tokenizer.float_code.code_range + [i.code_range[0] for i in vect_tokenizer.phi_degree_codes] + \
                 vect_tokenizer.theta_degree_code.code_range

    if vect_tokenizer.precision == 'integral':
        num_tokens_for_degrees = 1
    elif vect_tokenizer.precision == 'two_decimal':
        num_tokens_for_degrees = 2
    else:
        num_tokens_for_degrees = 3
    ctr = 0
    for idx, rng in enumerate(code_range):
        rng = list(range(*rng))

        if idx < vect_tokenizer.float_code_len:
            assert all(i in rng for i in tokens[:, idx])
        elif rng != code_range[-1]:
            left = vect_tokenizer.float_code_len + ctr * vect_tokenizer.phi_degree_codes[0].mapping[
                vect_tokenizer.phi_degree_codes[0].precision]
            right = vect_tokenizer.float_code_len + + (ctr + 1) * vect_tokenizer.phi_degree_codes[0].mapping[
                vect_tokenizer.phi_degree_codes[0].precision]
            assert all([set(i) - set(rng) == set() for i in tokens[:, left:right]])
            # [set(i) - set(rng) == set() for i in tokens[:, vect_tokenizer.float_code_len:-num_tokens_for_degrees]])
            ctr += 1
        else:
            assert all([set(i) - set(rng) == set() for i in tokens[:, -num_tokens_for_degrees:]])


def test_decode(vect_tokenizer, data, data_cols):
    # df, columns, date_cols, data_cols = data
    vect_tokenizer.compute_code(data[data_cols])
    tokens = vect_tokenizer.encode(data[data_cols])
    out = vect_tokenizer.decode(tokens)  # , df=df[data_cols])
    # if this fails, it may be because there's not enough precision for the column's range
    assert ((out.isna() == data[data_cols].isna()).sum() == data[data_cols].shape[0]).all()
