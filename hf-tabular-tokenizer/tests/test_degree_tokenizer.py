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

from coder.degree_tokenizer import DegreeCode


@pytest.fixture(params=['integral', 'two_decimal', 'four_decimal', 'log'])
def degree_tokenizer(request):
    return DegreeCode(start_tokenid=0, max_degree=360, precision=request.param,
                      transform='identity' if request.param == 'log' else None, custom_num_tokens=4)


def test_create_integral_mapping(degree_tokenizer):
    if degree_tokenizer.precision == 'integral':
        assert degree_tokenizer.end_tokenid  == degree_tokenizer.max_degree
    elif degree_tokenizer.precision == 'two_decimal':
        assert degree_tokenizer.end_tokenid == degree_tokenizer.max_degree + 100
    elif degree_tokenizer.precision == 'four_decimal':
        assert degree_tokenizer.end_tokenid == degree_tokenizer.max_degree + 100 + 100
    elif degree_tokenizer.precision == 'log':
        assert degree_tokenizer.end_tokenid == degree_tokenizer.mapping['log']*degree_tokenizer.float_code.base
    assert len(degree_tokenizer.token_to_id) == len(degree_tokenizer.id_to_token)
    assert list(degree_tokenizer.token_to_id.keys()) == list(degree_tokenizer.id_to_token.values())
    assert list(degree_tokenizer.token_to_id.values()) == list(degree_tokenizer.id_to_token.keys())


def test_create_two_decimal_mapping(degree_tokenizer):
    if degree_tokenizer.precision == 'integral':
        assert degree_tokenizer.decimal_tok_to_id == degree_tokenizer.id_to_decimal_tok == {}
    elif degree_tokenizer.precision == 'two_decimal':
        assert all(len(i) == 2 for i in list(degree_tokenizer.decimal_tok_to_id.keys()))
        assert list(degree_tokenizer.decimal_tok_to_id.keys()) == list(degree_tokenizer.id_to_decimal_tok.values())
        assert list(degree_tokenizer.decimal_tok_to_id.values()) == list(degree_tokenizer.id_to_decimal_tok.keys())
    elif degree_tokenizer.precision == 'four_decimal':
        assert all(len(i) == 2 for i in list(degree_tokenizer.decimal_tok_to_id.keys()))
        assert all(len(i) == 2 for i in list(degree_tokenizer.four_decimal_tok_to_id.keys()))

        assert list(degree_tokenizer.decimal_tok_to_id.keys()) == list(degree_tokenizer.id_to_decimal_tok.values())
        assert list(degree_tokenizer.decimal_tok_to_id.values()) == list(degree_tokenizer.id_to_decimal_tok.keys())
        assert list(degree_tokenizer.four_decimal_tok_to_id.keys()) == list(degree_tokenizer.id_to_four_decimal_tok.values())
        assert list(degree_tokenizer.four_decimal_tok_to_id.values()) == list(degree_tokenizer.id_to_four_decimal_tok.keys())


def test_degrees2tokens(degree_tokenizer):
    with pytest.raises(ValueError) as neg_degree:
        _ = degree_tokenizer.degrees2tokens(-123)
    with pytest.raises(KeyError) as not_found:
        _ = degree_tokenizer.degrees2tokens(360)

    for degrees in [0, 123, 180, 359.99]:
        tokens = degree_tokenizer.degrees2tokens(degrees)

        if degree_tokenizer.precision == 'integral':
            assert len(tokens) == 1
        elif degree_tokenizer.precision == 'two_decimal':
            assert len(tokens) == 2
        elif degree_tokenizer.precision == 'four_decimal':
            assert len(tokens) == 3
        elif degree_tokenizer.precision == 'log':
            assert len(tokens) == degree_tokenizer.mapping['log']


def test_tokens2degrees(degree_tokenizer):
    with pytest.raises(ValueError):
        _ = degree_tokenizer.tokens2degrees([])
    with pytest.raises(ValueError):
        _ = degree_tokenizer.tokens2degrees(list(range((degree_tokenizer.mapping['log'])*100, -100, -100)))

    if degree_tokenizer.precision == 'integral':
        degrees = degree_tokenizer.tokens2degrees([0])
        assert degrees == 0
    elif degree_tokenizer.precision == 'two_decimal':
        degrees = degree_tokenizer.tokens2degrees([0, 360])
        assert degrees == 0
    elif degree_tokenizer.precision == 'four_decimal':
        degrees = degree_tokenizer.tokens2degrees([0, 360, 460])
        assert degrees == 0
    else:
        degrees = degree_tokenizer.tokens2degrees(list(range( (degree_tokenizer.mapping['log'] -1 )*100, -100, -100 )))
        assert degrees == 0
    print(degrees)


def test_fit(degree_tokenizer):
    assert True


def test_transform(degree_tokenizer):
    assert True


@pytest.mark.parametrize('degrees', (0, 1, 123, 180, 359.99, [0, 1, 123, 180, 359.99, ]))
def test_fit_transform(degree_tokenizer, degrees):
    tokens = degree_tokenizer.fit_transform(degrees)
    if hasattr(degrees, '__iter__'):
        degrees = [i // 1 if degree_tokenizer.precision == 'integral' else i for i in degrees]
        assert all([degree_tokenizer.tokens2degrees(tok) == deg for tok, deg in zip(tokens, degrees)])
    else:
        degrees = degrees // 1 if degree_tokenizer.precision == 'integral' else degrees
        assert all(degree_tokenizer.tokens2degrees(tok) == degrees for tok in tokens)
    print(tokens)


@pytest.mark.parametrize(('tokens', 'degrees'), [
    ([[0, 360], [1, 360], [123, 360], [180, 360], [359, 459]], [0, 1, 123, 180, 359.99, ])
]
                         )
def test_inverse_transform(degree_tokenizer, tokens, degrees):
    # mapping of precision : number of tokens
    mapping = {'integral': 1, 'two_decimal': 2, 'four_decimal': 3, 'log': 3}

    # integral_tokens = True if len(tokens[0]) == 1 else False
    if len(tokens[0]) != mapping[degree_tokenizer.precision]:
        with pytest.raises(ValueError):
            _ = degree_tokenizer.inverse_transform(tokens)
    else:

        if hasattr(degrees, '__iter__'):
            degrees = [i // 1 if degree_tokenizer.precision == 'integral' else i for i in degrees]
            assert degree_tokenizer.inverse_transform(tokens) == degrees

        else:
            degrees = degrees // 1 if degree_tokenizer.precision == 'integral' else degrees
