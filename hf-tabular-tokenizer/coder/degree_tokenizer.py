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

from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Union, List, Tuple
from itertools import product

import numpy as np

from .datatype_code import FloatCode


class DegreeCode(object):
    """
    Tokenize Degrees

    Example usage:
    tokenizer = DegreeTokenizer(start_tokenid=0, max_degree=360, precision='two_decimal')
    tokenizer.transform(X)

    """
    def __init__(self, start_tokenid: int, max_degree: Union[int, float], precision: str = 'four_decimal',
                 transform=None, custom_num_tokens: int = 0):
        if precision not in ['integral', 'two_decimal', 'four_decimal', 'log']:
            raise NotImplementedError
        if max_degree not in [180, 360]:
            raise ValueError('Should choose a max degree to be 180 or 360')
        if transform is not None and precision != 'log':
            raise ValueError('transform parameter only used when precision == "log"')
        # for theta = 360 , and phis each have max degrees of 180
        self.include_max_degree = 1 if max_degree == 180 else 0

        self.precision = precision
        self.mapping = {'integral': 1, 'two_decimal': 2, 'four_decimal': 3, 'log': 3}
        self.start_tokenid = start_tokenid
        self.end_tokenid = start_tokenid
        self.max_degree = max_degree
        self.token_to_id = {}
        self.id_to_token = {}
        self.decimal_tok_to_id = {}
        self.id_to_decimal_tok = {}
        self.four_decimal_tok_to_id = {}
        self.id_to_four_decimal_tok = {}
        if self.precision == 'integral':
            self.create_integral_mapping()
        elif self.precision == 'two_decimal':
            self.create_two_decimal_mapping()
        elif self.precision == 'four_decimal':
            self.create_four_decimal_mapping()
        elif self.precision == 'log' and transform is not None:
            if custom_num_tokens > 0:
                self.mapping['log'] = custom_num_tokens
            self.float_code = FloatCode('degree',
                                        code_len=self.mapping['log'],
                                        base=100,
                                        start_id=self.start_tokenid,
                                        has_nan=False,
                                        transform=transform
                                        )
            self.create_log_decimal_mapping()
        else:
            raise ValueError('some parameter combination is incorrect')

    @property
    def code_range(self) -> List[Tuple[int, int]]:
        """
        get the vocab id range for each of the encoded tokens
        @returns [(min, max), (min, max), ...]
        """
        return [(self.start_tokenid, self.end_tokenid)]

    def create_integral_mapping(self):
        for degree_token in range(0, self.max_degree + self.include_max_degree):
            self.token_to_id[str(degree_token)] = self.end_tokenid
            self.id_to_token[self.end_tokenid] = str(degree_token)
            self.end_tokenid += 1

    def create_two_decimal_mapping(self):
        """
        # todo refactor
        creates integral mapping for degrees followed by two digit decimal mapping
        :return:
        """
        self.create_integral_mapping()
        two_decimal = [''.join(decimal_tuple) for decimal_tuple in product([str(digit) for digit in range(10)],
                                                                           repeat=2)]
        for dec in two_decimal:
            self.decimal_tok_to_id[dec] = self.end_tokenid
            self.id_to_decimal_tok[self.end_tokenid] = dec
            self.end_tokenid += 1

    def create_four_decimal_mapping(self):
        """
        # todo refactor
        creates integral mapping for degrees followed by two digit decimal mapping
        :return:
        """
        self.create_two_decimal_mapping()

        two_decimal = [''.join(decimal_tuple) for decimal_tuple in product([str(digit) for digit in range(10)],
                                                                           repeat=2)]
        for dec in two_decimal:
            self.four_decimal_tok_to_id[dec] = self.end_tokenid
            self.id_to_four_decimal_tok[self.end_tokenid] = dec
            self.end_tokenid += 1

    def create_log_decimal_mapping(self):
        self.float_code.compute_code(np.linspace(0, self.max_degree, self.max_degree * 10 ** 3))
        self.end_tokenid = self.float_code.end_id

    def degrees2tokens(self, degrees: float) -> List:
        """
        transform from degrees to tokens
        :param degrees:
        :return:
        """
        if degrees < 0:
            raise ValueError('Degree tokenizer only accepts positive degree values to save space by not encoding the '
                             'parity')
        elif degrees >= self.max_degree:
            raise KeyError('Degrees is >= max degree')
        integral_degrees, decimal_degrees = f'{np.round(degrees,4):.4f}'.split('.')
        if self.precision == 'integral':
            return [self.token_to_id[integral_degrees]]
        elif self.precision == 'two_decimal':
            return [self.token_to_id[integral_degrees], self.decimal_tok_to_id[decimal_degrees[:2]]]
        elif self.precision == 'four_decimal':
            return [self.token_to_id[integral_degrees], self.decimal_tok_to_id[decimal_degrees[:2]],
                    self.four_decimal_tok_to_id[decimal_degrees[2:]]]
        else:
            return self.float_code.encode(degrees)

    def tokens2degrees(self, tokens: list) -> float:
        if len(tokens) == 0 or len(tokens) != self.mapping[self.precision]:
            raise ValueError(f'length of tokens should be 1, 2, or 3. Input len(tokens)={len(tokens)} '
                             f'for precision={self.precision}')
        if len(tokens) == 1 and self.precision == 'integral':
            # integral
            return float(self.id_to_token[tokens[0]])
        elif len(tokens) == 2 and self.precision == 'two_decimal':
            # two decimal
            degree = float('.'.join([self.id_to_token[tokens[0]], self.id_to_decimal_tok[tokens[1]]]))
        elif len(tokens) == 3 and self.precision == 'four_decimal':
            # four decimal
            degree = float('.'.join([self.id_to_token[tokens[0]], self.id_to_decimal_tok[tokens[1]] +  # join decimals
                                     self.id_to_four_decimal_tok[tokens[2]]
                                     ]))
        else:
            degree = float(self.float_code.decode(tokens))
        # degree = self.max_degree if degree > self.max_degree else degree
        return degree

    def fit(self, X):
        return self

    def transform(self, X: Union[float, List]) -> List:
        """
        A single degree or list of degrees to tokenize
        """
        if isinstance(X, (int, float)):
            return [self.degrees2tokens(X)]
        return [self.degrees2tokens(degree) for degree in X]

    def fit_transform(self, X, y=None) -> List:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: List[List]) -> List:
        """
        A list of tokens, or list of lists of tokens to decode back to degrees
        """
        return [self.tokens2degrees(tokens) for tokens in X]

    def decode(self, tokens: list):
        return self.inverse_transform(tokens)
