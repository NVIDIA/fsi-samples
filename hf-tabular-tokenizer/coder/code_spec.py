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

from typing import List, Tuple

from numpy import ndarray

__all__ = ["Code"]


class Code(object):

    def __init__(self, col_name: str, code_len: int, start_id: int):
        """
        @params:
            col_name: name of the column
            code_len: number of tokens used to code the column.
            start_id: offset for token_id.
            fillall: if True, reserve space for digit number even the digit number is
            not present in the data_series. Otherwise, only reserve space for the numbers
            in the data_series.
            hasnan: if True, reserve space for nan
        """
        self.name = col_name
        self.code_len = code_len
        self.start_id = start_id
        self.end_id = start_id

    def compute_code(self, data_series: ndarray):
        """
        @params:
            data_series: an array of input data used to calculate mapping from item to token_id and vice versa
        """
        raise NotImplementedError()

    def encode(self, item: str) -> List[int]:
        raise NotImplementedError()

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError()

    @property
    def code_range(self) -> List[Tuple[int, int]]:
        """
        get the vocab id range for each of the encoded tokens
        @returns [(min, max), (min, max), ...]
        """
        return [(self.start_id, self.end_id)]
