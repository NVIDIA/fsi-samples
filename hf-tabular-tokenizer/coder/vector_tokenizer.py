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

from typing import List, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer  # MinMaxScaler, StandardScaler,

from .hyperspherical_coords import hyperspherical2cartesian_matrix, \
    cartesian2hyperspherical_matrix  # cartesian2hyperspherical

from .code_spec import Code
from .datatype_code import FloatCode
from .degree_tokenizer import DegreeCode


class VectorCode(Code):
    """
    Take an input matrix, treat each row as a cartesian vector.
    1. Apply a quantile transformation to each column in the matrix. Map output to -1, 1 by 2 * quantile_transform -1
    2. Convert matrix (i.e. each row in the matrix) to hyperspherical coordinates
    3. Scale the radius with a second quantile transform? Optional
    4. encode (tokenize) the radius w/ 4 tokens, and each of the degrees as 2 tokens
    5. decode the tokens  and apply the reverse steps
    """

    def __init__(
            self,
            col_name: Union[List[str], Tuple[str]],
            start_id: int,
            radius_code_len: int = 5,
            base: int = 100,
            hasnan: bool = True,
            frac_below_min=0,
            degrees_precision='log',
            custom_degrees_tokens=0
    ):
        # basic input validation
        # if not isinstance(col_names, list) or not isinstance(code_lens, list) or not isinstance(transforms, list):
        #     raise TypeError('col_names, code_lens, and transforms should be a list type')
        # if len(col_names) < 2 or len(code_lens) < 2 or len(transforms) < 2:
        #     raise ValueError('col_names, code_lens, and transforms should be list of length 2 or greater')
        # if not (len(col_names) == len(code_lens) == len(transforms)):
        #     raise ValueError('lengths of col_names, code_lens, and transforms should be equal')

        # for now, does not give granularity to specify fillall and hasnans on per column level
        super().__init__(col_name='vector', code_len=5, start_id=start_id)
        if isinstance(col_name, list):
            col_name = tuple(col_name)
        self.names = col_name

        self.base = base
        self.hasnan = hasnan
        assert frac_below_min >= 0, 'fraction below minimum value should be a positive float'
        self.frac_below_min = frac_below_min
        assert frac_below_min != 1
        if self.hasnan:
            self.frac_below_min = self.frac_below_min or 0.1
        self.scaling_numerator = 2
        self.scaling_factor = self.scaling_numerator / (1 - self.frac_below_min)
        self.nan_rng = None

        self.float_code = None
        self.float_code_len = radius_code_len
        # self.phi_degree_code = None
        self.phi_degree_codes = []
        self.theta_degree_code = None
        self.precision = degrees_precision
        if self.precision == 'integral':
            self.num_tokens_for_degrees = 1
        elif self.precision == 'two_decimal':
            self.num_tokens_for_degrees = 2
        elif self.precision == 'four_decimal':
            self.num_tokens_for_degrees = 3
        elif self.precision == 'log':
            if custom_degrees_tokens > 0:
                self.num_tokens_for_degrees = custom_degrees_tokens
            else:
                self.num_tokens_for_degrees = 3

        # if scaler == 'std':
        #     self.scaler = StandardScaler()
        # elif scaler == 'quantile':
        # todo this could be an optimization point on per col basis, ex. n_quantiles increase
        self.scaler = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
        # self.radius_scaler = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
        # elif scaler is None:
        #     self.scaler = None
        # else:
        #     raise ValueError('scaler should be one of "quantile" or "std"')

    def fit(self, dataframe: pd.DataFrame):
        self.compute_code(dataframe)
        return self

    def compute_code(self, dataframe: pd.DataFrame):
        # handles nans, to check mean/median use np.nanmean() and np.nanmedian(), etc.
        # ex. np.nanmax(scaled_data[:,0]) == 1  NOT np.nanmax(scaled_data[0,:])
        # scales the data on the columns. Convert to cartesian range of -1,1 for spherical coord transformation
        # frac_below_min = 0
        if dataframe.isna().any().sum() > 0 and self.hasnan is False:
            self.hasnan = True
            self.frac_below_min = self.frac_below_min or 0.1

        if self.hasnan:
            # nan_mask = dataframe.isna().values
            # frac_below_min = 0.1
            # nan_rng = frac_below_min * (df[data_cols].max(axis=0) - df[data_cols].min(axis=0))
            # df[data_cols] = df[data_cols].fillna(nan_rng, axis=0)
            self.nan_rng = dataframe.min(axis=0) - self.frac_below_min * (dataframe.max(axis=0) - dataframe.min(axis=0))
            dataframe = dataframe.fillna(self.nan_rng, axis=0)

        # scaling factor, stretching and shifting range from (0,1) to (-1,1), going slightly less than -1 for the NaNs
        # thus the L2 norm of a vector with NaNs will be slightly longer than otherwise
        # self.scaling_factor = self.scaling_numerator / (1 - self.frac_below_min)
        scaled_data = self.scaling_factor * self.scaler.fit_transform(dataframe) - (
                    self.scaling_factor - self.scaling_numerator / 2)
        # -1 to shift domain

        # todo figure out nans
        # spherical_data = np.empty(shape=scaled_data[nan_mask.any(axis=1) == False].shape)
        # scaled_data = np.nan_to_num(scaled_data)  # todo
        spherical_data = cartesian2hyperspherical_matrix(scaled_data)
        # for row in np.argwhere(nan_mask.any(axis=1)): # todo
        #     scaled_data[row, :]

        # spherical_data[:, 0] = self.radius_scaler.fit_transform(spherical_data[:, 0, np.newaxis])[:,0]
        self.float_code = FloatCode('radius',
                                    self.float_code_len,
                                    self.start_id,
                                    base=self.base,
                                    transform='best')

        self.float_code.compute_code(spherical_data[:, 0])
        # FloatCode end_id is exlusive of the range, i.e. not in the vocab

        phi_degree_code = DegreeCode(start_tokenid=self.float_code.end_id,
                                     max_degree=180,
                                     precision=self.precision,
                                     transform='identity' if self.precision == 'log' else None,
                                     custom_num_tokens=4
                                     )
        self.phi_degree_codes.append(phi_degree_code)

        for degree_col in range(spherical_data.shape[1] - 3):
            phi_degree_code = DegreeCode(start_tokenid=self.phi_degree_codes[-1].end_tokenid,
                                         max_degree=180,
                                         precision=self.precision ,
                                         # compute the best transform only once.
                                         transform='identity' if self.precision == 'log' else None,
                                         custom_num_tokens=4
                                         )
            self.phi_degree_codes.append(phi_degree_code)

        self.theta_degree_code = DegreeCode(start_tokenid=self.phi_degree_codes[-1].end_tokenid,
                                            max_degree=360,
                                            precision=self.precision,
                                            transform='identity' if self.precision == 'log' else None,
                                            custom_num_tokens=4
                                            )

        self.code_len = self.float_code_len + (len(self.names) - 1) * self.num_tokens_for_degrees
        self.end_id = self.theta_degree_code.end_tokenid

    def encode(self, dataframe) -> np.array:
        """

        :param array: CAN BE STRING?
        :return:
        """

        # handle nans and convert input data to spherical coordinates
        if not isinstance(dataframe, pd.DataFrame):
            # attempt to convert array into dataframe
            dataframe = pd.DataFrame(dataframe, columns=self.names)
        if self.hasnan:
            dataframe = dataframe.fillna(self.nan_rng, axis=0)

        scaled_data = self.scaling_factor * self.scaler.transform(dataframe) - (
                    self.scaling_factor - self.scaling_numerator / 2)

        spherical_data = cartesian2hyperspherical_matrix(scaled_data)

        # now convert spherical data to tokens
        out = np.empty((spherical_data.shape[0], self.float_code.code_len +
                        self.num_tokens_for_degrees * (spherical_data.shape[1] - 1)),
                       dtype=np.uint16)

        out[:, :self.float_code_len] = np.array([self.float_code.encode(str(i)) for i in spherical_data[:, 0]])
        ctr = self.float_code_len
        for col in range(1, spherical_data.shape[1] - 1):
            out[:, ctr:ctr + self.num_tokens_for_degrees] = self.phi_degree_codes[col-1].transform(spherical_data[:, col])
            ctr += self.num_tokens_for_degrees
        out[:, -self.num_tokens_for_degrees:] = self.theta_degree_code.transform(spherical_data[:, -1])
        return out  # .tolist()

    def decode(self, ids, **kwargs) -> pd.DataFrame:
        """

        :param ids:
        :return:
        """
        if not isinstance(ids, np.ndarray):
            ids = np.array(ids)
        # convert tokens to spherical coordinates
        spherical_coords = np.empty((ids.shape[0], len(self.names)))

        # radius
        spherical_coords[:, 0] = np.array([float(self.float_code.decode(i)) for i in ids[:, :self.float_code_len]])
        # phi
        ctr = 1
        for col in range(self.float_code_len, ids.shape[1] - self.num_tokens_for_degrees, self.num_tokens_for_degrees):
            spherical_coords[:, ctr] = self.phi_degree_codes[ctr-1].inverse_transform(ids[:, col:col + self.num_tokens_for_degrees])
            ctr += 1
        # theta
        spherical_coords[:, -1] = self.theta_degree_code.inverse_transform(ids[:, -self.num_tokens_for_degrees:])

        # now convert spherical coordinates back to original data space
        scaled_data = hyperspherical2cartesian_matrix(spherical_matrix=spherical_coords)

        # scaled_data = self.scaling_factor * self.scaler.transform(dataframe) - (self.scaling_factor - 1)
        data = self.scaler.inverse_transform(
            (scaled_data + (self.scaling_factor - self.scaling_numerator / 2)) / self.scaling_factor)
        # get back the NaNs
        data[np.isclose(data, self.nan_rng, rtol=1e-3)] = np.nan

        dataframe = pd.DataFrame(data, columns=self.names)
        return dataframe
