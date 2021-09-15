"""
 ////////////////////////////////////////////////////////////////////////////
 //
 // Copyright (C) NVIDIA Corporation.  All rights reserved.
 //
 // NVIDIA Sample Code
 //
 // Please refer to the NVIDIA end user license agreement (EULA) associated
 // with this source code for terms and conditions that govern your use of
 // this software. Any use, reproduction, disclosure, or distribution of
 // this software and related documentation outside the terms of the EULA
 // is strictly prohibited.
 //
 ////////////////////////////////////////////////////////////////////////////
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
python -m unittest tests/unit/test_max_drawdown.py -v
'''
import unittest
import pandas as pd
import cupy
import numpy as np
import cudf
from greenflow_hrp_plugin.kernels import _get_log_return_matrix
from greenflow_hrp_plugin.kernels import _get_month_start_pos
from greenflow_hrp_plugin.kernels import drawdown_kernel


class TestMaxDrawdown(unittest.TestCase):

    def create_df(self):
        date_df = cudf.DataFrame()
        date_df['date'] = pd.date_range('1/1/1990', '12/31/1992', freq='B')
        full_df = cudf.concat([date_df, date_df])
        sample_id = cupy.repeat(cupy.arange(2), len(date_df))
        full_df['sample_id'] = sample_id
        full_df['year'] = full_df['date'].dt.year
        full_df['month'] = full_df['date'].dt.month-1
        cupy.random.seed(3)
        full_df[0] = cupy.random.normal(0, 0.02, len(full_df))
        full_df[1] = cupy.random.normal(0, 0.02, len(full_df))
        full_df[2] = cupy.random.normal(0, 0.02, len(full_df))
        return full_df

    def setUp(self):
        self.df = self.create_df()

    def compute_drawdown(self, times):
        cumsum = np.cumsum(times)
        cumsum = np.exp(cumsum)
        maxreturn = np.maximum.accumulate(np.concatenate([np.array([1.0]),
                                                          cumsum]))[1:]
        drawdown = cumsum/maxreturn - 1.0
        return -drawdown.min()

    def test_max_drawdown(self):
        total_samples = 2
        window = 12
        log_return = self.df

        first_sample = log_return['sample_id'].min().item()
        all_dates = log_return[first_sample == log_return['sample_id']]['date']
        all_dates = all_dates.reset_index(drop=True)
        months_start = _get_month_start_pos(all_dates)
        log_return_ma = _get_log_return_matrix(total_samples, log_return)
        _, assets, timelen = log_return_ma.shape
        number_of_threads = 128
        num_months = len(months_start) - window
        number_of_blocks = num_months * total_samples
        drawdown = cupy.zeros((total_samples, num_months, assets))
        drawdown_kernel[(number_of_blocks, ),
                        (number_of_threads, )](drawdown, log_return_ma,
                                               months_start, window)
        for s in range(total_samples):
            for a in range(assets):
                for i in range(num_months):
                    gpu_drawdown = drawdown[s][i][a]
                    cpu_drawdown = self.compute_drawdown(
                        log_return_ma[s][a][
                            months_start[i]:months_start[i+window]].get())
                    self.assertTrue(cupy.allclose(gpu_drawdown,
                                                  cpu_drawdown))
