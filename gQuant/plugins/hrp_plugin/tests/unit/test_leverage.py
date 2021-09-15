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
python -m unittest tests/unit/test_leverage.py -v
'''
import unittest
import pandas as pd
import cupy
import cudf
from greenflow_hrp_plugin.kernels import _get_log_return_matrix
from greenflow_hrp_plugin.kernels import _get_month_start_pos
from greenflow_hrp_plugin.kernels import leverage_for_target_vol, MAX_YEARS
import math


class TestDistance(unittest.TestCase):

    def create_df(self):
        date_df = cudf.DataFrame()
        date_df['date'] = pd.date_range('1/1/1990', '12/31/1991', freq='B')
        full_df = cudf.concat([date_df, date_df])
        sample_id = cupy.repeat(cupy.arange(2), len(date_df))
        full_df['sample_id'] = sample_id
        full_df['year'] = full_df['date'].dt.year
        full_df['month'] = full_df['date'].dt.month-1
        cupy.random.seed(3)
        full_df['portfolio'] = cupy.random.rand(len(full_df))
        return full_df

    def setUp(self):
        self.df = self.create_df()

    def test_months_start(self):
        log_return = self.df
        first_sample = log_return['sample_id'].min().item()
        all_dates = log_return[first_sample == log_return['sample_id']]['date']
        months_start = _get_month_start_pos(all_dates)
        self.assertTrue(months_start[0].item() == 0)
        for i in range(1, len(months_start)):
            start_day_month = log_return.iloc[months_start[i].item(
            )]['date'].dt.month
            last_day_month = log_return.iloc[(
                months_start[i].item()-1)]['date'].dt.month
            diff = start_day_month.values[0] - last_day_month.values[0]
            self.assertTrue(abs(diff) != 0)

    def test_distance(self):
        total_samples = 2
        # window = 3
        long_window = 59
        short_window = 19
        target_vol = 0.05
        log_return = self.df
        first_sample = log_return['sample_id'].min().item()
        all_dates = log_return[first_sample == log_return['sample_id']]['date']
        all_dates = all_dates.reset_index(drop=True)
        months_start = _get_month_start_pos(all_dates)
        for window in range(len(months_start)):
            if (months_start[window] - long_window) > 0:
                break
        print(window)
        print('offset', months_start[window] - long_window)
        port_return_ma = log_return['portfolio'].values.reshape(
            total_samples, -1)
        number_of_threads = 256
        num_months = len(months_start) - window
        if num_months == 0:  # this case, use all the data to compute
            num_months = 1
        number_of_blocks = num_months * total_samples
        leverage = cupy.zeros((total_samples, num_months))
        leverage_for_target_vol[(number_of_blocks, ), (number_of_threads, ), 0,
                                256 * MAX_YEARS * 8](leverage, port_return_ma,
                                                     months_start, num_months,
                                                     window,
                                                     long_window, short_window,
                                                     target_vol)

        for sample in range(2):
            for num in range(num_months):

                end_id = months_start[num + window]
                mean = port_return_ma[sample,
                                      end_id - long_window:end_id].mean()
                sd_long = cupy.sqrt(
                    ((port_return_ma[sample, end_id - long_window:end_id] -
                      mean)**2).mean())
                # print('long', sd_long)
                mean = (port_return_ma[sample,
                                       end_id - short_window:end_id].mean())
                sd_short = cupy.sqrt(
                    ((port_return_ma[sample, end_id - short_window:end_id] -
                      mean)**2).mean())

                # print('sort', sd_short)
                max_sd = max(sd_long, sd_short)
                lev = target_vol / (max_sd * math.sqrt(252))
                # print(lev)
                # print(leverage[sample, num], lev-leverage[sample, num])
                # compute = means[sample][num]
                self.assertTrue(cupy.allclose(leverage[sample, num], lev))

        # for sample in range(2):
        #     for num in range(num_months):
        #         s = log_return_ma[sample, :, months_start[num]:months_start[
        #             num + window]]
        #         truth = (cupy.cov(s, bias=True))
        #         compute = cov[sample][num]
        #         self.assertTrue(cupy.allclose(compute, truth))

        # for sample in range(1):
        #     for num in range(1):
        #         cov_m = cov[sample][num]
        #         corr_m = cov_m.copy()
        #         for i in range(3):
        #             for j in range(3):
        #                 corr_m[i, j] = corr_m[i, j] / \
        #                     math.sqrt(cov_m[i, i] * cov_m[j, j])
        #         dis = (1.0 - corr_m)/2.0
        #         truth = (squareform(dis.get()))
        #         compute = distance[sample][num]
        #         self.assertTrue(cupy.allclose(compute, truth))
