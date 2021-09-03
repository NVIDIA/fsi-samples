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
python -m unittest tests/unit/test_distance.py -v
'''
import unittest
import pandas as pd
import cupy
import cudf
from greenflow_hrp_plugin.kernels import _get_log_return_matrix
from greenflow_hrp_plugin.kernels import _get_month_start_pos
from greenflow_hrp_plugin.kernels import compute_cov, MAX_YEARS
from scipy.spatial.distance import squareform
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
        full_df[0] = cupy.random.rand(len(full_df))
        full_df[1] = cupy.random.rand(len(full_df))
        full_df[2] = cupy.random.rand(len(full_df))
        return full_df

    def setUp(self):
        self.df = self.create_df()

    def test_months_start(self):
        log_return = self.df
        first_sample = log_return['sample_id'].min().item()
        all_dates = log_return[first_sample == log_return['sample_id']]['date']
        months_start = _get_month_start_pos(all_dates)
        print(type(months_start))

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
        window = 6
        log_return = self.df
        first_sample = log_return['sample_id'].min().item()
        all_dates = log_return[first_sample == log_return['sample_id']]['date']
        months_start = _get_month_start_pos(all_dates)
        log_return_ma = _get_log_return_matrix(total_samples, log_return)
        _, assets, timelen = log_return_ma.shape
        number_of_threads = 256
        num_months = len(months_start) - window
        number_of_blocks = num_months * total_samples
        means = cupy.zeros((total_samples, num_months, assets))
        cov = cupy.zeros((total_samples, num_months, assets, assets))
        distance = cupy.zeros(
            (total_samples, num_months, (assets - 1) * assets // 2))

        compute_cov[(number_of_blocks, ), (number_of_threads, ), 0,
                    256 * MAX_YEARS * 8](means, cov, distance, log_return_ma,
                                         months_start, num_months, assets,
                                         timelen, window)
        print('return shape', log_return_ma.shape)
        num = 0
        for sample in range(2):
            for num in range(num_months):
                truth = (
                    log_return_ma[sample, :, months_start[num]:months_start[
                        num + window]].mean(axis=1))
                compute = means[sample][num]
                self.assertTrue(cupy.allclose(compute, truth))

        for sample in range(2):
            for num in range(num_months):
                s = log_return_ma[sample, :, months_start[num]:months_start[
                    num + window]]
                truth = (cupy.cov(s, bias=True))
                compute = cov[sample][num]
                self.assertTrue(cupy.allclose(compute, truth))

        for sample in range(2):
            for num in range(num_months):
                cov_m = cov[sample][num]
                corr_m = cov_m.copy()
                for i in range(3):
                    for j in range(3):
                        corr_m[i, j] = corr_m[i, j] / \
                            math.sqrt(cov_m[i, i] * cov_m[j, j])
                dis = cupy.sqrt((1.0 - corr_m)/2.0)
                res = cupy.zeros_like(dis)
                for i in range(3):
                    for j in range(3):
                        res[i, j] = cupy.sqrt(
                            ((dis[i, :] - dis[j, :])**2).sum())
                truth = (squareform(res.get()))
                compute = distance[sample][num]
                self.assertTrue(cupy.allclose(compute, truth))
