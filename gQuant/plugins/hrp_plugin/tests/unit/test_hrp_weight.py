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
python -m unittest tests/unit/test_hrp_weight.py -v
'''
import unittest
import cupy
from greenflow_hrp_plugin.kernels import HRP_weights
import numpy as np
import pandas as pd


def compute_HRP_weights(covariances, res_order):
    weights = pd.Series(1, index=res_order)
    clustered_alphas = [res_order]

    while len(clustered_alphas) > 0:
        clustered_alphas = [cluster[start:end] for cluster in clustered_alphas
                            for start, end in ((0, len(cluster) // 2),
                                               (len(cluster) // 2, len(cluster)))
                            if len(cluster) > 1]
        for subcluster in range(0, len(clustered_alphas), 2):
            left_cluster = clustered_alphas[subcluster]
            right_cluster = clustered_alphas[subcluster + 1]

            left_subcovar = covariances[left_cluster, :][:, left_cluster]
            inv_diag = 1 / cupy.diag(left_subcovar)
            parity_w = inv_diag * (1 / cupy.sum(inv_diag))
            left_cluster_var = cupy.dot(
                parity_w, cupy.dot(left_subcovar, parity_w))

            right_subcovar = covariances[right_cluster, :][:, right_cluster]
            inv_diag = 1 / cupy.diag(right_subcovar)
            parity_w = inv_diag * (1 / cupy.sum(inv_diag))
            right_cluster_var = cupy.dot(
                parity_w, cupy.dot(right_subcovar, parity_w))

            alloc_factor = 1 - left_cluster_var / \
                (left_cluster_var + right_cluster_var)

            weights[left_cluster] *= alloc_factor.item()
            weights[right_cluster] *= 1 - alloc_factor.item()
            
    return weights


class TestHRPWeight(unittest.TestCase):

    def setUp(self):
        self.assets = 10
        self.samples = 5
        self.numbers = 30
        seq = 100
        cupy.random.seed(10)
        self.cov_matrix = cupy.zeros(
            (self.samples, self.numbers, self.assets, self.assets))
        self.order_matrix = cupy.random.randint(
            0, self.assets, (self.samples, self.numbers, self.assets))
        for i in range(self.samples):
            for j in range(self.numbers):
                cov = cupy.cov(cupy.random.rand(self.assets, seq))
                self.cov_matrix[i, j] = cov
                order = cupy.arange(self.assets)
                cupy.random.shuffle(order)
                self.order_matrix[i, j] = order

    def test_order(self):
        num_months = self.numbers
        total_samples = self.samples
        assets = self.assets

        number_of_threads = 1

        number_of_blocks = num_months * total_samples

        weights = cupy.ones((total_samples, num_months, assets))

        HRP_weights[(number_of_blocks,), (number_of_threads,)](
            weights,
            self.cov_matrix,
            self.order_matrix,
            assets,
            num_months)
        for i in range(self.samples):
            for j in range(self.numbers):
                cpu_weights = compute_HRP_weights(
                    self.cov_matrix[i][j], self.order_matrix[i][j].get())
                cpu_weights = cpu_weights[range(self.assets)].values
                self.assertTrue(np.allclose(cpu_weights, weights[i][j].get()))
