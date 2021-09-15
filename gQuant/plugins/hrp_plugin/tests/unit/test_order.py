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
python -m unittest tests/unit/test_order.py -v
'''
import unittest
import cupy
from greenflow_hrp_plugin.kernels import single_linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
import numpy as np


def seriation(Z, N, cur_index):
    """Returns the order implied by a hierarchical tree (dendrogram).
    
       :param Z: A hierarchical tree (dendrogram).
       :param N: The number of points given to the clustering process.
       :param cur_index: The position in the tree for the recursive traversal.
       
       :return: The order implied by the hierarchical tree Z.
    """
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))


class TestOrder(unittest.TestCase):

    def setUp(self):
        self.assets = 10
        self.samples = 5
        self.numbers = 30
        seq = 100
        self.distance = cupy.zeros(
            (self.samples, self.numbers, self.assets * (self.assets-1) // 2))
        cupy.random.seed(10)
        for i in range(self.samples):
            for j in range(self.numbers):
                cov = cupy.cov(cupy.random.rand(self.assets, seq))
                dia = cupy.diag(cov)
                corr = cov / cupy.sqrt(cupy.outer(dia, dia))
                dist = (1.0 - corr) / 2.0
                self.distance[i, j] = cupy.array(squareform(dist.get()))

    def test_order(self):
        num_months = self.numbers
        total_samples = self.samples
        assets = self.assets

        number_of_threads = 1 
        number_of_blocks = num_months * total_samples
        output = cupy.zeros((total_samples, num_months, assets-1, 3))
        orders = cupy.zeros(
            (total_samples, num_months, assets), dtype=cupy.int64)
        single_linkage[(number_of_blocks,), (number_of_threads,)](
            output,
            orders,
            self.distance,
            num_months, assets)

        for i in range(self.samples):
            for j in range(self.numbers):
                gpu_order = orders[0][0]
                gpu_linkage = output[0][0]
                cpu_linkage = linkage(self.distance[0][0].get())
                cpu_order = seriation(cpu_linkage, assets, assets*2 - 2)
                self.assertTrue(np.allclose(gpu_order.get(), cpu_order))
                self.assertTrue(np.allclose(
                    gpu_linkage.get(), cpu_linkage[:, :-1]))
