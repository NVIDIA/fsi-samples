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
python -m unittest tests/unit/test_bootstrap.py -v
'''
import unittest
import cupy
from greenflow_hrp_plugin.kernels import boot_strap


class TestBootstrap(unittest.TestCase):

    def setUp(self):
        pass

    def test_bootstrap(self):
        number_samples = 2
        block_size = 2
        number_of_threads = 256
        length, assets = (6, 2)
        ref = cupy.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],
                          [5.0, 6.0]])
        output = cupy.zeros((number_samples, assets, length))  # output results
        num_positions = (
            length - 2
        ) // block_size + 1
        # number of positions to sample to cover the whole seq length
        # sample starting position, exclusive
        sample_range = length - block_size
        print('price_len', length, 'sample range', sample_range)
        sample_positions = cupy.array([0, 1, 2, 3, 2, 1])
        number_of_blocks = len(sample_positions)
        boot_strap[(number_of_blocks,), (number_of_threads,)](
            output,
            ref.T,
            block_size,
            num_positions,
            sample_positions)
        truth0 = cupy.array([[0., 1., 2., 2., 3., 3.],
                             [0., 2., 3., 3., 4., 4.]])
        truth1 = cupy.array([[0., 4., 5., 3., 4., 2.],
                             [0., 5., 6., 4., 5., 3.]])
        self.assertTrue(cupy.allclose(truth0, output[0]))
        self.assertTrue(cupy.allclose(truth1, output[1]))
        print(output)
