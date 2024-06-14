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

import numpy as np

from coder.hyperspherical_coords import cartesian2hyperspherical, hyperspherical2cartesian, \
    cartesian2polar, identity_coordinate_conversion, cartesian2hyperspherical_matrix


def test_cartesian2hyperspherical(pos_vector, pos_spherical_vector, neg_vector, neg_spherical_vector):
    for i, (cartesian_vector, spherical_vector) in enumerate(zip(pos_vector, pos_spherical_vector)):
        out_spherical = cartesian2hyperspherical(cartesian_vector)
        result = np.allclose(out_spherical, spherical_vector)
        # print(cartesian_vector, '\t', out_spherical, '\t', spherical_vector)
        assert result, f'{i}, {result}, {cartesian_vector}, {spherical_vector}'

    for i, (cartesian_vector, spherical_vector) in enumerate(zip(neg_vector, neg_spherical_vector)):
        out_spherical = cartesian2hyperspherical(cartesian_vector)
        result = np.allclose(out_spherical, spherical_vector)
        # print(cartesian_vector, '\t', out_spherical, '\t', spherical_vector)
        assert result, f'{i}, {result}, {cartesian_vector}, {spherical_vector}'


def test_hyperspherical2cartesian(pos_vector, pos_spherical_vector, neg_vector, neg_spherical_vector):
    for i, (cartesian_vector, spherical_vector) in enumerate(zip(pos_vector, pos_spherical_vector)):
        out_cartesian = hyperspherical2cartesian(spherical_vector)
        result = np.allclose(out_cartesian, cartesian_vector)
        # print(cartesian_vector, '\t', out_cartesian, '\t', cartesian_vector)
        assert result, f'{i}, {result}, {cartesian_vector}, {spherical_vector}'

    for i, (cartesian_vector, spherical_vector) in enumerate(zip(neg_vector, neg_spherical_vector)):
        out_cartesian = hyperspherical2cartesian(spherical_vector)
        result = np.allclose(out_cartesian, cartesian_vector)
        # print(cartesian_vector, '\t', out_cartesian, '\t', cartesian_vector)
        assert result, f'{i}, {result}, {cartesian_vector}, {spherical_vector}'


def test_cartesian2polar(pos_polar, neg_polar):
    for i, (v, ex) in enumerate(pos_polar):
        out = cartesian2polar(v)
        result = np.allclose(out, ex)
        # print(v, '\t', out, '\t', ex)
        assert result, f'{i}, {result}, {v}, {ex}'

    for i, (v, ex) in enumerate(neg_polar):
        out = cartesian2polar(v)
        result = np.allclose(out, ex)
        # print(v, '\t', out, '\t', ex)
        assert result, f'{i}, {result}, {v}, {ex}'


def test_identity_coordinate_conversion(pos_vector, pos_spherical_vector, neg_vector, neg_spherical_vector):
    for spherical_vector in np.concatenate([pos_spherical_vector, neg_spherical_vector]):
        identity = identity_coordinate_conversion(spherical_vector, input_space='spherical')
        result = np.allclose(identity, spherical_vector)
        assert result, f'{spherical_vector}\t{identity}'

    for cartesian_vector in np.concatenate([pos_vector, neg_vector]):
        identity = identity_coordinate_conversion(cartesian_vector, input_space='cartesian')
        result = np.allclose(identity, cartesian_vector)
        assert result, f'{cartesian_vector}\t{identity}'

    for extra_cartesian_vector in [np.array([1, 2, 3, 4, 5, 6]),
                                   np.array([-1, 2, -3, 4, -5, 6]),
                                   np.array([1, -2, 3, -4, 5, -6]),
                                   np.array([0, 0, 0, 0, 0, 0]),
                                   np.array([1, 0, 0, 0, 0, 0]),
                                   np.array([0, -1, 0, 0, 0, 0]),
                                   np.array([0, -1, 0, -1, 0, 0]),
                                   ]:
        identity = identity_coordinate_conversion(extra_cartesian_vector, input_space='cartesian')
        result = np.allclose(identity, extra_cartesian_vector)
        assert result, f'{extra_cartesian_vector}\t{identity}'

    from tqdm import tqdm
    vector = np.arange(1000)
    for i in tqdm(range(1000)):
        identity = identity_coordinate_conversion(vector, input_space='cartesian')
        result = np.allclose(identity, vector)
        assert result, f'{vector}\t{identity}'

    matrix = np.ones(shape=(1000, 1000)).cumsum(axis=1)
    identity = identity_coordinate_conversion(matrix, input_space='cartesian')
    result = np.allclose(identity, matrix)
    assert result  # , f'{vector}\t{identity}'


def test_cartesian2hyperspherical_matrix():
    matrix = np.ones(shape=(1000, 1000)).cumsum(axis=1)
    spherical_array = cartesian2hyperspherical_matrix(matrix)
    # result = np.allclose(identity, vector)
    # assert result, f'{vector}\t{identity}'
    assert (spherical_array == spherical_array[0, :]).all()
