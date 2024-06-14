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
from deprecated import deprecated

from utils.logger import create_logger


logger = create_logger(__name__)

@deprecated('not the same output as hyperspherical. This function outputs (r,theta, phi) vs (r, phi, theta)'
            'and does not account for keeping angles positive.'
            'To compare use: cartesian2spherical(np.array([x,y,z])), cartesian2hyperspherical(np.array([z,x,y]))')
def cartesian2spherical(vector: np.array, degrees=True):
    """
    rho = L2 norm (vector)
    phi = arccos(z/rho)
    theta = arcsin( y / (rho * sin(phi))

    or equivalently:
    x = rho * sin(phi) * cos(theta)
    y = rho * sin(phi) * sin(theta)
    z = rho * cos(phi)

    :param vector:
    :param degrees:
    :return:
    """
    assert len(vector) == 3
    x, y, z = vector
    rho = np.linalg.norm(vector)
    phi = np.arccos(z/rho)
    theta = np.arcsin(y / (rho * np.sin(phi)))
    if degrees:
        phi, theta = np.rad2deg([phi, theta])
    return rho, theta, phi


def cartesian2polar(vector, degrees=True):
    """
    branch cut is along 0/2pi (0,360) instead of pi/-pi (180,-180) this is so encoding scheme doesn't require
    tokenizing the parity
    :param vector: 2d vector
    :param degrees: (bool) return theta as degrees
    :return: r, theta
    """

    assert len(vector) == 2
    r = np.linalg.norm(vector)
    theta = np.arctan2(vector[1], vector[0])

    if theta < 0:
        theta += 2 * np.pi

    if degrees:
        theta = np.degrees(theta)
        logger.debug(theta)
        assert 0 <= theta <= 360, 'theta calculated wrong'

    return r, theta


def polar2cartesian(vector) -> tuple:
    """
    convert polar vector in format (radius, degrees) -> (x,y)

    :param vector:
    :return:
    """
    assert len(vector) == 2
    r, theta = vector
    return r * np.cos(np.deg2rad(theta)), r * np.sin(np.deg2rad(theta))


def cumulative_norm(vector: np.array) -> np.array:
    """
    Calculates the cumulative decreasing L2 norm of a vector i.e.
    ```python
    for idx in range(len(vector)):
        norm = np.linalg.norm(vector[idx:])

    i.e. np.linalg.norm(vector[0:]), np.linalg.norm(vector[1:]), etc.
    ```

    This can be vectorized (pun intended) by:
        1. squaring each element of the array, and reversing it-> z^2, y^2, x^2
        2. doing a cumulative sum: (z^2, z^2 + y^2, z^2 + y^2 + x^2)
        3. and reversing it: (z^2 + y^2 + x^2, z^2 + y^2,  z^2)
        4. and finally taking the square root of each element

    :param vector:
    :return:
    """
    if len(vector.shape) == 1:
        return np.sqrt(np.cumsum((vector**2)[::-1])[::-1])
    elif len(vector.shape) == 2:
        # add support for matrix input
        return np.sqrt(np.cumsum((vector**2)[:, ::-1], axis=1)[:, ::-1])


def cartesian2hyperspherical_matrix(matrix: np.array, degrees=True) -> np.array:
    """
    matrix analog of cartesian2hyperspherical. Assumes each row is a vector
    :param matrix:
    :param degrees:
    :return:
    """
    spherical_array = np.empty(shape=matrix.shape)
    spherical_array[:, 0] = np.linalg.norm(matrix, axis=1)
    norm_vectors = cumulative_norm(matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        arccos_vectors = np.arccos(matrix/norm_vectors)
        spherical_array[:, 1:-1] = arccos_vectors[:, :matrix.shape[1] - 2]

        # spherical_array[:, -1] = np.where(matrix[:, -1] >= 0, arccos_vectors[-2], 2*np.pi - arccos_vectors[-2])
        gt_mask = matrix[:, -1] >= 0
        lt_mask = matrix[:, -1] < 0
        spherical_array[gt_mask, -1] = arccos_vectors[gt_mask, -2]
        spherical_array[lt_mask, -1] = 2*np.pi - arccos_vectors[lt_mask, -2]

        # if vector[-1] >= 0:
        #     # theta = np.arccos(vector[-2] / np.sqrt(vector[-1] ** 2 + vector[-2] ** 2))
        #     spherical_vector[-1] = arccos_vectors[-2]
        # else:
        #     # theta = 2*np.pi - np.arccos(vector[-2] / np.sqrt(vector[-1] ** 2 + vector[-2] ** 2))
        #     spherical_vector[-1] = 2*np.pi - arccos_vectors[-2]
        if degrees:
            spherical_array[:, 1:] = np.rad2deg(spherical_array[:, 1:])
        return np.nan_to_num(spherical_array)


def cartesian2hyperspherical(vector: np.array, degrees=True) -> np.array:
    """
    Start from +x-axis, rotate to y axis (shortest path), then rotate from +y axis along +z axis
    Alternatively, start from +y axis and rotate, then rotate from +x axis to wherever you landed up from +y rotation

    x,y,z = 0, 0, -1
    cartesian2hyperspherical(np.array([z,x,y])) --> array([r, phi, theta) # theta(0,360), phi (0,180)

    arr = np.array([0,-1,0]) -> array([  1.,  90., 180.])
    arr = np.array([-1,-1,0]) -> array([ 1.41421356, 135., 180.])  # in this case, 135 degrees is rotation to -y axis

    arr = np.array([0,0,0]) -> array([ 0., nan, nan])  # Zero vector see notes below
    arr = np.array([1,0,0,0]) -> .array([ 1.,  0., nan, nan])

    Notes:

    if x_i !=0, i < n and x_i+1 .. x_n = 0,
    then phi_i = 0 if x_i > 0 and phi_i=180 if x_i<0

    the transform is not unique if all x_i ... x_n == 0 ( i.e. calculating arccos( 0/0) gives NaN, so we assign
    phi_i .. phi_n to be zero.


    :param vector:
    :param degrees: (bool)
    :return: r , phi_i's. phi_n is range 0,360, the ones before phi_i, i<n are range 0,180 degrees
    """
    spherical_vector = np.empty(shape=vector.shape[0])
    # r = np.linalg.norm(vector)
    spherical_vector[0] = np.linalg.norm(vector)

    norm_vectors = cumulative_norm(vector)
    with np.errstate(divide='ignore', invalid='ignore'):  # ignore divide by zero warning in arccos calculation
        arccos_vectors = np.arccos(vector/norm_vectors)
        spherical_vector[1:-1] = arccos_vectors[:len(vector) - 2]
        if vector[-1] >= 0:
            # theta = np.arccos(vector[-2] / np.sqrt(vector[-1] ** 2 + vector[-2] ** 2))
            spherical_vector[-1] = arccos_vectors[-2]
        else:
            # theta = 2*np.pi - np.arccos(vector[-2] / np.sqrt(vector[-1] ** 2 + vector[-2] ** 2))
            spherical_vector[-1] = 2*np.pi - arccos_vectors[-2]
        if degrees:
            spherical_vector[1:] = np.rad2deg(spherical_vector[1:])
        return np.nan_to_num(spherical_vector)


@deprecated(reason='please use cartesian2hyperspherical instead due to slight differences in implementation')
def cartesian2hyperspherical_v2(vector: np.array, degrees=True) -> np.array:
    """
    arr = np.array([0,-1,0]) -> array([ 1., 90.,  0.])
    arr = np.array([-1,-1,0]) -> array([ 1.41421356, 135., 0.])
    arr = np.array([0,0,0]) -> array([ 0., 0, 0])  # see notes below
    arr = np.array([1,0,0,0]) -> array([1., 0., 0., 0.])
    :param vector:
    :param degrees:
    :return:
    """
    r = np.linalg.norm(vector)
    phis = []
    for idx in range(len(vector) - 2):
        # phis.append(np.arccos(vector[idx]/np.linalg.norm(vector[idx:])))
        phis.append(np.arctan2(np.linalg.norm(vector[idx+1:]), vector[idx]))
    theta = 2 * np.arctan2(vector[-1], (vector[-2] + np.linalg.norm(vector[-2:])))

    # if vector[-1] >= 0:
    #     theta = np.arccos(vector[-2] / np.sqrt(vector[-1] ** 2 + vector[-2] ** 2))
    # else:
    #     theta = 2*np.pi - np.arccos(vector[-2] / np.sqrt(vector[-1] ** 2 + vector[-2] ** 2))
    if degrees:
        return np.concatenate([np.array([r], dtype=np.float),  np.rad2deg(np.array(phis + [theta], dtype=np.float))])
    return np.array([r] + phis + [theta], dtype=np.float64)


def hyperspherical2cartesian(vector, degrees=True):
    # assume radians
    r, *phis = vector
    if degrees:
        phis = np.deg2rad(phis)
    cum_prod_sin_phis = np.cumprod(np.sin(phis))
    cos_phis = np.cos(phis)

    cartesian_vector = np.empty(vector.shape[0])
    cartesian_vector[0] = r * cos_phis[0]
    cartesian_vector[1:-1] = r * cum_prod_sin_phis[:-1] * cos_phis[1:]
    cartesian_vector[-1] = r * cum_prod_sin_phis[-1]
    return cartesian_vector


def hyperspherical2cartesian_matrix(spherical_matrix, degrees=True):
    # assume radians
    r, phis = spherical_matrix[:, 0], spherical_matrix[:, 1:]
    if degrees:
        phis = np.deg2rad(phis)
    cum_prod_sin_phis = np.cumprod(np.sin(phis), axis=1)
    cos_phis = np.cos(phis)

    cartesian_matrix = np.empty(shape=spherical_matrix.shape)
    cartesian_matrix[:, 0] = r * cos_phis[:, 0]
    cartesian_matrix[:, 1:-1] = r.reshape(-1, 1) * cum_prod_sin_phis[:, :-1] * cos_phis[:, 1:]
    cartesian_matrix[:, -1] = r * cum_prod_sin_phis[:, -1]
    return cartesian_matrix


def identity_coordinate_conversion(array, input_space='cartesian', degrees=True):
    if input_space == 'cartesian' and len(array.shape) == 1:
        return hyperspherical2cartesian(cartesian2hyperspherical(array, degrees))
    elif input_space == 'spherical' and len(array.shape) == 1:
        return cartesian2hyperspherical(hyperspherical2cartesian(array, degrees))
    elif input_space == 'cartesian' and len(array.shape) == 2:
        return hyperspherical2cartesian_matrix(cartesian2hyperspherical_matrix(array, degrees))
    elif input_space == 'spherical' and len(array.shape) == 2:
        return cartesian2hyperspherical_matrix(hyperspherical2cartesian_matrix(array, degrees))
    else:
        raise ValueError('input space must be "cartesian" or "spherical" and array shape length of 1 or 2')


@deprecated(reason='Not needed for cartesian2hyperspherical since the return signature has changed')
def wrap(out):
    l = [out[0]] + [out[1]] + out[2]
    return l
