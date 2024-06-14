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

"""
Used for FunctionTransformer. New functions/inverse functions can be added here. Do not use lambda functions else the
vocab won't be pickleable.
"""

import numpy as np


def inverse_log1p(x):
    return np.exp(x) - 1
