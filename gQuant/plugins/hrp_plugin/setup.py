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

from setuptools import setup, find_packages

setup(
    name='greenflow_hrp_plugin',
    install_requires=[
        "matplotlib", "shap"
        ],
    packages=find_packages(include=['greenflow_hrp_plugin']),
    entry_points={
        'greenflow.plugin': [
            'investment_nodes = greenflow_hrp_plugin',
        ],
    }
)
