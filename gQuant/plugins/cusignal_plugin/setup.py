'''
Greenflow Cusignal Plugin
'''
from setuptools import setup, find_packages

setup(
    name='greenflow_cusignal_plugin',
    version='1.0',
    description='greenflow cusignal plugin - RAPIDS Cusignal Nodes for Greenflow',  # noqa: E501
    install_requires=["greenflow", "cusignal"],
    packages=find_packages(include=['greenflow_cusignal_plugin',
                                    'greenflow_cusignal_plugin.*']),
    entry_points={
        'greenflow.plugin': [
            'greenflow_cusignal_plugin = greenflow_cusignal_plugin',
            'greenflow_cusignal_plugin.convolution = greenflow_cusignal_plugin.convolution',  # noqa: E501
            'greenflow_cusignal_plugin.filtering = greenflow_cusignal_plugin.filtering',  # noqa: E501
            'greenflow_cusignal_plugin.gensig = greenflow_cusignal_plugin.gensig',  # noqa: E501
            'greenflow_cusignal_plugin.spectral_analysis = greenflow_cusignal_plugin.spectral_analysis',  # noqa: E501
            'greenflow_cusignal_plugin.windows = greenflow_cusignal_plugin.windows'  # noqa: E501
        ],
    }
)
