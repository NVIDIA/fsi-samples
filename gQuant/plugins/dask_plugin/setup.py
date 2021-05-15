from setuptools import setup, find_packages

setup(
    name='greenflow_dask_plugin',
    version='0.0.1',
    packages=find_packages(include=['greenflow_dask_plugin']),
    install_requires=[
        "greenflow"
    ],
    entry_points={
        'greenflow.plugin': [
            'greenflow_dask_plugin = greenflow_dask_plugin',
        ],
    }
)
