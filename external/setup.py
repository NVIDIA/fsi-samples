from setuptools import setup, find_packages

setup(
    name='example_plugin',
    packages=find_packages(include=['example']),
    entry_points={
        'gquant.plugin': [
            'custom_nodes = example',
        ],
    }
)
