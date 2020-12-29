from setuptools import setup

setup(
    name='example_plugin',
    entry_points={
        'gquant.plugin': [
            'custom_nodes = example',
        ],
    }
)