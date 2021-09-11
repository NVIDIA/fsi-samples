'''
'''
import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

install_requires = ['dask[distributed]', 'dask[dataframe]', 'configparser',
                    'cloudpickle', 'PyYaml',
                    'jsonpath_ng', 'ruamel.yaml', 'pandas']

setup(
    name='greenflow',
    version='1.0.5',
    description='greenflow - RAPIDS Financial Services Algorithms',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='NVIDIA Corporation',
    url='https://github.com/NVIDIA/fsi-samples/tree/main/greenflow',
    packages=find_packages(include=['greenflow', 'greenflow.*']),
    install_requires=install_requires,
    license="Apache",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX :: Linux',
    ],
    entry_points={
        'console_scripts': ['greenflow-flow=greenflow.flow:main'],
    }
)
