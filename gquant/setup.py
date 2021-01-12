'''
'''
import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

install_requires = ['dask', 'configparser', 'cloudpickle', 'PyYaml',
                    'jsonpath_ng']

setup(
    name='gquant',
    version='1.0.1',
    description='gquant - RAPIDS Financial Services Algorithms',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='NVIDIA Corporation',
    url='https://github.com/rapidsai/gQuant',
    packages=find_packages(include=['gquant', 'gquant.*']),
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
        'console_scripts': ['gquant-flow=gquant.flow:main'],
    }
)
