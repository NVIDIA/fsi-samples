'''
'''
from setuptools import setup, find_packages

install_requires = []

setup(
    name='gquant',
    version='1.0',
    description='gquant - RAPIDS Financial Services Algorithms',
    author='NVIDIA Corporation',
    packages=find_packages(include=['gquant', 'gquant.*']),
    install_requires=install_requires,
    license="Apache",
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX :: Linux',
    ],
    entry_points={
        'console_scripts': ['gquant-flow=gquant.flow:main'],
    }
)
