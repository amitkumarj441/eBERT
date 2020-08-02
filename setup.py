#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'eBERT', '_version.py'), encoding='utf-8') as f:
    exec(f.read())

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='eBERT',
    version=__version__,
    license='GPLv3',
    description='All-in-One PyTorch-based BERT Module',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Amit Kumar Jaiswal',
    maintainer='Amit Kumar Jaiswal',
    maintainer_email='amitkumarj441@gmail.com',
    url='http://eBERT.readthedocs.io',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'jieba',
        'torch',
    ],
    package_dir={'eBERT':'eBERT'},

    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Machine Learning/NLP',

        #'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    project_urls={
        "Bug Tracker": "https://github.com/amitkumarj441/eBERT/issues",
        "Source Code": "https://github.com/amitkumarj441/eBERT",
    },

    python_requires='>=3.5',
)
