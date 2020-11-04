import setuptools
from os import path


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setuptools.setup(

     name='pycrostate',

     version='0.0.1a',

     author="Victor Férat",

     author_email="victor.ferat@unige.ch",

     description="A simple open source Python package for EEGmicrostate segmentation",

     long_description=long_description,

     long_description_content_type="text/markdown",

     url=None,

     license="BSD-3-Clause",

     python_requires='>=3.6',

     install_requires=["mne", "numpy", "scipy", "joblib"],

     packages=setuptools.find_packages(exclude=['docs', 'tests']),

     classifiers=[

        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
 )