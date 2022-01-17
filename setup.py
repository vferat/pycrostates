#!/usr/bin/env python

from pathlib import Path
from setuptools import setup, find_packages


# Version
version = None
with open(Path(__file__).parent/'pycrostates'/'_version.py', 'r') as file:
    for line in file:
        line = line.strip()
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version.')


# Descriptions
short_description = "A simple open source Python package for EEGmicrostate segmentation"
long_description_file = Path(__file__).parent / 'README.md'
with open(long_description_file, 'r') as file:
    long_description = file.read()
if long_description_file.suffix == '.md':
    long_description_content_type='text/markdown'
elif long_description_file.suffix == '.rst':
    long_description_content_type='text/x-rst'
else:
    long_description_content_type='text/plain'


# Dependencies
def get_requirements(path):
    """Get mandatory dependencies from file."""
    install_requires = list()
    with open(path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            req = line.strip()
            if len(line) == 0:
                continue
            install_requires.append(req)

    return install_requires

install_requires = get_requirements('requirements.txt')


setup(
     name='pycrostates',
     version='0.1.0',
     author="Victor Férat",
     author_email="victor.ferat@unige.ch",
     description="A simple open source Python package for EEGmicrostate segmentation",
     long_description=long_description,
     long_description_content_type=long_description_content_type,
     url=None,
     license="BSD-3-Clause",
     platforms='any',
     python_requires='>=3.6',
     install_requires=["mne", "numpy", "scipy", "joblib"],
     packages=find_packages(exclude=['docs', 'tests']),
     classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    # Include other files
    include_package_data = True
 )

