#! /usr/bin/env python
"""A module for connectivity data analysis with MNE."""

import codecs
import os

from setuptools import find_packages, setup

# get the version from __init__.py
version = None
with open(os.path.join('mne_connectivity', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

DISTNAME = 'mne-connectivity'
DESCRIPTION = 'A module for connectivity data analysis with MNE.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Adam Li'
MAINTAINER_EMAIL = 'ali39@jhu.edu'
URL = 'https://github.com/mne-tools/mne-connectivity'
LICENSE = 'BSD-3'
DOWNLOAD_URL = 'https://github.com/mne-tools/mne-connectivity'
VERSION = version
INSTALL_REQUIRES = ['numpy', 'scipy', 'mne', 'xarray', 'netCDF4', 'pandas', 'tqdm']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9',
               ]
EXTRAS_REQUIRE = {
    'optional': [
        'scikit-learn',
        'tqdm',
        'vtk',
        'sip',
        'pyvista',
        'pyvistaqt',
        'pyqt5'],
    'tests': [
        'pytest',
        'pytest-cov',
        'flake8',
        'pydocstyle'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      python_requires='>=3.7',
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
