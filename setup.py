#! /usr/bin/env python
"""A module for connectivity data analysis with MNE."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('mne_connectivity', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'mne-connectivity'
DESCRIPTION = 'A module for connectivity data analysis with MNE.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Adam Li'
MAINTAINER_EMAIL = 'ali39@jhu.edu'
URL = 'https://github.com/mne-tools/mne-connectivity'
LICENSE = 'BSD-3'
DOWNLOAD_URL = 'https://github.com/mne-tools/mne-connectivity'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'scipy', 'mne']
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
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9',
               ]
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
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
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
