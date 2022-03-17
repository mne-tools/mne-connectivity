.. -*- mode: rst -*-

|GH|_ |Circle|_ |Azure|_ |Codecov|_ |PyPI|_ |conda-forge|_

.. |GH| image:: https://github.com/mne-tools/mne-connectivity/actions/workflows/unit_tests.yml/badge.svg
.. _GH: https://github.com/mne-tools/mne-connectivity/actions/workflows/unit_tests.yml

.. |Circle| image:: https://circleci.com/gh/mne-tools/mne-connectivity.svg?style=shield
.. _Circle: https://circleci.com/gh/mne-tools/mne-connectivity

.. |Azure| image:: https://dev.azure.com/mne-tools/mne-connectivity/_apis/build/status/mne-tools.mne-connectivity?branchName=main
.. _Azure: https://dev.azure.com/mne-tools/mne-connectivity/_build/latest?definitionId=1&branchName=main

.. |Codecov| image:: https://codecov.io/gh/mne-tools/mne-connectivity/branch/main/graph/badge.svg
.. _Codecov: https://codecov.io/gh/mne-tools/mne-connectivity

.. |conda-forge| image:: https://img.shields.io/conda/dn/conda-forge/mne-connectivity.svg?label=Conda%20downloads
.. _conda-forge: https://anaconda.org/conda-forge/mne-connectivity

.. |PyPI| image:: https://img.shields.io/pypi/dm/mne-connectivity.svg?label=PyPI%20downloads
.. _PyPI: https://pypi.org/project/mne-connectivity/

.. _MNE-Connectivity: https://mne.tools/mne-connectivity/dev/
.. _MNE-Python: https://mne.tools/stable
.. _MNE-Connectivity documentation: https://mne.tools/mne-connectivity/stable/index.html
.. _installation guide: https://mne.tools/mne-connectivity/dev/install/index.html
.. _pip: https://pip.pypa.io/en/stable/
.. _Frites: https://github.com/brainets/frites
.. _contributing guide: https://github.com/mne-tools/mne-connectivity/blob/main/CONTRIBUTING.md

MNE-Connectivity
================

`MNE-Connectivity`_ is an open-source Python package for connectivity and
related measures of MEG, EEG, or iEEG data built on top of the 
`MNE-Python`_ API. It includes modules for data input/output, visualization,
common connectivity analysis, and post-hoc statistics and processing.


.. target for :end-before: title-end-content

This project was initially ported over from mne-python starting v0.23, by Adam
Li as part of Google Summer of Code 2021. Subsequently v0.1 and v0.2 releases
were done as part of GSoC period. Future development will occur in subsequent
versions. 

Documentation
^^^^^^^^^^^^^

Stable `MNE-Connectivity documentation`_ is available online.

Installing MNE-Connectivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the latest stable version of MNE-Connectivity, you can use pip_ in a terminal:

.. code-block:: bash

    pip install -U mne-connectivity

For more complete instructions and more advanced installation methods (e.g. for
the latest development version), see the `installation guide`_.


Get the latest code
^^^^^^^^^^^^^^^^^^^

To install the latest version of the code using pip_ open a terminal and type:

.. code-block:: bash

    pip install -U https://github.com/mne-tools/mne-connectivity/archive/main.zip

To get the latest code using `git <https://git-scm.com/>`__, open a terminal and type:

.. code-block:: bash

    git clone https://github.com/mne-tools/mne-connectivity.git

Alternatively, you can also download a
`zip file of the latest development version <https://github.com/mne-tools/mne-connectivity/archive/main.zip>`__.


Contributing to MNE-Connectivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please see the documentation on the MNE-Connectivity homepage:

https://github.com/mne-tools/mne-connectivity/blob/main/CONTRIBUTING.md


Forum
^^^^^^

https://mne.discourse.group

A Note About Connectivity
^^^^^^^^^^^^^^^^^^^^^^^^^

In the neuroscience community as of 2021, the term "functional connectivity" can 
have many different meanings and comprises many different measures. Some of 
these measures are directed (i.e. try to map a statistical causal relationship between
brain regions), others are non-directed. Please note that the interpretation of your 
functional connectivity measure depends on the data and underlying
assumptions. 
For a taxonomy of functional connectivity measures and information on the 
interpretation of those measures, we refer to
`Bastos and Schoffelen`_.

In mne-connectivity, we do not claim that any of our measures imply causal
connectivity.

.. _Bastos and Schoffelen: https://pubmed.ncbi.nlm.nih.gov/26778976/