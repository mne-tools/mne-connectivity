.. -*- mode: rst -*-

|GH|_ |Circle|_ |Codecov|_ |PyPI|_

.. |GH| image:: https://github.com/mne-tools/mne-connectivity/actions/workflows/unit_tests.yml/badge.svg
.. _GH: https://github.com/mne-tools/mne-connectivity/actions/workflows/unit_tests.yml

.. |Circle| image:: https://circleci.com/gh/mne-tools/mne-connectivity.svg?style=shield
.. _Circle: https://circleci.com/gh/mne-tools/mne-connectivity

.. |Codecov| image:: https://codecov.io/gh/mne-tools/mne-connectivity/branch/main/graph/badge.svg
.. _Codecov: https://codecov.io/gh/mne-tools/mne-connectivity

.. |PyPI| image:: https://img.shields.io/pypi/dm/mne-connectivity.svg?label=PyPI%20downloads
.. _PyPI: https://pypi.org/project/mne-connectivity/

.. _MNE-Connectivity: https://mne.tools/mne-connectivity/dev/
.. _MNE-Python: https://mne.tools/stable
.. _MNE-Connectivity documentation: https://mne.tools/mne-connectivity/stable/index.html
.. _installation guide: https://mne.tools/mne-connectivity/dev/install/index.html
.. _pip: https://pip.pypa.io/en/stable/

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

    git clone git://github.com/mne-tools/mne-connectivity.git

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

In the neuroscience community as of 2021, the term "connectivity" can have many
different meanings. There is the common question of whether or not
"connectivity" detected via some measure actually implies a connection between
two brain regions. That is, is the connectivity causal? Even if one calls their
connectivity measure, "causal", it depends on the data and the underlying
assumptions. Some common interpretations of connectivity are in are in
`Schoffelen`_.

In mne-connectivity, we do not claim that any of our measures imply causal
connectivity.

.. _Schoffelen: https://pubmed.ncbi.nlm.nih.gov/26778976/