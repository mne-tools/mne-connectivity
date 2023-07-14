:orphan:

Installation
============

Dependencies
------------

* ``mne`` (>=0.23)
* ``numpy`` (>=1.14)
* ``scipy`` (>=1.5.0 for certain operations with EEGLAB data)
* ``xarray`` (>=0.18)
* ``joblib`` (>=1.0.0)
* ``scikit-learn`` (>=0.24.2)
* ``pandas`` (>=0.23.4, optional, for generating event statistics)
* ``matplotlib`` (optional, for using the interactive data inspector)

We require that you use Python 3.7 or higher.
You may choose to install ``mne-connectivity`` `via pip <#Installation via pip>`_,
or conda.

Installation via Conda
----------------------

To install MNE-Connectivity using conda in a virtual environment,
simply run the following at the root of the repository:

.. code-block:: bash

   # with python>=3.8 at least
   conda create -n mne
   conda activate mne
   conda install -c conda-forge mne-connectivity


Installation via Pip
--------------------

To install MNE-Connectivity including all dependencies required to use all features,
simply run the following at the root of the repository:

.. code-block:: bash

    python -m venv .venv
    pip install -U mne-connectivity

If you want to install a snapshot of the current development version, run:

.. code-block:: bash

   pip install --user -U https://api.github.com/repos/mne-tools/mne-connectivity/zipball/main

To check if everything worked fine, the following command should not give any
error messages:

.. code-block:: bash

   python -c 'import mne_connectivity'

MNE-Connectivity works best with the latest stable release of MNE-Python. To ensure
MNE-Python is up-to-date, run:

.. code-block:: bash

   pip install --user -U mne
