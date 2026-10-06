:orphan:

Installation
============

We require that you use Python ≥ 3.11 and MNE ≥ 1.8.

You may choose to install ``mne-connectivity`` `via pip <#Installation via pip>`_,
or conda.

Installation via Conda
----------------------

To install MNE-Connectivity using conda in a virtual environment,
simply run the following at the root of the repository:

.. code-block:: bash

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
