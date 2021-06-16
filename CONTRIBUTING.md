# Contributions

Contributions are welcome in the form of pull requests.

Once the implementation of a piece of functionality is considered to be bug
free and properly documented (both API docs and an example script),
it can be incorporated into the `main` branch.

To help developing `mne-connectivity`, you will need a few adjustments to your
installation as shown below.

## Running tests

### (Optional) Install development version of MNE-Python
If you want to run the tests with a development version of MNE-Python,
you can install it by running

    $ pip install -U https://github.com/mne-tools/mne-python/archive/main.zip

### Install development version of MNE-Connectivity
First, you should [fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) the `mne-connectivity` repository. Then, clone the fork and install it in
"editable" mode.

    $ git clone https://github.com/<your-GitHub-username>/mne-connectivity
    $ pip install -e ./mne-connectivity


### Install Python packages required to run tests
Install the following packages for testing purposes, plus all optonal MNE-connectivity
dependencies to ensure you will be able to run all tests.

    $ pip install -r requirements_testing.txt

### Invoke pytest
Now you can finally run the tests by running `pytest` in the
`mne-connectivity` directory.

    $ cd mne-connectivity
    $ pytest

## Building the documentation

The documentation can be built using sphinx. For that, please additionally
install the following:

    $ pip install -r requirements_doc.text

To build the documentation locally, one can run:

    $ cd doc/
    $ make html

or

    $ make html-noplot

if you don't want to run the examples to build the documentation. This will result in a faster build but produce no plots in the examples.
