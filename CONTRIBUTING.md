# Contributions

Contributions are welcome in the form of pull requests. We heavily rely on the
contribution guides of [MNE-Python](https://mne.tools/stable/install/contributing.html)

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
Install the following packages for testing purposes, plus all optional MNE-connectivity
dependencies to ensure you will be able to run all tests.

    $ pip install .[test]

### Invoke pytest
Now you can finally run the tests by running `pytest` in the
`mne-connectivity` directory.

    $ cd mne-connectivity
    $ pytest

## Building the documentation

The documentation can be built using sphinx. For that, please additionally
install the following:

    $ pip install .[doc]

To build the documentation locally, one can run:

    $ cd doc/
    $ make html

or

    $ make html-noplot

if you don't want to run the examples to build the documentation. This will result in a faster build but produce no plots in the examples.

### Issues with Memory Usage

All documentation examples are built on a CI pipeline that occurs online for free. For example, our docs are built with circleCI perhaps. This limits the ability for us to run large data examples that have a lot of RAM usage. For this reason, many times we crop, downsample, or limit the analysis in some way to reduce RAM usage.

Some good tools for profiling memory are ``mprof``. For example, one can memory profile a specific example, such as:

    mprof run examples/connectivity_classes.py

Then one could plot the memory usage:

    mprof plot


# Online documentation

MNE-Connectivity's online documentation is deployed from the `gh-pages` branch. Here, the documentation for different versions is stored in different folders:

- `stable/`: documentation for the latest stable release.

- `dev/`: documentation for the latest development version (i.e., the `main` branch).

- `X.Y/`: documentation for the latest micro release of each major-minor version (e.g.: `v0.8/` for `v0.8.1`; `v0.7/` for `v0.7.0`).

When changes are made to the `main` or `maint/X.Y` branches, the documentation is automatically built and transferred to the `gh-pages` branch by [CircleCI](https://app.circleci.com/pipelines/github/mne-tools/mne-connectivity) (`.circleci/config.yml`). The `build_docs` job generates the documentation, and the `deploy_docs` job transfers it to the `gh-pages` branch. The `deploy_docs` job determines which `gh-pages` folder to transfer the documentation to based on the branch that the changes were made to:

- If the changes were made to the `main` branch, the documentation is transferred to the `dev/` folder.

- If the changes were made to a `maint/X.Y` branch, the documentation is transferred to the `stable/` folder.

This means that changes should not be pushed to `maint/X.Y` branches which **are not** the current stable version.

The actual deployment of the documentation from `gh-pages` to the online documentation site is performed by the [`pages-build-deployment` action on GitHub](https://github.com/mne-tools/mne-connectivity/actions/workflows/pages/pages-build-deployment).


# Making a Release

If the procedure is followed correctly, there is no need to set any version information manually. Rather, [`setuptools_scm`](https://setuptools-scm.readthedocs.io/en/latest/) will dynamically determine the version number based on the git tags.

## Releasing a major-minor version

1. In the `gh-pages` branch, create a commit with the documentation from `stable/` copied to a new folder named with the major-minor version number of the current version (e.g., `0.7/`).

2. Create a pull request to the `main` branch with the following changes:

    a. Update the version information for the online documentation in `doc/_static/versions.json`.

    b. Create the changelog file for the new version based on the entries in `doc/changes/dev/`:
    
       towncrier --version X.Y.Z
    
    where `X.Y.Z` is the new major-minor-micro version number (e.g., `0.8.0`). The current date will be added to the file by default, but you can specify a different date with the `--date` parameter if needed.
    
    The changelog entries will be written to `doc/changes/dev.rst`. Move the **additions** to that file to a new file `doc/changes/vX.Y.rst`. There are existing contents in `doc/changes/dev.rst` that should not be copied over to the new file. After the additions have been moved, the diff should show no changes to `doc/changes/dev.rst`.

    c. Run the `tools/generate_citation.py` script with the new major-minor-micro version number as an argument (e.g., `0.8.0`) to update the information in `CITATION.cff` (and in turn the package citation in `doc/references.bib`). Note, this will use the current date for the release date field.

3. With the pull request merged, create a release tag for the new major-minor-micro version number (e.g., `v0.8.0`) on the `main` branch, and publish the release on GitHub.

4. Create a new maintenance branch named with the major-minor version number of the new version (e.g., `maint/0.8`). Creating the maintenance branch **after** the release tag is important in ensuring that the documentation built from the maintenance branch will have the correct version number set by `setuptools_scm`.

5. Trusted publishing action (`.github/workflows/release.yml`) will automatically add the new release to PyPI, which will in turn be picked up by the [conda-forge feedstock](https://github.com/conda-forge/mne-connectivity-feedstock).


## Releasing a micro version

For new releases that only bump the micro version, a slightly different workflow should be followed:

- In step 1, there is no need to create a new folder in the `gh-pages` branch, as the new version's documentation will just overwrite the old documentation in the `stable/` folder.

- In step 2, the same steps should be followed, except that that pull request should be made to the maintenance branch (e.g., `maint/0.8`) instead of the `main` branch. Also, the contents of the `towncrier`-generated changelog file should be added to the top of the existing changelog file for the current major-minor version (e.g., for minor version `0.8.1`, the contents should be added to the top of `doc/changes/v0.8.rst`).

- In step 3, the release tag should be created on the maintenance branch (e.g., `maint/0.8`) instead of the `main` branch.

- In step 4, we do not create a new maintenance branch. However, building the documentation from the updated maintenance branch will already have started when the pull request was merged in step 2, but before the release tag was created in step 3. This means that the documentation will have the wrong version number set by `setuptools_scm`. To fix this, we re-trigger the [documentation build job on CircleCI](https://app.circleci.com/pipelines/github/mne-tools/mne-connectivity) for the maintenance branch (e.g., `maint/0.8`) after step 3 is completed.
