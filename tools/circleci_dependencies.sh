#!/bin/bash -ef

echo "Installing setuptools and sphinx"
python -m pip install --progress-bar off --upgrade "pip!=20.3.0" setuptools wheel
python -m pip install --upgrade --progress-bar off sphinx

echo "Installing doc build dependencies"
python -m pip uninstall -y pydata-sphinx-theme
python -m pip install --upgrade --progress-bar off --only-binary matplotlib PyQt6 -r requirements.txt -r requirements_testing.txt -r requirements_doc.txt
python -m pip install --upgrade --progress-bar off https://github.com/mne-tools/mne-python/zipball/main
python -m pip install --progress-bar off https://github.com/sphinx-gallery/sphinx-gallery/zipball/master https://github.com/pyvista/pyvista/zipball/main https://github.com/pyvista/pyvistaqt/zipball/main
python -m pip install -e .
