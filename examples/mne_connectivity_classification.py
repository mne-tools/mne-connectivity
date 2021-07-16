"""
.. _ex-var-classification:

===============================================
Classify sleep stages by computing connectivity
===============================================

Compute a connectivity envelope correlation model using EEG data
and then apply a Random Forest classifier to predict sleep stage.

In this example, we will demonstrate how to use scikit-learn
and mne-connectivity to classify what stage of sleep
a subject is. This tutorial is inspired by MNE-Python's
`Sleep stage classification tutorial
<https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html>`_.
"""

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne_connectivity import envelope_correlation, degree


##############################################################################
# Load the data
# -------------
# Here, we first download the data from two subjects,
# and then set annotations and channel mappings for the
# available electrode data.
data_path = mne.datasets.sample.data_path()

ALICE, BOB = 0, 1

[alice_files, bob_files] = fetch_data(subjects=[ALICE, BOB], recording=[1])

mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'resp',
           'EMG submental': 'emg',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

raw_train = mne.io.read_raw_edf(alice_files[0])
annot_train = mne.read_annotations(alice_files[1])

raw_train.set_annotations(annot_train, emit_warning=False)
raw_train.set_channel_types(mapping)

# Now we band-pass filter our data.
raw_train.load_data()
raw_train.filter(14, 30)

# plot some data
# scalings were chosen manually to allow for simultaneous visualization of
# different channel types in this specific dataset
# raw_train.plot(start=60, duration=60,
#                scalings=dict(eeg=1e-4, resp=1e3, eog=1e-4, emg=1e-7,
#                              misc=1e-1))

##############################################################################
# Load the annotations
# --------------------
# Now, let us set up the annotations in the dataset
# for the sleep stages we are interested in classifying.
annotation_desc_2_event_id = {'Sleep stage W': 1,
                              'Sleep stage 1': 2,
                              'Sleep stage 2': 3,
                              'Sleep stage 3': 4,
                              'Sleep stage 4': 4,
                              'Sleep stage R': 5}

# keep last 30-min wake events before sleep and first 30-min wake events after
# sleep and redefine annotations on raw data
annot_train.crop(annot_train[1]['onset'] - 30 * 60,
                 annot_train[-2]['onset'] + 30 * 60)
raw_train.set_annotations(annot_train, emit_warning=False)

events_train, _ = mne.events_from_annotations(
    raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)

# create a new event_id that unifies stages 3 and 4
event_id = {'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3/4': 4,
            'Sleep stage R': 5}

# plot events
# fig = mne.viz.plot_events(events_train, event_id=event_id,
#                           sfreq=raw_train.info['sfreq'],
#                           first_samp=events_train[0, 0])

# keep the color-code for further plotting
stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

##############################################################################
# Create Epochs Using MNE-Python
# ------------------------------
# Now that we have setup our dataset, we will create an Epochs class
# that comprises of the different sleep stage events with 30 seconds
# of data in each Epoch.

tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included

epochs_train = mne.Epochs(raw=raw_train, events=events_train,
                          event_id=event_id, tmin=0., tmax=tmax, baseline=None)
times = epochs_train.times
ch_names = raw_train.ch_names

epochs_train.drop_bad()

print(epochs_train)

##############################################################################
# Applying the same steps to the test data from Bob
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
raw_test = mne.io.read_raw_edf(bob_files[0])
annot_test = mne.read_annotations(bob_files[1])
annot_test.crop(annot_test[1]['onset'] - 30 * 60,
                annot_test[-2]['onset'] + 30 * 60)
raw_test.set_annotations(annot_test, emit_warning=False)
raw_test.set_channel_types(mapping)

# Now we band-pass filter our data.
raw_test.load_data()
raw_test.filter(14, 30)

events_test, _ = mne.events_from_annotations(
    raw_test, event_id=annotation_desc_2_event_id, chunk_duration=30.)
epochs_test = mne.Epochs(raw=raw_test, events=events_test, event_id=event_id,
                         tmin=0., tmax=tmax, baseline=None)

epochs_test.drop_bad()

print(epochs_test)

##############################################################################
# Compute classification model
# ----------------------------
# Here, we use :func:`mne_connectivity.envelope_correlation` and a
# ``RandomForestClassifier`` trained
# on the envelope correlation values to classify sleep stage. We leverage
# scikit-learn's ``FunctionTransformer`` to create a ``Pipeline`` object
# for classification.

kw_args = dict(names=ch_names,)


def apply_degree_across_epochs(conn, **kwargs):
    degrees = []
    epoch_conn = conn.get_data(output='dense')
    for epoch_idx in range(conn.n_epochs):
        conn_data = epoch_conn[epoch_idx, ...].squeeze()
        degrees.append(degree(conn_data, **kwargs))
    return np.array(degrees)

# create sklearn pipeline
pipe = make_pipeline(
    FunctionTransformer(envelope_correlation, kw_args=kw_args),
    FunctionTransformer(lambda x: x.get_data(output='raveled').squeeze()),
    # FunctionTransformer(apply_degree_across_epochs,
    # kw_args=dict(threshold_prop=0.01)),
    RandomForestClassifier(n_estimators=100, random_state=42),
)

# Train
y_train = epochs_train.events[:, 2]
pipe.fit(epochs_train, y_train)

# Test
y_pred = pipe.predict(epochs_test)

# Assess the results
y_test = epochs_test.events[:, 2]
acc = accuracy_score(y_test, y_pred)

# dummy classifier
dummby_clf = DummyClassifier(strategy='most_frequent')
dummby_clf.fit(epochs_train, y_train)
dummby_pred = dummby_clf.predict(epochs_test)
dummby_acc = accuracy_score(y_test, dummby_pred)

# We perform slightly better then chance as an initial first pass.
print("Accuracy score: {}".format(acc))
print(f"Naive dummby accuracy score: {dummby_acc}")

##############################################################################
# References
# ----------
# .. footbibliography::
