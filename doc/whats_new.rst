:orphan:

.. _whats_new:


What's new?
===========

Here we list a changelog of MNE-realtime.

.. contents:: Contents
   :local:
   :depth: 3

.. currentmodule:: mne_realtime

.. _current:

Current
-------

Changelog
~~~~~~~~~

Bug
~~~

API
~~~

Changes when mne-realtime was part of MNE-Python
------------------------------------------------

Changelog
~~~~~~~~~

- Add ``mne_realtime.MockLSLStream`` to simulate an LSL stream for testing and examples by `Teon Brooks`_

- Add ``mne_realtime.LSLClient`` for realtime data acquisition with LSL streams of data by `Teon Brooks`_ and `Mainak Jas`_

- Add connector to FieldTrip realtime client by `Mainak Jas`_

- New realtime module containing RtEpochs, RtClient and MockRtClient class by `Martin Luessi`_, `Christopher Dinh`_, `Alex Gramfort`_, `Denis Engemann`_ and `Mainak Jas`_


Bug
~~~

- Fix bug in ``mne_realtime.RtEpochs`` where events during the buildup of the buffer were not correctly processed when incoming data buffers are smaller than the epochs by `Henrich Kolkhorst`_

- Fix handling of events in ``mne_realtime.RtEpochs`` when the triggers were split between two buffers resulting in missing and/or duplicate epochs by `Mainak Jas`_ and `Antti Rantala`_

- Fixed bug with ``mne_realtime.FieldTripClient.get_data_as_epoch`` when ``picks=None`` which crashed the function by `Mainak Jas`_

- Fix :class:``mne_realtime.StimServer`` by removing superfluous argument ``ip`` used while initializing the object by `Mainak Jas`_.

API
~~~

- Removed blocking (waiting for new epochs) in ``mne_realtime.RtEpochs.get_data()`` by `Henrich Kolkhorst`_

- The default value of ``stop_receive_thread`` in ``mne_realtime.RtEpochs.stop`` has been changed to ``True`` by `Henrich Kolkhorst`_

Authors
~~~~~~~

People who contributed to MNE-realtime while it was part of MNE-Python (in alphabetical order):

* Alex Gramfort
* Antti Rantala
* Christopher Dinh
* Denis Engemann
* Henrich Kolkhorst
* Mainak Jas
* Martin Luessi
* Teon Brooks

.. _Mainak Jas: https://perso.telecom-paristech.fr/mjas/
.. _Teon Brooks: https://teonbrooks.github.io/
.. _Alex Gramfort: http://alexandre.gramfort.net
.. _Antti Rantala: https://github.com/Odingod
.. _Henrich Kolkhorst: https://github.com/hekolk
.. _Martin Luessi: https://www.martinos.org/user/8245
.. _Denis Engemann: http://denis-engemann.de
.. _Christopher Dinh: https://github.com/chdinh
