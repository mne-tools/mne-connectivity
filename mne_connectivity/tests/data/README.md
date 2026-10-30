These files are used for regression testing of:
- Envelope correlation (`envelope_test_data.pkl`; input data and expected output) against the FieldTrip implementation (which uses the square log norm version). In MATLAB, the followning was performed on the input data (Hilbert transformed) to get the expected output:
```matlab
ft_connectivity_powcorr_ortho(reshape(data_hilbert(1,:,:), [4, 64]))
ft_connectivity_powcorr_ortho(reshape(data_hilbert(2,:,:), [4, 64]))
```
- wSMI (`wsmi_test_data.pkl`; input data, hyperparameters, and expected output) against the original wSMI code used within the PICNIC Lab.