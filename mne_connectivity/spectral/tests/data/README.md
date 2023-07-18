Author: Thomas S. Binns <t.s.binns@outlook.com>

The files found here are used for the regression test of the multivariate
connectivity methods for MIC, MIM, GC, and TRGC
(`test_multivariate_spectral_connectivity_epochs_regression()` of
`test_spectral.py`).

`example_multivariate_data.pkl` consists of four channels of randomly-generated
data with 15 epochs and 200 timepoints per epoch. Connectivity was computed in
MATLAB using the original implementations of these methods and saved as a
dictionary in `example_multivariate_matlab_results.pkl`. A publicly-available
implementation of the methods in MATLAB can be found here:
https://github.com/sccn/roiconnect. 

As the MNE code for computing the cross-spectral density matrix is not
available in MATLAB, the CSD matrix was computed using MNE and then loaded into
MATLAB to compute the connectivity from the original implementations using the
same processing settings in MATLAB and Python. That is: a sampling frequency of
100 Hz; method='multitaper'; fskip=0; faverage=False; tmin=0; tmax=None;
mt_bandwidth=4; mt_low_bias=True; mt_adaptive=False; gc_n_lags=20;
rank=([2], [2]) - i.e. no rank subspace projection; indices=([0, 1], [2, 3]) -
i.e. connection from first two channels to last two channels. It is 
important that no changes are made to the settings for computing the CSD or the
final connectivity scores, otherwise this test will be invalid!

One key difference is that the MATLAB implementation for computing MIC returns
the absolute value of the results, so we must take the absolute value of the
results returned from the MNE function to make the comparison. We do not return
the absolute values of the results, as relevant information such as phase angle
differences are lost.