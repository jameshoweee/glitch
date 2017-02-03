# GLITCH: A Discrete Gaussian Testing Suite For Lattice-Based Cryptography

03-Feb-2017 James Howe jhowe02{at}qub{dot}ac{dot}uk
Centre for Secure Information Technologies (CSIT), Queen's University Belfast, UK

## Introduction

The code provided in testgauss.py is the main file, which runs on data files given in the samples directory. The samples directory contains data of different sample sizes, from different samplers, as well as 'buggy' samples. An environment file test_gauss.yml is also provided in order to recreate the Anaconda environment through which testgauss.py works.

## Brief Description of Code

The code is entirely adaptable, but for use in testing discrete Gaussian samplers, there should not be any need to change the code except for the target standard deviation (target_sigma). The main parameters that would be of use to adapt are:

1. The precision, defined as 'getcontext()Context(prec = 128, traps=[Overflow, DivisionByZero, InvalidOperation])' using the decimal library. Change the prec variable as required.

2. Below this are the target values for the parameters (prepended 'target_') of the discrete Gaussian sampler, that is, the expected values for:

1. Mean
2. Standard deviation
3. Precision
4. Tail cut
5. Skewness[^fn-sample_footnote]
6. Kurtosis[^fn-sample_footnote]
7. Hyperskewness[^fn-sample_footnote]
8. Hyperkurtosis[^fn-sample_footnote]

[^fn-sample_footnote]: These values should not require any change.

To run in Anaconda, simply run: `python testgauss.py`

***

For background information on discrete Gaussian samplers, the publications by [Howe *et al.*][1] and [Dwarakanath and Galbraith][2] are recommended.

[1]: Howe, J., Khalid, A., Rafferty, C., Regazzoni, F. and O'Neill, M., 2016. On Practical Discrete Gaussian Samplers For Lattice-Based Cryptography. IEEE Transactions on Computers. Available at: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7792671

[2] Dwarakanath, N.C. and Galbraith, S.D., 2014. Sampling from discrete Gaussians for lattice-based cryptography on a constrained device. Applicable Algebra in Engineering, Communication and Computing, 25(3), pp.159-180. Available at: http://link.springer.com/article/10.1007/s00200-014-0218-3

**Absolutely no warranty whatsoever**