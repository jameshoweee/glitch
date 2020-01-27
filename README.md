# GLITCH: A Discrete Gaussian Testing Suite For Lattice-Based Cryptography
Paper: https://eprint.iacr.org/2017/438

An improved and updated version of this code is available at: [https://eprint.iacr.org/2019/1411](https://eprint.iacr.org/2019/1411).

## Introduction

The code provided in testgauss.py is the main file, which runs on data files given in the samples directory. The samples directory contains data of different sample sizes, from different samplers, as well as 'buggy' samples. An environment file test_gauss.yml is also provided in order to recreate the Anaconda environment through which testgauss.py works.

## Brief Description

The code is entirely adaptable, but for use in testing discrete Gaussian samplers, there should not be any need to change the code except for the target standard deviation (target_sigma). The main parameters that would be of use to adapt are:

1. The precision, defined as `getcontext()Context(prec = 128, traps=[Overflow, DivisionByZero, InvalidOperation])` using the decimal library. Change the `prec` variable as required.

2. Below this are the target values for the parameters (prepended `target_`) of the discrete Gaussian sampler, that is, the expected values for:

  * Mean  
  * Standard deviation  
  * Precision  
  * Tail cut  
  * Skewness[1]  
  * Kurtosis[1]  
  * Hyperskewness[1]  
  * Hyperkurtosis[1]  

[1]: These values should not require any change.

To run in Anaconda, simply return: `python testgauss.py`

***

For background information on discrete Gaussian samplers, the publications by Howe *et al.*[2] and Dwarakanath and Galbraith[3] are recommended.

[2]: Howe, J., Khalid, A., Rafferty, C., Regazzoni, F. and O'Neill, M., 2016. On Practical Discrete Gaussian Samplers For Lattice-Based Cryptography. IEEE Transactions on Computers. Available at: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7792671

[3] Dwarakanath, N.C. and Galbraith, S.D., 2014. Sampling from discrete Gaussians for lattice-based cryptography on a constrained device. Applicable Algebra in Engineering, Communication and Computing, 25(3), pp.159-180. Available at: http://link.springer.com/article/10.1007/s00200-014-0218-3

**Provided with absolutely no warranty whatsoever**
