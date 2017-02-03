# GLITCH: A Discrete Gaussian Testing Suite For Lattice-Based Cryptography

The code provided in testgauss.py is the main file, which runs on data files given in the sample directory. The sample directory contains data of different sizes, from different samplers, as well as 'buggy' samples. An environment file test_gauss.yml is also provided in order to recreate the Anaconda environment through which testgauss.py works.

## Brief Description of Code

The code is entirely adaptable to suit the needs of the user, the main parameters that would be of use to adapt are:

1. The precision, defined as 'getcontext()Context(prec = 128, traps=[Overflow, DivisionByZero, InvalidOperation])' using decimal. Change the prec variable as required.

2. Below this are the target values for the parameters (prepended 'target_') of the discrete Gaussian sampler, that is, the expected values for:

..* Mean
..* Standard deviation
..* Precision
..* Tail cut
..* Skewness *
..* Kurtosis *
..* Hyperskewness *
..* Hyperkurtosis *

* these values should not require any change.