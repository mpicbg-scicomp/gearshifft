# MKL Wrappers for FFTW3

## Installation

- consult the [MKL webpage](https://software.intel.com/en-us/mkl-developer-reference-c-fftw3-interface-to-intel-math-kernel-library) on the matter
- setup parallel studio (using environment modules, singularity, or just the compilevars.sh that ships with parallel studio xe)
- then the following should work for you:
```
$ cd ${MKLROOT} 
$ cd interfaces/fftw3xc
$ make libintel64 compiler=intel INSTALL_DIR=/my/path
$ make libintel64 compiler=gnu INSTALL_DIR=/my/path
$ ls /my/path/libfftw3xc_*.a
/my/path/libfftw3xc_gnu.a  /my/path/libfftw3xc_intel.a
```
