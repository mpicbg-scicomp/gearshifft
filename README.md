# gearshifft
Benchmark Suite for Heterogenuous FFT Implementations

This is a simple and easy extensible benchmark system to answer the question, which FFT library performs best under which conditions.
Conditions are given by compute architecture, inplace or outplace as well as real or complex transforms, data precision, and so on.
This project is in development, but benchmarks are already possible for cuFFT and clFFT.

Timer and allocation statistics of a benchmark are stored into a csv file.

## Build
CUDA: Check src/CMakeLists.txt for device architectures
```
mkdir build && cd build
cmake ..
make -j 4
```
CMake tries to find the libraries and enables the corresponding make targets.
After make finished you can run e.g. `./gearshifft_cufft_float`.
The result file is called e.g. `gearshifft_cufft_float.csv`.

## Requirements
- cmake 2.8+
- C++14 capable compiler
- CUDA FFT library cuFFT or clFFT for OpenCL
- FFTW
- boost version 1.56+
  - should be compiled with same compiler version or ...
  - ... enable compiler definition `-D_GLIBCXX_USE_CXX11_ABI=0` in ./CMakeLists.txt

## Tested on ...

- gcc 5.3.0
- CUDA 7.5.18
- cuFFT from CUDA 7.5.18
- clFFT 2.12.0 (against FFTW 3.3.4)
- OpenCL 1.2-4.4.0.117 (Nvidia)
- Nvidia Kepler K80 GPU and Kepler K20X GPU

## Issues

- clFFT does not support arbitrary transform sizes. The benchmark will print only these tests as failed.
- at the moment this is for single-GPUs, batches are not considered

## Roadmap

- [x] cuFFT
- [ ] clFFT: emulation of arbitrary transform sizes / non-supported radices
- [ ] hcFFT: ROC based hcFFT library
- [ ] liFFT: include library independent FFT framework
- [ ] integration into test environment (ctest)
- [ ] scripts for creating benchmark summary of the individual results
- [ ] callbacks to benchmark a typical FFT use case
