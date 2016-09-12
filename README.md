# ![gearshifft](images/gearshifft_logo_img_100.png)

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
- CUDA FFT library cuFFT 7.5+ or clFFT 2.12.0+ for OpenCL or FFTW 3.3.4+
- boost version 1.56+
  - should be compiled with same compiler version or ...
  - ... disable the C++11 ABI for GCC with the `-DUSE_CXX11_ABI=OFF` cmake option 

## Tested on ...

- gcc 5.3.0
- CUDA 7.5.18
- cuFFT from CUDA 7.5.18
- clFFT 2.12.0 and 2.12.1
- FFTW 3.3.4 and 3.3.5
- OpenCL 1.2-4.4.0.117 (Nvidia)
- Nvidia Kepler K80 GPU and Kepler K20X GPU

## Issues

- cuFFT 7.5 contexts might become messed up after huge allocations failed (see [link](https://devtalk.nvidia.com/default/topic/956093/gpu-accelerated-libraries/cufft-out-of-memory-yields-quot-irreparable-quot-context/))
- clFFT does not support arbitrary transform sizes. The benchmark will print only these tests as failed.
- at the moment this is for single-GPUs, batches are not considered

## Roadmap

- [x] cuFFT
- [x] clFFT: emulation of arbitrary transform sizes / non-supported radices
- [x] fftw
- [ ] hcFFT: ROC based hcFFT library
- [ ] liFFT: include library independent FFT framework
- [ ] scripts for creating benchmark summary of the individual results
- [ ] callbacks to benchmark a typical FFT use case
