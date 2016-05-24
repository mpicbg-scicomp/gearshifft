# gearshifft
Benchmark Suite for Heterogenuous FFT Implementations

This is a simple and easy extensible benchmark system to answer the question, which FFT library performs best under which conditions.
Conditions are given by compute architecture, inplace or outplace as well as real or complex transforms, data precision, and so on.
This project is in development, but benchmarks are already possible for cuFFT and clFFT.

Timer and allocation statistics of a benchmark are stored into a csv file.

## Requirements
- cmake 2.8+
- C++14 capable compiler
- CUDA FFT library cuFFT or clFFT for OpenCL
- FFTW

## Tested on ...
- gcc 5.3.0
- CUDA 7.5.18
- cuFFT from CUDA 7.5.18
- clFFT 2.12.0 (against FFTW 3.3.4)
- OpenCL 1.2-4.4.0.117 (Nvidia)
- Nvidia Kepler K80 GPU and Kepler K20X GPU

## Roadmap
- [x] cuFFT
- [ ] clFFT: emulation of arbitrary transform sizes / non-supported radices
- [ ] liFFT: include library independent FFT framework
- [ ] scripts for creating benchmark summary of the individual results
- [ ] callbacks to benchmark a typical FFT use case
