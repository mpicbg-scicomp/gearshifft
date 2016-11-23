# ![gearshifft](images/gearshifft_logo_img_100.png)

Benchmark Suite for Heterogeneous Implementations of FFTs

This is a simple and easy extensible benchmark system to answer the question, which FFT library performs best under which conditions.
Conditions are given by compute architecture, inplace or outplace as well as real or complex transforms, data precision, and so on.
This project is still in development. 

## Requirements

- cmake 2.8+
- C++14 capable compiler
- CUDA FFT library cuFFT 7.5+ or clFFT 2.12.0+ (OpenCL) or FFTW 3.3.4+
- Boost version 1.56+
  - should be compiled with same compiler version or ...
  - ... disable the C++11 ABI for GCC with the `-DGEARSHIFFT_CXX11_ABI=OFF` cmake option 

## Build
Go to the gearshifft directory (created by git clone ...):
```
mkdir release && cd release
cmake ..
make -j 4
```
CMake tries to find the libraries and enables the corresponding make targets.
After make have finished you can run e.g. `./gearshifft_cufft`.

## Usage

See help message (pass `--help|-h`) for the command line options.
```
-h [ --help ]                     Print help messages
-e [ --extent ] arg               specific extent (eg. 1024x1024) [>=1 nr. of
                                  args possible]
-f [ --file ] arg                 file with extents (row-wise csv) [>=1 nr.
                                  of args possible]
-o [ --output ] arg (=result.csv) output csv file, will be overwritten!
-v [ --verbose ]                  for console output
-d [ --device ] arg (=gpu)        Compute device = (gpu|cpu|acc|<ID>). If
                                  device is not supported by FFT lib, then it
                                  is ignored and default is used.
-n [ --ndevices ] arg (=0)        Number of devices (0=all), if supported by
                                  FFT lib (e.g. clfft and fftw with n CPU
                                  threads).
-l [ --list-devices ]             List of available compute devices with IDs,
                                  if supported.
-b [ --list-benchmarks ]          Show registered benchmarks
-r [ --run-benchmarks ] arg       Run specific benchmarks (wildcards
                                  possible, e.g. ClFFT/float/*/Inplace_Real)
```
**Examples**

Runs complete benchmark for clFFT (also applies for cuFFT, fftw, ..)
```
./gearshifft_clfft
// equals
./gearshifft_clfft -f ../config/extents.csv -o result.csv -d gpu -n 0
```
List compute devices
```
./gearshifft_clfft -l
$ "ID",0:0,"ClFFT Informations","Device","Tesla K20Xm","Hardware","OpenCL 1.2 CUDA" <snip>
$ "ID",1:0,"ClFFT Informations","Device","Intel(R) Xeon(R) CPU E5-2450 0 @ 2.10GHz" <snip>
```
Select compute devices by id returned by `--list-devices|-l`
```
./gearshifft_clfft -d 1:0
```
512x512-point FFT, single and double precision, all transforms, print result to console.
```
./gearshifft_clfft -e 512x512 -v
```
1048576-point FFT, only real outplace transform in single precision.
```
./gearshifft_clfft -e 1048576 -r */float/*/Outplace_Real
```
1024-point and 128x256x64-point FFTs, on CPU with 1 thread.
- If no device parameter is given, then ClFFT/OpenCL will use GPU, if available
```
./gearshifft_clfft -e 1024 128,256,64 -d cpu -n 1
```
1024x1024-point FFT, double precision inplace transforms.
- `--list-benchmarks|-b` gives a list of available extents read in `--file|-f` (default is ../config/extents.csv)
```
./gearshifft_clfft -r */double/1024x1024/Inplace_*
```

## Measurement

The FFT scenario is a roundtrip FFT, i.e. forward and backward transformation.
The result is compared with the original input data and an error is shown, if there was a mismatch.
If a benchmark cannot be completed due to an error, it proceeds with the next benchmark.
The library dependent FFT steps are abstracted, where following steps are wrapped by timers.
- buffer allocation
- plan creation
- memory transfers (up-/download)
- forward and backward transforms
- cleanup
- timer which measures FFT process from upload to download (called "Time Device")
- total time (allocation, plan, transfers, FFTs, cleanup)
- device initialization/teardown (only once per runtime)

## CSV Output

The results of the benchmark runs are stored to CSV, after the last run has been completed.
To ease evaluation, the entries are sorted by columns 
- FFT transform kind and precision
- dimkind (oddshape, powerof2, radix357)
 - oddshape: at least one extent is not a combination of a power of 2,3,5,7
 - powerof2: all extents are powers of 2
 - radix357: extents are combination of powers of 2,3,5,7 and not all are powers of 2
- extents and runs
See CSV header for column titles.

## Tested on ...

- gcc 5.3.0, gcc 6.2.0
- CUDA 7.5.18, CUDA 8.0
- cuFFT from CUDA 7.5.18 and CUDA 8.0
- clFFT 2.12.0 and 2.12.1
- FFTW 3.3.4 and 3.3.5
- OpenCL 1.2-4.4.0.117 (Nvidia)
- Nvidia Kepler K80 GPU and Kepler K20X GPU

## Issues

- cuFFT 7.5 contexts might become messed up after huge allocations failed (see [link](https://devtalk.nvidia.com/default/topic/956093/gpu-accelerated-libraries/cufft-out-of-memory-yields-quot-irreparable-quot-context/))
- clFFT does not support arbitrary transform sizes. The benchmark renders such tests as failed.
- At the moment this is for single-GPUs, batches are not considered
- if gearshifft is killed before, no output is created, which might be an issue on a job scheduler system like slurm (exceeding memory assignment)
- in case the boost version (e.g. 1.62.0) you have is more recent than your cmake (say 2.8.12.2), use `cmake -DBoost_ADDITIONAL_VERSIONS=1.62.0 -DBOOST_ROOT=/path/to/boost/1.62.0 <more flags>`

## Roadmap

- [x] cuFFT
- [x] clFFT
- [x] fftw
- [ ] hcFFT: ROC based hcFFT library
- [ ] liFFT: include library independent FFT framework
- [ ] callbacks to benchmark a typical FFT use case

## Results
fftw/haswell contains results for FFTW_MEASURE, FFTW_ESTIMATE and FFTW_WISDOM_ONLY. The planning time limit is set to FFTW_NO_TIMELIMIT (can be set with cmake option GEARSHIFFT_FFTW_TIMELIMIT).
fftw was compiled with:
```
--enable-static=yes --enable-shared=yes --with-gnu-ld  --enable-silent-rules --with-pic --enable-openmp --enable-sse2
```
fftw/haswell was run on:
```
2x Intel(R) Xeon(R) CPU E5-2680 v3 (12 cores) @ 2.50GHz, MultiThreading disabled, 128 GB SSD local disk, 64 GB RAM
```
