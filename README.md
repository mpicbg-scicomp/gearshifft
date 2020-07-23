# ![gearshifft](images/gearshifft_logo_img_100.png)

## Benchmark Suite for Heterogeneous Implementations of FFTs [![DOI](https://zenodo.org/badge/60610005.svg)](https://zenodo.org/badge/latestdoi/60610005) [![Build Status](https://travis-ci.org/mpicbg-scicomp/gearshifft.svg?branch=master)](https://travis-ci.org/mpicbg-scicomp/gearshifft)

This is a simple and easy extensible benchmark system to answer the question, which FFT library performs best under which conditions.
Conditions are given by compute architecture, inplace or outplace as well as real or complex transforms, data precision, and so on.
This project is still in development. 

If you want to just browse our results, see the [raw benchmark data](https://www.github.com/mpicbg-scicomp/gearshifft_results/) or our [online visualisation and comparison tool](https://www.kcod.de/gearshifft/).

`gearshifft` was published at [ISC2017](https://link.springer.com/chapter/10.1007/978-3-319-58667-0_11). The respective preprint is available on the [arxiv](https://arxiv.org/abs/1702.00629).

## Requirements

- cmake 3.1+ or 3.6.2+ for allowing cmake to automatically download of half-code (float16 support)
- C++14 capable compiler
- CUDA FFT library cuFFT 8.0+ or clFFT 2.12.0+ (OpenCL) or FFTW 3.3.4+
- Boost version 1.59+
  - should be compiled with same compiler version or ...
  - ... disable the C++11 ABI for GCC with the `-DGEARSHIFFT_CXX11_ABI=OFF` cmake option 
- [half-code](http://half.sourceforge.net) by [Christian Rau](http://sourceforge.net/users/rauy) for float16 support (currently used for cufft half precision FFTs)
  - if your Nvidia GPU does not support float16, you will see a `CUFFT_INVALID_DEVICE` message in the results

## Build

Go to the `gearshifft` directory (created by `git clone`):
```bash
mkdir release && cd release
cmake ..
make -j 4
```
CMake tries to find the libraries and enables the corresponding make targets.
After `make` have finished you can run e.g. `./gearshifft_cufft`.

If the FFT library paths cannot be found, `CMAKE_PREFIX_PATH` has to be used, e.g.:

```bash
export CMAKE_PREFIX_PATH=${HOME}/software/clFFT-cuda8.0-gcc5.4/:/opt/cuda:$CMAKE_PREFIX_PATH
```

Enable float16 support for cuFFT:

```bash
cmake -DGEARSHIFFT_FLOAT16_SUPPORT=ON ..
make half-code         # for float16 data type
make gearshifft_cufft
```

Another example with Boost and FFTW dependencies:

```bash
BOOST_VER=1.67.0
FFTW_VER=3.3.8
BOOST_ROOT=${HOME}/sw/boost-${BOOST_VER}
FFTW_ROOT=${HOME}/sw/fftw-${FFTW_VER}/

export CMAKE_PREFIX_PATH==${BOOST_ROOT}/lib:${BOOST_ROOT}/include:${FFTW_ROOT}:${CMAKE_PREFIX_PATH}
if [ -d "CMakeFiles" ]; then
  make clean
  rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake Makefile
fi
cmake ..
make
```

The following recipe illustrates how to build the benchmark with Score-P user instrumentation.
Note that you need a working Score-P installation availble in your shell session.

```bash
SCOREP_WRAPPER=off cmake -DCMAKE_CXX_COMPILER=scorep-g++ ..
make SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--user --nocompiler --thread=none" -j $(nproc)
```

This will instrument the forward and backward transforms, as well as the planning phases, i.e.
those code paths that are also measured by the following columns in the output `csv` file:

- `Time_PlanInitFwd [ms]` -- `plan_forward`,
- `Time_PlanInitInv [ms]` -- `plan_backward_reuse` or `plan_backward_no_reuse`,
- `Time_FFT [ms]` -- `transform_forward`,
- `Time_iFFT [ms]` -- `transform_backward`.

On each execution of the gearshifft app, the Score-P environment will generate a new directory
`scorep_<date_time_id>` containing a corresponding Cube profile `profile.cubex`.
Here is an example of how to read the PAPI performance counter for single precision floating point
operations:

```bash
export SCOREP_PROFILING_ENABLE_CLUSTERING=false     # always keep executions of the same region seperate
export SCOREP_METRIC_PAPI=PAPI_SP_OPS               # set the desired performance counter
./gearshifft_fftw -e 4096 -r Fftw/float/*/Inplace_Real --rigor=estimate         # run benchmark
cube_info -m PAPI_SP_OPS scorep-20200722_1818_122049453456972/profile.cubex     # show results
```

The output might look something like this:

```
|     PAPI_SP_OPS | Diff-Calltree
|         4512295 |  * gearshifft_fftw
|         4311444 |  |  * fft_benchmark
|         1337772 |  |  |  * plan_forward
|          111481 |  |  |  |  * instance=1
|          111481 |  |  |  |  * instance=2
|          111481 |  |  |  |  * instance=3
|          111481 |  |  |  |  * instance=4
|          111481 |  |  |  |  * instance=5
|          111481 |  |  |  |  * instance=6
|          111481 |  |  |  |  * instance=7
|          111481 |  |  |  |  * instance=8
|          111481 |  |  |  |  * instance=9
|          111481 |  |  |  |  * instance=10
|          111481 |  |  |  |  * instance=11
|          111481 |  |  |  |  * instance=12
|         1306836 |  |  |  * plan_backward_no_reuse
|          108903 |  |  |  |  * instance=1
|          108903 |  |  |  |  * instance=2
                       ...
|         1137828 |  |  |  * transform_forward
|           94819 |  |  |  |  * instance=1
                       ...
|          528948 |  |  |  * transform_backward
|           44079 |  |  |  |  * instance=1
                       ...
```

This report states that e.g. every executed forward transformation consumed 94,819 single precision
floating point operations.
For the sake of precise time measurement, this also includes one operation that is caused by the
timer implementation (as is the case with the other regions).
The FP operations in the planning stages are accumulated during calculation of twiddle factors.
Any other operation counted in the `fft_benchmark` or `gearshifft_*` regions that exceed the amount
of their inner regions, are due to validation of the FFT results by gearshifft.

More information on profiling and tracing with Score-P can be found
[in the documentation](https://www.vi-hps.org/projects/score-p/).

## Install

Set `CMAKE_INSTALL_PREFIX` and `GEARSHIFFT_INSTALL_CONFIG_PATH` as you wish, otherwise defaults are used.
```bash
mkdir release && cd release
cmake -DCMAKE_INSTALL_PREFIX=${HOME}/gearshifft \
      -DGEARSHIFFT_INSTALL_CONFIG_PATH=${HOME}/gearshifft/configs \
      ..
make -j 4 install

```

### Build Boost from Source

<details>

```bash
## compile_boost.sh
# boost version to download
BOOST_VERSION=1.67.0
# where to download
BOOST_SRC="${HOME}/Downloads/boost"
# boost install dir
BOOST_ROOT="${HOME}/sw/boost-${BOOST_VERSION}"
# if install dir is empty, then download && build
if [[ -z "$(ls -A ${BOOST_ROOT})" ]]; then
  mkdir -p ${BOOST_SRC}
  wget http://sourceforge.net/projects/boost/files/boost/${BOOST_VERSION}/boost_${BOOST_VERSION//\./_}.tar.bz2 -nc -O "${BOOST_SRC}/../boost.tar.bz2"
  (cd ${BOOST_SRC}/../; tar jxf boost.tar.bz2 --strip-components=1 -C ${BOOST_SRC})
  (cd ${BOOST_SRC}; ./bootstrap.sh --with-libraries=program_options,filesystem,system,test)
  (cd ${BOOST_SRC}; ./b2 --prefix="$BOOST_ROOT" -d0 install --variant=release)
fi
```

</details>

### Build FFTW from Source

<details>

```bash
## compile_fftw.sh
# fftw version to download
VERS=3.3.8
# we compile in separated directories, so binaries do not clash
VERS_single=${VERS}_single
VERS_double=${VERS}_double
# the install directory for fftw and fftwf
FFTW_ROOT=${HOME}/sw/fftw-${VERS}/
# "./compile_fftw.sh clean" removes directories
if [ "$1" == "clean" ]; then
  echo "clean sources"
  rm -rf fftw-${VERS_single} fftw-${VERS_double}
fi
# if directories do not exist, create them and unpack fftw_**.tar.gz 
if [ ! -d "fftw-${VERS_single}" ] && [ ! -d "fftw-${VERS_double}" ]; then
  wget http://www.fftw.org/fftw-${VERS}.tar.gz
  mkdir fftw-${VERS_single}
  echo "unpacking"
  tar xfz fftw-${VERS}.tar.gz -C fftw-${VERS_single}
  cp -r fftw-${VERS_single} fftw-${VERS_double}
fi
IFLAG_STD="--enable-static=yes --enable-shared=yes --with-gnu-ld  --enable-silent-rules --with-pic"
IFLAGS="--prefix=${FFTW_ROOT} --enable-openmp --enable-sse2 -q $IFLAG_STD"
# float
cd fftw-${VERS_single}/fftw-${VERS}
./configure $IFLAGS "--enable-float"
make -j8
make install
# double
cd ../../fftw-${VERS_double}/fftw-${VERS}
./configure $IFLAGS
make -j8
make install
```

</details>


## Testing

The tests can be executed by `make test` after you have compiled the binaries.

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
  --rigor arg (=measure)            FFTW rigor (measure, estimate, wisdom, 
                                    patient or exhaustive)
  --wisdom_sp arg                   Wisdom file for single-precision.
  --wisdom_dp arg                   Wisdom file for double-precision.
  --plan_timelimit arg (=-1)        Timelimit in seconds for planning in FFTW.
```

### FFT Extents Presets

`gearshifft` comes with preset files which contain a broad range of FFT extents and are located in:

```bash
/gearshifft/share/gearshifft/*.conf

```

For FFTW benchmarks with, e.g., `FFTW_MEASURE` runtimes grow very quickly. You can delete or comment out the unwanted lines in the configuration file with `#`.
The extents configurations are loaded by the `gearshifft` command-line interface:

```bash
./gearshifft_fftw -f myextents.conf
```

### Examples 

Runs complete benchmark for clFFT (also applies for cuFFT, FFTW, ..)
```bash
./gearshifft_clfft
# equals
./gearshifft_clfft -f myextents.conf -o result.csv -d gpu -n 0
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

#### Hints for FFTW Usage

`gearshifft_fftw` offers some additional options, e.g., for the plan rigor or the time limit of plans, just check `./gearshifft_fftw -h`.
FFTW itself allows you to pregenerate FFT plans by the so-called wisdom files. These can be generated by FFTW tools, e.g.:
```bash
./fftw-3.3.5_single/tools/fftwf-wisdom -v -c -n -T 24 -o wisdomf  # single precision
./fftw-3.3.5_double/tools/fftw-wisdom  -v -c -n -T 24 -o wisdom   # double precision
```
It is recommended to compile double-precision (fftw-...) and single-precision (fftwf-...) to separate directories to avoid linker issues.
The wisdoms settings must match the `gearshifft_fftw` configuration (number of cores, precision, extents).
Most of the times you do not benefit from a multi-core setting, because the FFT is already computed almost in no time.
FFTW spends a lot of time in planning, except you use `FFTW_ESTIMATE` or time limits, which are also only lower borders.

## Measurement

The FFT scenario is a roundtrip FFT, i.e. forward and backward transformation.
The result is compared with the original input data and an error is shown, if there was a mismatch.
If a benchmark cannot be completed due to an error, it proceeds with the next benchmark.
The library dependent FFT steps are abstracted, where following steps are wrapped by timers.
- buffer allocation
- plan creation (forward/backward plan)
- memory transfers (up-/download)
- forward and backward transforms
- cleanup
- total time (allocation, planning, transfers, FFTs, cleanup)
- device initialization/teardown (only once per runtime)

Furthermore, the required buffer sizes to run the FFT are recorded. 

## CSV Output

The results of the benchmark runs are stored into a comma-separated values file (.csv), after the last run has been completed.
To ease evaluation, the entries are sorted by columns 
- FFT transform kind and precision
- dimkind (oddshape, powerof2, radix357)
 - oddshape: at least one extent is not a combination of a power of 2,3,5,7
 - powerof2: all extents are powers of 2
 - radix357: extents are combination of powers of 2,3,5,7 and not all are powers of 2
- extents and runs

See CSV header for column titles and meta-information (memory, number of runs, error-bound, hostname, timestamp, ...).

During runtime, the results are stored in a backup file (suffixed with "~") in the order in which they occur.
The benchmark will accumulate a number of results before writing them to disk.
This number can be set at compile time by defining `GEARSHIFFT_DUMP_FREQUENCY`.
The default value is 1 which means every result will be written right away.

## Tested on ...

- linux (CentOS, RHEL, ArchLinux, Ubuntu, Fedora)
- gcc 5.3.0, gcc 6.2.0, gcc 7.1.1, gcc 8.1.1, clang 3.8 & 4.0.1 (FFTW threads disabled)
- CUDA 8.0.x, CUDA 9.x
- clFFT 2.12.0, 2.12.1, 2.12.2
- FFTW 3.3.4, 3.3.5, 3.3.6pl1, 3.3.8
- OpenCL 1.2 (Nvidia, AMD, Intel)
- Nvidia Pascal V100, P100, Kepler K80, K20Xm, GTX1080, Broadwell, Haswell and Sandybridge Xeon CPUs

## Issues

- clFFT does not support arbitrary transform sizes. The benchmark renders such tests as failed.
- clFFT on CPU cannot transform the 4096-FFT and 4096x4096-FFTs (see [this issue](https://github.com/clMathLibraries/clFFT/issues/171))
- clFFT seems to have lower transform size limits on CPU than on GPU (a complex 16384x16384 segfaults on clfft CPU, while it works on GPU). gearshifft marks these cases as "Unsupported lengths" and skips them.
- At the moment this is for single-GPUs, batches are not considered
- in case the Boost version (e.g. 1.62.0) you have is more recent than your `cmake` (say 2.8.12.2), use `cmake -DBoost_ADDITIONAL_VERSIONS=1.62.0 -DBOOST_ROOT=/path/to/boost/1.62.0 <more flags>`
- Windows or MacOS is not supported yet, feel free to add a pull-request
- cufft float16 transforms overflow at >=1048576 elements
