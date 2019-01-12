# gearshifft cmake options

option(GEARSHIFFT_VERBOSE "Verbose output during build generation." ON)
option(GEARSHIFFT_USE_STATIC_LIBS "Force static linking Boost and FFTW (use libraries' cmake variables otherwise)." OFF)
option(GEARSHIFFT_TESTS_ADD_CPU_ONLY "Only add tests which run on CPU." OFF)

set(GEARSHIFFT_CXX11_ABI "1" CACHE STRING "Enable _GLIBCXX_USE_CXX11_ABI in GCC 5.0+")
set_property(CACHE GEARSHIFFT_CXX11_ABI PROPERTY STRINGS "0;1")

# FFT back-ends
option(GEARSHIFFT_BACKEND_CUFFT "Compile gearshifft_cufft if possible" ON)
option(GEARSHIFFT_BACKEND_CLFFT "Compile gearshifft_clfft if possible" ON)
option(GEARSHIFFT_BACKEND_FFTW  "Compile gearshifft_fftw if possible" ON)
option(GEARSHIFFT_BACKEND_FFTW_OPENMP "Use OpenMP parallel FFTW libraries if found" ON)
option(GEARSHIFFT_BACKEND_FFTW_PTHREADS "Use pthreads parallel FFTW libraries if found" OFF)
option(GEARSHIFFT_BACKEND_ROCFFT "Compile gearshifft_rocfft if possible (requires HCC as compiler)" ON)

# precisions
set(GEARSHIFFT_PRECISION_HALF_ONLY "0" CACHE STRING "Compile benchmarks in half precision only.")
set_property(CACHE GEARSHIFFT_PRECISION_HALF_ONLY PROPERTY STRINGS "0;1")

set(GEARSHIFFT_PRECISION_SINGLE_ONLY "0" CACHE STRING "Compile benchmarks in single precision only.")
set_property(CACHE GEARSHIFFT_PRECISION_SINGLE_ONLY PROPERTY STRINGS "0;1")

set(GEARSHIFFT_PRECISION_DOUBLE_ONLY "0" CACHE STRING "Compile benchmarks in double precision only.")
set_property(CACHE GEARSHIFFT_PRECISION_DOUBLE_ONLY PROPERTY STRINGS "0;1")

set(GEARSHIFFT_FLOAT16_SUPPORT "0" CACHE STRING "Enable float16 data type (cufft and ComputeCapability 5.3+ only)")
set_property(CACHE GEARSHIFFT_FLOAT16_SUPPORT PROPERTY STRINGS "0;1")

# benchmark setting
set(GEARSHIFFT_NUMBER_WARM_RUNS "10" CACHE STRING "Number of repetitions of an FFT benchmark after a warmup.")
set(GEARSHIFFT_NUMBER_WARMUPS "2" CACHE STRING "Number of warmups of an FFT benchmark.")
set(GEARSHIFFT_ERROR_BOUND "-1" CACHE STRING "Error-bound for FFT benchmarks (<0 for dynamic error bound).")

#-------------------------------------------------------------------------------

if(GEARSHIFFT_VERBOSE)
  include(${GEARSHIFFT_ROOT}/cmake/get-gearshifft-options.cmake)
  get_gearshifft_options(gearshifft_options "\n ")
  message(" ${gearshifft_options}\n")
endif()