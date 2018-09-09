# gearshifft cmake options

option(GEARSHIFFT_VERBOSE "Verbose output during build generation." ON)
option(GEARSHIFFT_USE_STATIC_LIBS "Static linking of libraries." OFF)
option(GEARSHIFFT_TESTS_ADD_CPU_ONLY "Only add tests which run on CPU." OFF)

option(GEARSHIFFT_BACKEND_CUFFT "Compile gearshifft_cufft if possible" ON)
option(GEARSHIFFT_BACKEND_CLFFT "Compile gearshifft_clfft if possible" ON)
option(GEARSHIFFT_BACKEND_FFTW  "Compile gearshifft_fftw if possible" ON)
option(GEARSHIFFT_BACKEND_FFTW_OPENMP "Use OpenMP parallel FFTW libraries if found" ON)
option(GEARSHIFFT_BACKEND_FFTW_PTHREADS "Use pthreads parallel FFTW libraries if found" OFF)
option(GEARSHIFFT_BACKEND_HCFFT "< Not implemented yet >" OFF)

set( GEARSHIFFT_EXT_DIR ${GEARSHIFFT_ROOT}/ext CACHE STRING "Install directory for external dependencies")

set(GEARSHIFFT_CXX11_ABI "1" CACHE STRING "Enable _GLIBCXX_USE_CXX11_ABI in GCC 5.0+")
set_property(CACHE GEARSHIFFT_CXX11_ABI PROPERTY STRINGS "0;1")

set(GEARSHIFFT_FLOAT16_SUPPORT "0" CACHE STRING "Enable float16 data type (cufft and ComputeCapability 5.3+ only)")
set_property(CACHE GEARSHIFFT_FLOAT16_SUPPORT PROPERTY STRINGS "0;1")

set(GEARSHIFFT_NUMBER_WARM_RUNS "10" CACHE STRING "Number of repetitions of an FFT benchmark after a warmup.")
set(GEARSHIFFT_NUMBER_WARMUPS "2" CACHE STRING "Number of warmups of an FFT benchmark.")
set(GEARSHIFFT_ERROR_BOUND "-1" CACHE STRING "Error-bound for FFT benchmarks (<0 for dynamic error bound).")

#-------------------------------------------------------------------------------

if(GEARSHIFFT_BACKEND_HCFFT)
  message(FATAL_ERROR "HCFFT is not implemented yet.")
endif()


if(GEARSHIFFT_VERBOSE)
  include(${GEARSHIFFT_ROOT}/cmake/get-gearshifft-options.cmake)
  get_gearshifft_options(gearshifft_options "\n ")
  message(" ${gearshifft_options}\n")
endif()
