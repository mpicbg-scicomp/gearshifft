include(CMakeDependentOption)

# gearshifft cmake options

option(GEARSHIFFT_VERBOSE "Verbose output during build generation." OFF)
option(GEARSHIFFT_USE_STATIC_LIBS "Force static linking Boost and FFTW (use libraries' cmake variables otherwise)." OFF)
option(GEARSHIFFT_TESTS_ADD_CPU_ONLY "Only add tests which run on CPU." OFF)

set(GEARSHIFFT_CXX11_ABI "1" CACHE STRING "Enable _GLIBCXX_USE_CXX11_ABI in GCC 5.0+")
set_property(CACHE GEARSHIFFT_CXX11_ABI PROPERTY STRINGS "0;1")

# FFT back-ends
option(GEARSHIFFT_BACKEND_CUFFT "Compile gearshifft_cufft if possible" ON)
option(GEARSHIFFT_BACKEND_CLFFT "Compile gearshifft_clfft if possible" ON)
option(GEARSHIFFT_BACKEND_ROCFFT "Compile gearshifft_rocfft if possible (requires HCC as compiler)" ON)

option(GEARSHIFFT_BACKEND_FFTW  "Compile gearshifft_fftw if possible" ON)
cmake_dependent_option(
  GEARSHIFFT_BACKEND_FFTW_OPENMP "Use OpenMP parallel FFTW libraries if found" ON
  "GEARSHIFFT_BACKEND_FFTW" ON)
cmake_dependent_option(
  GEARSHIFFT_BACKEND_FFTW_PTHREADS "Use pthreads parallel FFTW libraries if found" OFF
  "GEARSHIFFT_BACKEND_FFTW" ON)

option(GEARSHIFFT_BACKEND_FFTWWRAPPERS  "Compile gearshifft_fftwwrappers if possible" ON)

# backend-disabler

option(GEARSHIFFT_BACKEND_ROCFFT_ONLY "Disable all other backends" OFF)
option(GEARSHIFFT_BACKEND_CLFFT_ONLY "Disable all other backends" OFF)
option(GEARSHIFFT_BACKEND_CUFFT_ONLY "Disable all other backends" OFF)
option(GEARSHIFFT_BACKEND_FFTW_ONLY "Disable all other backends" OFF)
option(GEARSHIFFT_BACKEND_FFTWWRAPPERS_ONLY "Disable all other backends" OFF)

# precisions

# half precision currently only supported in cufft backend
# TODO: check other backends
if(GEARSHIFFT_BACKEND_CUFFT)
  set(GEARSHIFFT_FLOAT16_SUPPORT "0" CACHE STRING "Enable float16 data type (cufft and ComputeCapability 5.3+ only)")
  set_property(CACHE GEARSHIFFT_FLOAT16_SUPPORT PROPERTY STRINGS "0;1")

  set(GEARSHIFFT_PRECISION_HALF_ONLY "0" CACHE STRING "Compile benchmarks in half precision only.")
  set_property(CACHE GEARSHIFFT_PRECISION_HALF_ONLY PROPERTY STRINGS "0;1")
endif()

set(GEARSHIFFT_PRECISION_SINGLE_ONLY "0" CACHE STRING "Compile benchmarks in single precision only.")
set_property(CACHE GEARSHIFFT_PRECISION_SINGLE_ONLY PROPERTY STRINGS "0;1")

set(GEARSHIFFT_PRECISION_DOUBLE_ONLY "0" CACHE STRING "Compile benchmarks in double precision only.")
set_property(CACHE GEARSHIFFT_PRECISION_DOUBLE_ONLY PROPERTY STRINGS "0;1")


# benchmark setting

set(GEARSHIFFT_NUMBER_WARM_RUNS "10" CACHE STRING "Number of repetitions of an FFT benchmark after a warmup.")
set(GEARSHIFFT_NUMBER_WARMUPS "2" CACHE STRING "Number of warmups of an FFT benchmark.")
set(GEARSHIFFT_ERROR_BOUND "-1" CACHE STRING "Error-bound for FFT benchmarks (<0 for dynamic error bound).")

#-------------------------------------------------------------------------------

if(GEARSHIFFT_VERBOSE)
  include(get_gearshifft_options)
  get_gearshifft_options(gearshifft_options "\n ")
  message(" ${gearshifft_options}\n")
endif()

if(GEARSHIFFT_BACKEND_CUFFT_ONLY
    OR
    GEARSHIFFT_BACKEND_CLFFT_ONLY
    OR
    GEARSHIFFT_BACKEND_ROCFFT_ONLY
    OR
    GEARSHIFFT_BACKEND_FFTW_ONLY
    OR
    GEARSHIFFT_BACKEND_FFTWWRAPPERS_ONLY
    )
  set(GEARSHIFFT_BACKEND_CUFFT ${GEARSHIFFT_BACKEND_CUFFT_ONLY} CACHE BOOL "" FORCE)
  set(GEARSHIFFT_BACKEND_CLFFT ${GEARSHIFFT_BACKEND_CLFFT_ONLY} CACHE BOOL "" FORCE)
  set(GEARSHIFFT_BACKEND_ROCFFT ${GEARSHIFFT_BACKEND_ROCFFT_ONLY} CACHE BOOL "" FORCE)
  set(GEARSHIFFT_BACKEND_FFTW ${GEARSHIFFT_BACKEND_FFTW_ONLY} CACHE BOOL "" FORCE)
  set(GEARSHIFFT_BACKEND_FFTWWRAPPERS ${GEARSHIFFT_BACKEND_FFTWWRAPPERS_ONLY} CACHE BOOL "" FORCE)
endif()
