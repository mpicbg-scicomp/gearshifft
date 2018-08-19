cmake_minimum_required(VERSION 3.7)

find_package(gearshifft REQUIRED  HINTS ${CMAKE_SOURCE_DIR})

if(CUDA_FOUND AND GEARSHIFFT_CUFFT)
  gearshifft_add_cufft_executable(
    gearshifft_cufft SOURCES benchmark.cpp options.cpp INSTALL)
endif()

if(CLFFT_FOUND AND GEARSHIFFT_CLFFT)
  gearshifft_add_clfft_executable(
    gearshifft_clfft SOURCES benchmark.cpp options.cpp INSTALL)
endif()

if(FFTW_FOUND AND GEARSHIFFT_FFTW)
  gearshifft_add_fftw_executable(
    gearshifft_fftw SOURCES benchmark.cpp options.cpp INSTALL)
endif()

if(HCFFT_FOUND AND GEARSHIFFT_HCFFT)
  gearshifft_add_hcfft_executable(
    gearshifft_hcfft SOURCES benchmark.cpp options.cpp INSTALL)
endif()
