cmake_minimum_required(VERSION 3.1)

# cmake general setup

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

set(CMAKE_COLOR_MAKEFILE ON)
# And optionally
#set(CMAKE_VERBOSE_MAKEFILE ON)

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(SEND_ERROR "In-source builds are not allowed.")
endif ()
if (WIN32)
  set(CMAKE_SHARED_LIBRARY_PREFIX "")
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

# set project build type
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
message(">> Setting up ${CMAKE_BUILD_TYPE} build")

# gearshifft cmake options

option(GEARSHIFFT_CXX11_ABI "enable _GLIBCXX_USE_CXX11_ABI in GCC 5.0+" ON)
option(GEARSHIFFT_FFTW_OPENMP "use OpenMP parallel fftw libraries if found" ON)
option(GEARSHIFFT_FFTW_PTHREADS "use pthreads parallel fftw libraries if found" OFF)
option(GEARSHIFFT_CUFFT "Compile gearshifft_cufft if available?" ON)
option(GEARSHIFFT_CLFFT "Compile gearshifft_clfft if available?" ON)
option(GEARSHIFFT_FFTW  "Compile gearshifft_fftw if available?" ON)
option(GEARSHIFFT_HCFFT "< Not implemented yet >" OFF)

set(GEARSHIFFT_NUMBER_WARM_RUNS "10" CACHE STRING "Number of repetitions of an FFT benchmark after a warmup.")
set(GEARSHIFFT_NUMBER_WARMUPS "2" CACHE STRING "Number of warmups of an FFT benchmark.")
set(GEARSHIFFT_ERROR_BOUND "0.00001" CACHE STRING "Error-bound for FFT benchmarks.")
set(GEARSHIFFT_INSTALL_CONFIG_PATH "/usr/share/gearshifft/" CACHE STRING "Default install path of config files.")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if(${GEARSHIFFT_CXX11_ABI})
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
    message(">> CXX11_ABI enabled.")
  else()
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
    message(">> CXX11_ABI disabled.")
  endif()

  if (NOT (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0
        OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0))
    message(FATAL_ERROR "gearshifft requires g++ 5.0 or greater.")
  endif()
  #gcc4.8+ uses dwarf-4. If you have gdb <7.5 then use this line
  #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -gdwarf-2")
  #gdb 7.0-7.5 may require
  #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fvar-tracking-assignments")
endif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")


set(DEV_TESTS "" CACHE STRING "Tests for developing (No Benchmarks)")
if(DEV_TESTS)
  message(">> Test mode")
  add_definitions(-DDEV_TESTS)
  set(TESTS tests.cpp)
endif()

add_subdirectory(src)
