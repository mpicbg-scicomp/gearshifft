cmake_minimum_required(VERSION 3.1)
project(gearshifft)

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
set(GEARSHIFFT_ERROR_BOUND "0.00001" CACHE STRING "Error-bound for FFT benchmarks (<0 for dynamic error bound).")
set(GEARSHIFFT_INSTALL_CONFIG_PATH "${CMAKE_INSTALL_PREFIX}/share/gearshifft/" CACHE STRING "Default install path of config files.")

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

set(CMAKE_CXX_STANDARD 14)



#------------------------------------------------------------------------------
# half-code
#------------------------------------------------------------------------------
include(ExternalProject)
ExternalProject_Add(half-code
  PREFIX ${CMAKE_BINARY_DIR}/third-party
  URL https://downloads.sourceforge.net/project/half/half/1.12.0/half-1.12.0.zip
  URL_MD5 86d023c0729abf3465bcd55665a39013
  INSTALL_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  LOG_DOWNLOAD ON
  LOG_CONFIGURE OFF
  LOG_BUILD OFF
  )
include_directories(${CMAKE_BINARY_DIR}/third-party/src)

#------------------------------------------------------------------------------
# Boost
#------------------------------------------------------------------------------

find_package(Boost 1.56 QUIET COMPONENTS system unit_test_framework program_options REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
endif()

#------------------------------------------------------------------------------
# CUDA+CUFFT
#------------------------------------------------------------------------------

find_package(CUDA)
if(CUDA_FOUND AND GEARSHIFFT_CUFFT)
  include(FindCUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  list(APPEND FFTLIBS "cufft") # register fft lib client for compilation
  message(">> cuFFT -> " ${CUDA_CUFFT_LIBRARIES})
else()
  message("<< cuFFT benchmark disabled.")
endif()

#------------------------------------------------------------------------------
# OPENCL+CLFFT
#------------------------------------------------------------------------------

find_package(OpenCL)
if(OpenCL_FOUND AND GEARSHIFFT_CLFFT)
  find_package(clFFT)
  if(CLFFT_FOUND)
    include_directories(${OpenCL_INCLUDE_DIRS} ${CLFFT_INCLUDE_DIRS} )
    list(APPEND FFTLIBS "clfft") # register fft lib client for compilation
    string(REPLACE ";" " " local_clfft_libraries "${CLFFT_LIBRARIES}")
    message(">> clFFT -> " ${local_clfft_libraries})
  endif()
endif()

if(NOT CLFFT_FOUND)
  message("<< clFFT benchmark disabled.")
endif()

#------------------------------------------------------------------------------
# ROCM+HCFFT
#------------------------------------------------------------------------------

find_package(hcFFT)
if(HCFFT_FOUND AND GEARSHIFFT_HCFFT)
  include_directories(${HCFFT_INCLUDE_DIRS})
  list(APPEND FFTLIBS "hcfft") # register fft lib client for compilation
  message(">> hcFFT -> " ${HCFFT_LIBRARIES} " " ${HCFFT_INCLUDE_DIRS})
  message(">> Warning: no runner for hcfft implemented yet")
endif()

if(NOT HCFFT_FOUND)
  message("<< hcFFT benchmark disabled.")
endif()

#------------------------------------------------------------------------------
# FFTW
#------------------------------------------------------------------------------

find_package(FFTW COMPONENTS float double)
if(FFTW_FOUND AND GEARSHIFFT_FFTW)
  include_directories(${FFTW_INCLUDE_DIR})
  link_directories(${FFTW_LIBRARY_DIR})
  list(APPEND FFTLIBS fftw) # register fft lib client for compilation
  find_package(OpenMP)

  foreach(_LIBSTEM IN LISTS FFTW_SERIAL_LIBS)
    list(APPEND FFTW_LIBS_TO_USE ${_LIBSTEM})
  endforeach()

  if(GEARSHIFFT_FFTW_OPENMP AND OPENMP_FOUND AND "${FFTW_LIBRARIES}" MATCHES ".*_omp.*")
    foreach(_LIBSTEM IN LISTS FFTW_OPENMP_LIBS)
      list(APPEND FFTW_LIBS_TO_USE ${_LIBSTEM})
    endforeach()
  endif()

  if(GEARSHIFFT_FFTW_PTHREADS AND "${FFTW_LIBRARIES}" MATCHES ".*_threads.*")
    foreach(_LIBSTEM IN LISTS FFTW_THREADS_LIBS)
      list(APPEND FFTW_LIBS_TO_USE ${_LIBSTEM})
    endforeach()

  endif()

  set(FFTW_LIBRARIES ${FFTW_LIBS_TO_USE})
  message(">> FFTW -> ${FFTW_LIBRARIES} ${FFTW_INCLUDES}")
else()
  message("<< FFTW benchmark disabled.")
endif()


if(NOT FFTLIBS)
  message(FATAL_ERROR ">> No FFT library for benchmark found !!!")
endif()

enable_testing()

add_subdirectory(src)
add_subdirectory(test)
