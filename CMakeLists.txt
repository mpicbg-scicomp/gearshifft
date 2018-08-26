cmake_minimum_required(VERSION 3.7) # better language support, more features
# https://cliutils.gitlab.io/modern-cmake/chapters/intro/newcmake.html
# cmake configurations are inspired by VexCL
# See: https://github.com/ddemidov/vexcl/

if(${CMAKE_VERSION} VERSION_LESS 3.12)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.12)
endif()

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type")
endif()
#-------------------------------------------------------------------------------

project(gearshifft
  VERSION 0.4.0
  LANGUAGES C CXX)
#-------------------------------------------------------------------------------

# guard against in-source builds and bad build-type strings
include(cmake/safeguards.cmake)
#-------------------------------------------------------------------------------

# cmake module path with additional finders
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/)
#-------------------------------------------------------------------------------
# gearshifft cmake options

option(GEARSHIFFT_VERBOSE "Verbose output during build generation." ON)
option(GEARSHIFFT_CUFFT "Compile gearshifft_cufft if possible" ON)
option(GEARSHIFFT_CLFFT "Compile gearshifft_clfft if possible" ON)
option(GEARSHIFFT_FFTW  "Compile gearshifft_fftw if possible" ON)
option(GEARSHIFFT_FFTW_OPENMP "Use OpenMP parallel FFTW libraries if found" ON)
option(GEARSHIFFT_FFTW_PTHREADS "Use pthreads parallel FFTW libraries if found" OFF)
option(GEARSHIFFT_HCFFT "< Not implemented yet >" OFF)

set(GEARSHIFFT_CXX11_ABI "1" CACHE STRING "Enable _GLIBCXX_USE_CXX11_ABI in GCC 5.0+")
set_property(CACHE GEARSHIFFT_CXX11_ABI PROPERTY STRINGS "0;1")

set(GEARSHIFFT_FLOAT16_SUPPORT "0" CACHE STRING "Enable float16 data type (cufft and ComputeCapability 5.3+ only)")
set_property(CACHE GEARSHIFFT_FLOAT16_SUPPORT PROPERTY STRINGS "0;1")

set(GEARSHIFFT_NUMBER_WARM_RUNS "10" CACHE STRING "Number of repetitions of an FFT benchmark after a warmup.")
set(GEARSHIFFT_NUMBER_WARMUPS "2" CACHE STRING "Number of warmups of an FFT benchmark.")
set(GEARSHIFFT_ERROR_BOUND "-1" CACHE STRING "Error-bound for FFT benchmarks (<0 for dynamic error bound).")

set(GEARSHIFFT_INSTALL_CONFIG_PATH "${CMAKE_INSTALL_PREFIX}/share/gearshifft/" CACHE STRING "Default install path of config files.")

#-------------------------------------------------------------------------------

if(GEARSHIFFT_HCFFT)
  message(FATAL_ERROR "HCFFT is not implemented yet.")
endif()

# half-code for half precision data type
if(GEARSHIFFT_FLOAT16_SUPPORT)
  include(half)
endif()

if(GEARSHIFFT_VERBOSE)
  set(_QUIET "")
else()
  set(_QUIET "QUIET")
endif()


#------------------------------------------------------------------------------
# Boost
#------------------------------------------------------------------------------

option(Boost_USE_STATIC_LIBS "Use static versions of Boost libraries" OFF)
if(WIN32)
  set(Boost_USE_STATIC_LIBS ON)
endif()

find_package(Boost 1.59 REQUIRED ${_QUIET} COMPONENTS
  system
  unit_test_framework
  program_options
  )

#------------------------------------------------------------------------------
# Threads
#------------------------------------------------------------------------------

# find_package(Threads) # TODO: not needed by gearshifft, so find dependency target

#----------------------------------------------------------------------------
# Generic target
#----------------------------------------------------------------------------

add_library(Common INTERFACE)
add_library(gearshifft::Common ALIAS Common)

# setting language requirements for all targets
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# TODO: in cmake 3.8+ use target_compile_features(myTarget PUBLIC cxx_std_11)

target_include_directories(Common INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/inc>
#  $<INSTALL_INTERFACE:include> # include dirs are not installed
#  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> # include dirs are not installed
  )

# do not export this interface (only for internal usage requirements)
target_compile_options(Common INTERFACE
  # g++
  $<$<CXX_COMPILER_ID:GNU>:$<BUILD_INTERFACE:-Wall>>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-missing-braces>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated-declarations>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-ignored-attributes>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-local-typedefs>
  # Clang
  $<$<CXX_COMPILER_ID:Clang>:$<BUILD_INTERFACE:-Wall>>
  $<$<CXX_COMPILER_ID:Clang>:-Wno-missing-braces>
  $<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-declarations>
  $<$<CXX_COMPILER_ID:Clang>:-Wno-ignored-attributes>
  # MSVC
  $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
  $<$<CXX_COMPILER_ID:MSVC>:/wd4003>
  $<$<CXX_COMPILER_ID:MSVC>:/wd4996>
  )

# gearshifft version.
# As this information is built as own library, targets only have to relink,
# when version changes.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gearshifft_version.cpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/gearshifft_version.cpp"
  @ONLY)

add_library(gearshifft_version STATIC
  "${CMAKE_CURRENT_BINARY_DIR}/gearshifft_version.cpp")

target_link_libraries(gearshifft_version PUBLIC Common)

# continue with Common interface
target_compile_definitions(Common INTERFACE
  $<$<CXX_COMPILER_ID:GNU>:_GLIBCXX_USE_CXX11_ABI=${GEARSHIFFT_CXX11_ABI}>
  GEARSHIFFT_NUMBER_WARM_RUNS=${GEARSHIFFT_NUMBER_WARM_RUNS}
  GEARSHIFFT_NUMBER_WARMUPS=${GEARSHIFFT_NUMBER_WARMUPS}
  GEARSHIFFT_ERROR_BOUND=${GEARSHIFFT_ERROR_BOUND}
  $<BUILD_INTERFACE:GEARSHIFFT_INSTALL_CONFIG_FILE=${CMAKE_SOURCE_DIR}/share/gearshifft/extents.conf>
  $<INSTALL_INTERFACE:GEARSHIFFT_INSTALL_CONFIG_FILE=${GEARSHIFFT_INSTALL_CONFIG_PATH}/extents.conf>
  GEARSHIFFT_FLOAT16_SUPPORT=${GEARSHIFFT_FLOAT16_SUPPORT}
  )
#TODO: check paths

target_link_libraries(Common INTERFACE
  gearshifft_version
  Boost::system
  Boost::unit_test_framework
  Boost::program_options
  #Threads::threads
  )

#------------------------------------------------------------------------------
# Find gearshifft back-ends
#------------------------------------------------------------------------------

# CUDA+CUFFT
if(GEARSHIFFT_CUFFT)
  find_package(CUDA ${_QUIET}) # no imported targets, will be language in cmake >= 3.10
  if(CUDA_FOUND)

    add_library(CUFFT INTERFACE)
    add_library(gearshifft::CUFFT ALIAS CUFFT)

    target_include_directories(CUFFT INTERFACE ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(CUFFT INTERFACE CUFFT_ENABLED)
    target_link_libraries(CUFFT INTERFACE
      Common
      ${CUDA_LIBRARIES}
      ${CUDA_CUFFT_LIBRARIES})

    if(GEARSHIFFT_FLOAT16_SUPPORT)
      add_dependencies(CUFFT half)
    endif()

    message(STATUS " gearshifft::CUFFT enabled.")
  endif()
endif()

#------------------------------------------------------------------------------
# OPENCL+CLFFT
#------------------------------------------------------------------------------

if(GEARSHIFFT_CLFFT)
  find_package(OpenCL ${_QUIET}) # cmake >= 3.7 provides imported targets
  if(OpenCL_FOUND)
    find_package(clFFT)

    add_library(CLFFT INTERFACE)
    add_library(gearshifft::CLFFT ALIAS CLFFT)

    target_include_directories(CLFFT INTERFACE ${CLFFT_INCLUDE_DIRS})
    target_compile_definitions(CLFFT INTERFACE CLFFT_ENABLED)
    target_link_libraries(CLFFT INTERFACE
      Common
      ${CLFFT_LIBRARIES}
      OpenCL::OpenCL
      )

    target_compile_options(Common INTERFACE
      $<$<CXX_COMPILER_ID:GNU>:-Wno-catch-value>
      )

    message(STATUS " gearshifft::CLFFT enabled")
  endif()
endif()

#------------------------------------------------------------------------------
# ROCM+HCFFT
#------------------------------------------------------------------------------

#find_package(hcFFT)

#------------------------------------------------------------------------------
# FFTW
#------------------------------------------------------------------------------
if(GEARSHIFFT_FFTW)
  find_package(FFTW ${_QUIET} COMPONENTS float double) #TODO: only optional float/double in benchmark

  if(FFTW_FOUND)

    add_library(FFTW INTERFACE)
    add_library(gearshifft::FFTW ALIAS FFTW)

    set(GEARSHIFFT_FFTW_THREADS 0)

    if(GEARSHIFFT_FFTW_OPENMP)

      if(FFTW_OPENMP_LIBS)

        find_package(OpenMP ${_QUIET})
        # Have to check several OPENMP_FOUND due to bug in
        # one version of CMake and the docs (fixed in patch release)
        # OpenMP is missing on macOS llvm default, for example
        if(OpenMP_FOUND OR OPENMP_FOUND OR OpenMP_CXX_FOUND)
          set(GEARSHIFFT_FFTW_THREADS 1)
          # CMake 3.9 FindOpenMP allows correct linking with Clang in more cases
          if(TARGET OpenMP::OpenMP_CXX)
            target_link_libraries(FFTW INTERFACE OpenMP::OpenMP_CXX)
          else()
            # Clang may need -fopenmp=libiomp5 instead, can't be detected here without CMake 3.9
            target_link_libraries(FFTW INTERFACE
              $<$<CXX_COMPILER_ID:GNU>:${OpenMP_CXX_FLAGS}>
              $<$<CXX_COMPILER_ID:Clang>:${OpenMP_CXX_FLAGS}>
              $<$<CXX_COMPILER_ID:Intel>:${OpenMP_CXX_FLAGS}>
              )
            target_compile_options(FFTW INTERFACE ${OpenMP_CXX_FLAGS})
          endif()
        else()

          message(WARNING "OpenMP not found")
          set(GEARSHIFFT_FFTW_OPENMP OFF)
        endif() # OpenMP_FOUND
      else()

        message(WARNING "FFTW libraries for OpenMP not found")
        set(GEARSHIFFT_FFTW_OPENMP OFF)
      endif() # FFTW_OPENMP_LIBS
    endif() # GEARSHIFFT_FFTW_OPENMP

    if(GEARSHIFFT_FFTW_PTHREADS)
      if(FFTW_THREADS_LIBS)
        find_package(Threads)
        set(GEARSHIFFT_FFTW_THREADS 1)
      else()
        message(WARNING "FFTW libraries for pthreads not found")
        set(GEARSHIFFT_FFTW_PTHREADS OFF)
      endif()
    endif()

    if(GEARSHIFFT_FFTW_PTHREADS AND GEARSHIFFT_FFTW_OPENMP)
      message(FATAL_ERROR "FFTW: Cannot use both OpenMP and pthreads, please disable one")
    endif()

    if(GEARSHIFFT_FFTW_PTHREADS
        OR GEARSHIFFT_FFTW_OPENMP
        OR FFTW_SERIAL_LIBS)

      target_include_directories(FFTW INTERFACE ${FFTW_INCLUDE_DIR})
      target_compile_definitions(FFTW INTERFACE
        FFTW_ENABLED
        GEARSHIFFT_FFTW_THREADS=${GEARSHIFFT_FFTW_THREADS})
      target_link_libraries(FFTW INTERFACE
        Common
        $<$<STREQUAL:${GEARSHIFFT_FFTW_OPENMP},ON>:${FFTW_OPENMP_LIBS}>
        $<$<STREQUAL:${GEARSHIFFT_FFTW_PTHREADS},ON>:${FFTW_THREADS_LIBS}>
        ${FFTW_SERIAL_LIBS}
        $<$<BOOL:${Threads_FOUND}>:Threads::Threads>
        )

      message(STATUS " gearshifft::FFTW enabled.")
    endif()
  endif()
endif()

if(NOT TARGET gearshifft::CUFFT
    AND NOT TARGET gearshifft::CLFFT
    AND NOT TARGET gearshifft::FFTW)
  message(WARNING "No FFT backend found.")
endif()

#-------------------------------------------------------------------------------

add_subdirectory(gearshifft)

include(CTest) # defines: BUILD_TESTING, default is ON

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

#-------------------------------------------------------------------------------

# installs config extents files

install(
  DIRECTORY share/gearshifft/
  DESTINATION ${GEARSHIFFT_INSTALL_CONFIG_PATH}
  FILES_MATCHING
  PATTERN *.conf
  )
install(
  FILES README.md LICENSE
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  )

#-------------------------------------------------------------------------------

# packaging
# check that we are in top-level CMakeLists
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_subdirectory(packaging)
  # include(CPack) will happen in here
endif()
