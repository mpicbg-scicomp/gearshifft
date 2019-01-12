## based on: https://github.com/statismo/statismo/tree/master/superbuild

#---------------------------------------------------------------------------
# Get and build rocfft, if it is not already installed in /gearshifft/ext

set(GEARSHIFFT_EXT_ROCFFT_VERSION "v0.8.8" CACHE STRING "rocFFT version to be built.")
set_property(CACHE GEARSHIFFT_EXT_ROCFFT_VERSION PROPERTY STRINGS "v0.8.8;develop")
set(ROCFFT_ROOT ${GEARSHIFFT_EXT_DIR}/rocfft/ CACHE PATH "'rocFFT' root directory (rocFFT depends on ROCM+HIP installation)")

find_path(ROCFFT_INCLUDE_DIR
  NAMES rocfft.h
  PATHS ${ROCFFT_ROOT}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)

find_library(ROCFFT_LIBRARY_DIR
  NAMES rocfft
  PATHS ${ROCFFT_ROOT}
  PATH_SUFFIXES lib lib64 lib/import lib64/import
  NO_DEFAULT_PATH)

if(NOT CMAKE_CXX_COMPILER MATCHES "hcc$")
  message(WARNING "rocFFT is currently only supported on HCC platform.")

elseif((NOT ROCFFT_INCLUDE_DIR) OR (NOT EXISTS ${ROCFFT_INCLUDE_DIR})
    OR
    (NOT ROCFFT_LIBRARY_DIR) OR (NOT EXISTS ${ROCFFT_LIBRARY_DIR})
    )

  message("'rocfft' library could not be found, so [make] will download and build it.")

  set(HIP_PLATFORM "hcc")

  if(GEARSHIFFT_USE_STATIC_LIBS)
    set(BUILD_SHARED_LIBS OFF)
  else()
    set(BUILD_SHARED_LIBS ON)
  endif()

  find_package(Git)
  if(NOT GIT_FOUND)
    message(ERROR "Cannot find git. git is required for Superbuild")
  endif()

  option( USE_GIT_PROTOCOL "If behind a firewall turn this off to use http instead." ON)

  set(git_protocol "git")
  if(NOT USE_GIT_PROTOCOL)
    set(git_protocol "http")
  endif()

  find_program(PROGRAM_HCC NAMES hcc)
  if(PROGRAM_HCC)
    ExternalProject_Add(rocfft
      GIT_REPOSITORY ${git_protocol}://github.com/ROCmSoftwarePlatform/rocFFT.git
      GIT_TAG ${GEARSHIFFT_EXT_ROCFFT_VERSION}
      UPDATE_COMMAND ""
      PATCH_COMMAND ""
      SOURCE_DIR ${GEARSHIFFT_EXT_SOURCE_DIR}/rocFFT
      # SOURCE_SUBDIR library/src
      BINARY_DIR ${GEARSHIFFT_EXT_SOURCE_DIR}/rocfft-build
      CMAKE_ARGS
      -DHIP_PLATFORM=${HIP_PLATFORM}
      -DBUILD_CLIENTS_SAMPLES=OFF
      -DBUILD_CLIENTS_TESTS=OFF
      -DBUILD_CLIENTS_BENCHMARKS=OFF
      -DBUILD_CLIENTS_SELFTEST=OFF
      -DBUILD_VERBOSE=${GEARSHIFFT_VERBOSE}
      -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${ROCFFT_ROOT}
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER:FILEPATH=${PROGRAM_HCC}
      #    -DCMAKE_CXX_COMPILER:STRING=${HIPCC_PATH} # rocFFT CMakeLists.txt wants that as a workaround to get nvcc, but it does not work as it still invokes hcc/rocm stuff :/
      )
    list(APPEND gearshifft_DEPENDENCIES rocfft)
  endif()
endif()
