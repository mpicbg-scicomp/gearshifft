## based on: https://github.com/statismo/statismo/tree/master/superbuild

#---------------------------------------------------------------------------
# Get and build clfft, if it is not already installed in /gearshifft/ext

set(GEARSHIFFT_EXT_CLFFT_VERSION "v2.12.2" CACHE STRING "clFFT version to be built.")
set_property(CACHE GEARSHIFFT_EXT_CLFFT_VERSION PROPERTY STRINGS "v2.12.2;develop")
set(CLFFT_ROOT ${GEARSHIFFT_SUPERBUILD_EXT_DIR}/clfft/ CACHE PATH "'clFFT' root directory")

find_path(CLFFT_INCLUDE_DIR
  NAMES clFFT.h
  PATHS ${CLFFT_ROOT}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)

find_library(CLFFT_LIBRARY_DIR
  NAMES clFFT
  PATHS ${CLFFT_ROOT}
  PATH_SUFFIXES lib lib64 lib/import lib64/import
  NO_DEFAULT_PATH)

if((NOT CLFFT_INCLUDE_DIR) OR (NOT EXISTS ${CLFFT_INCLUDE_DIR})
    OR
    (NOT CLFFT_LIBRARY_DIR) OR (NOT EXISTS ${CLFFT_LIBRARY_DIR}))

  message("'clfft' library could not be found, so [make] will download and build it (requires OpenCL).")


  find_package(Git)
  if(NOT GIT_FOUND)
    message(ERROR "Cannot find git. git is required for Superbuild")
  endif()

  option( USE_GIT_PROTOCOL "If behind a firewall turn this off to use http instead." ON)

  set(git_protocol "git")
  if(NOT USE_GIT_PROTOCOL)
    set(git_protocol "http")
  endif()

  if(GEARSHIFFT_USE_STATIC_LIBS)
    set(BUILD_SHARED_LIBS OFF)
  else()
    set(BUILD_SHARED_LIBS ON)
  endif()

  ExternalProject_Add(clfft
    GIT_REPOSITORY ${git_protocol}://github.com/clMathLibraries/clFFT.git
    GIT_TAG ${GEARSHIFFT_EXT_CLFFT_VERSION}
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    PREFIX ${GEARSHIFFT_SUPERBUILD_EXT_DIR}
    SOURCE_DIR ${GEARSHIFFT_SUPERBUILD_EXT_SOURCE_DIR}/clfft
    SOURCE_SUBDIR src
    BINARY_DIR ${GEARSHIFFT_SUPERBUILD_EXT_SOURCE_DIR}/clfft-build
    CMAKE_ARGS
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DBUILD_CALLBACK_CLIENT:BOOL=OFF
    -DBUILD_CLIENT:BOOL=ON
    -DBUILD_EXAMPLES:BOOL=OFF
    -DBUILD_TEST:BOOL=OFF
    -DBUILD64=ON
    -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${CLFFT_ROOT}
    )
  list(APPEND gearshifft_DEPENDENCIES clfft)

endif()
