cmake_minimum_required(VERSION 3.7)

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

set(GEARSHIFFT_ROOT "${CMAKE_SOURCE_DIR}/../..")

# gearshifft cmake options (requires GEARSHIFFT_ROOT)
include(${GEARSHIFFT_ROOT}/cmake/gearshifft-options.cmake)
# superbuild is aimed for static build (otherwise env must be set accordingly, so the libs are found)
set(GEARSHIFFT_USE_STATIC_LIBS ON CACHE BOOL "" FORCE)

include(ExternalProject)

#------------------------------------------------------------------------------
# Boost
#------------------------------------------------------------------------------
option(GEARSHIFFT_EXT_USE_SYSTEM_Boost "Use system libraries for Boost" OFF )

if(NOT GEARSHIFFT_EXT_USE_SYSTEM_Boost)
  # looks in /gearshifft/ext/boost, and may download+build it
  # sets BOOST_ROOT
  include(Boost.cmake)
else()
  set(BOOST_ROOT "")
endif()

#------------------------------------------------------------------------------
# FFTW
#------------------------------------------------------------------------------
option(GEARSHIFFT_EXT_USE_SYSTEM_FFTW "Use system libraries for FFTW" OFF )

if(NOT GEARSHIFFT_EXT_USE_SYSTEM_FFTW)
  # looks in /gearshifft/ext/fftw, and may download+build it
  # sets FFTW_ROOT
  include(FFTW.cmake)
else()
  set(FFTW_ROOT "")
endif()

#------------------------------------------------------------------------------
# clFFT
#------------------------------------------------------------------------------
option(GEARSHIFFT_EXT_USE_SYSTEM_CLFFT "Use system libraries for clFFT" OFF )

if(NOT GEARSHIFFT_EXT_USE_SYSTEM_CLFFT)
  # looks in /gearshifft/ext/clfft, and may download+build it
  # sets CLFFT_ROOT
  include(clfft.cmake)
else()
  set(CLFFT_ROOT "")
endif()

#------------------------------------------------------------------------------
# gearshifft
#------------------------------------------------------------------------------

include(${GEARSHIFFT_ROOT}/cmake/get-gearshifft-options.cmake)
get_gearshifft_options(cmake_pass_args ";-D") # ; to convert to a list

# to get BUILD_TESTING option, which is just forwarded to gearshifft
include(CTest)

ExternalProject_Add( gearshifft
  DEPENDS ${gearshifft_DEPENDENCIES}
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..
  BINARY_DIR gearshifft-build
  CMAKE_GENERATOR ${EP_CMAKE_GENERATOR}
  CMAKE_ARGS
  ${cmake_pass_args}
  -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
  -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
  -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  -DBUILD_TESTING:BOOL=${BUILD_TESTING}
  -DBOOST_ROOT:PATH=${BOOST_ROOT}
  -DFFTW_ROOT:PATH=${FFTW_ROOT}
  -DCLFFT_ROOT:PATH=${CLFFT_ROOT}
  INSTALL_COMMAND ""
  )
