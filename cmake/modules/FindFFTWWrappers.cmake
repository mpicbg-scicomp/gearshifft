# - Find the MKL libraries for FFTWWrappers
#
# Usage:
#   find_package(FFTWWrappers [REQUIRED] [QUIET])
#
# It sets the following variables:
#   FFTWWrappers_FOUND             ... true if FFTWWrappers/MKL is found on the system
#   FFTWWrappers_GNU_LIBRARIES     ... list of library that were build with the GNU binary layout
#   FFTWWrappers_INTEL_LIBRARIES   ... list of library that were build with the intel binary layout
#   FFTWWrappers_MSVS_LIBRARIES    ... list of library that were build with the MSVS binary layout
#   FFTWWrappers_LIBRARIES         ... list of library identifiers that were found (will be filled with serial/openmp/pthreads enabled file names if present, only stubs will be filled in not full paths)
#   FFTWWrappers_MKL_LIBRARIES     ... list of library in the MKL that need to be linked to
#   FFTWWrappers_MKL_INCLUDE_DIR   ... folder containing fftw3.h
#   FFTWWrappers_LIBRARY_DIR       ... library directory
#
# The following variables will be checked by the function
#   FFTWWrappers_USE_STATIC_LIBS   ... if true, only static libraries are found
#   FFTWWrappers_ROOT              ... if set, the libraries are exclusively searched
#                                      under this path
#   MKLROOT                        ... take the MKL libraries from here
#


#Check whether to search static or dynamic libs
set( CMAKE_FIND_LIBRARY_SUFFIXES_SAV ${CMAKE_FIND_LIBRARY_SUFFIXES} )

if( FFTWWrappers_USE_STATIC_LIBS ) # find static libs only
  set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX} )
endif()

# If environment variable FFTWWrappers_ROOT is defined, it has the same effect as the cmake variable
if( NOT FFTWWrappers_ROOT AND DEFINED ENV{FFTWWrappers_ROOT} )
  if( EXISTS "$ENV{FFTWWrappers_ROOT}/" )
    set( FFTWWrappers_ROOT $ENV{FFTWWrappers_ROOT} )
  else()
    message( "FFTWWrappers_ROOT set to ${FFTWWrappers_ROOT}, but folder does not exist")
  endif()
endif()

# If environment variable MKL_ROOT is defined, it has the same effect as the cmake variable
if( NOT MKL_ROOT AND DEFINED ENV{MKLROOT})
  if( EXISTS "$ENV{MKLROOT}/" )
    set( MKL_ROOT $ENV{MKLROOT} )
  else()
    message( "MKL_ROOT set to $ENV{MKLROOT}, but folder does not exist")
  endif()
endif()

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(MKL_INTERFACE_LIBDIR "intel64")
  set(MKL_INTERFACE_LIBNAME "mkl_intel_lp64") # index integers might be 32bit though
else()
  set(MKL_INTERFACE_LIBDIR "ia32")
  set(MKL_INTERFACE_LIBNAME "mkl_intel")
endif()

################################### FFTWWrappers related ##################################
# see: http://geco.mines.edu/files/userguides/techReports/mklWrappers/
set(_LIBSTUB_GNU "fftw3xc_gnu")
set(_LIBSTUB_INTEL "fftw3xc_intel")
set(_LIBSTUB_MSVS "fftw3xc_msvs")

#initialize library variables
find_library(
  FFTWWrappers_GNU_LIBRARIES
  NAMES ${_LIBSTUB_GNU}
  PATHS ${FFTWWrappers_ROOT}
  PATH_SUFFIXES "lib" "lib64"
  NO_DEFAULT_PATH
  )
find_library(
  FFTWWrappers_GNU_LIBRARIES
  NAMES ${_LIBSTUB_GNU}
  PATHS ${MKL_ROOT}
  PATH_SUFFIXES "lib" "lib/${MKL_INTERFACE_LIBDIR}_lin" "lib/${MKL_INTERFACE_LIBDIR}_mac" "lib/${MKL_INTERFACE_LIBDIR}"
  NO_DEFAULT_PATH
  )
find_library(
  FFTWWrappers_GNU_LIBRARIES
  NAMES ${_LIBSTUB_GNU}
  )

if(EXISTS ${FFTWWrappers_GNU_LIBRARIES})
  get_filename_component(FFTWWrappers_LIBRARY_DIR ${FFTWWrappers_GNU_LIBRARIES} DIRECTORY)
endif()

find_library(
  FFTWWrappers_INTEL_LIBRARIES
  NAMES ${_LIBSTUB_INTEL}
  PATHS ${FFTWWrappers_ROOT}
  PATH_SUFFIXES "lib" "lib64"
  NO_DEFAULT_PATH
  )
find_library(
  FFTWWrappers_INTEL_LIBRARIES
  NAMES ${_LIBSTUB_INTEL}
  PATHS ${MKL_ROOT}
  PATH_SUFFIXES "lib" "lib/${MKL_INTERFACE_LIBDIR}_lin" "lib/${MKL_INTERFACE_LIBDIR}_mac" "lib/${MKL_INTERFACE_LIBDIR}_win" "lib/${MKL_INTERFACE_LIBDIR}"
  NO_DEFAULT_PATH
  )
find_library(
  FFTWWrappers_INTEL_LIBRARIES
  NAMES ${_LIBSTUB_INTEL}
  )

if(EXISTS ${FFTWWrappers_INTEL_LIBRARIES})
  get_filename_component(FFTWWrappers_LIBRARY_DIR ${FFTWWrappers_INTEL_LIBRARIES} DIRECTORY)
endif()

find_library(
  FFTWWrappers_MSVS_LIBRARIES
  NAMES ${_LIBSTUB_MSVS}
  PATHS ${FFTWWrappers_ROOT}
  PATH_SUFFIXES "lib" "lib64"
  NO_DEFAULT_PATH
  )
find_library(
  FFTWWrappers_MSVS_LIBRARIES
  NAMES ${_LIBSTUB_MSVS}
  PATHS ${MKL_ROOT}
  PATH_SUFFIXES "lib/${MKL_INTERFACE_LIBDIR}_win" "lib/${MKL_INTERFACE_LIBDIR}"
  NO_DEFAULT_PATH
  )
find_library(
  FFTWWrappers_MSVS_LIBRARIES
  NAMES ${_LIBSTUB_MSVS}
  )

if(EXISTS ${FFTWWrappers_MSVS_LIBRARIES})
  get_filename_component(FFTWWrappers_LIBRARY_DIR ${FFTWWrappers_MSVS_LIBRARIES} DIRECTORY)
endif()

######################################### MKL related #####################################

find_file(
  FFTWWrapper_include_file
  NAMES "fftw3.h"
  PATHS ${MKL_ROOT} ${MKL_ROOT}/include/fftw
  PATH_SUFFIXES "include" "include/fftw"
  NO_DEFAULT_PATH
  )

if(EXISTS ${FFTWWrapper_include_file})
  get_filename_component(FFTWWrappers_MKL_INCLUDE_DIR ${FFTWWrapper_include_file} DIRECTORY)
else()
  if(NOT FFTWWrappers_FIND_QUIETLY)
    message( "FFTWWrappers was not able to find fftw3.h in ${MKL_ROOT}")
  endif()
endif()

find_library(
  MKL_INTEL
  NAMES ${MKL_INTERFACE_LIBNAME}
  PATHS ${MKL_ROOT}
  PATH_SUFFIXES "lib" "lib/${MKL_INTERFACE_LIBDIR}_lin" "lib/${MKL_INTERFACE_LIBDIR}_mac" "lib/${MKL_INTERFACE_LIBDIR}_win" "lib/${MKL_INTERFACE_LIBDIR}"
  NO_DEFAULT_PATH
  )

if(EXISTS ${MKL_INTEL})
  list(APPEND FFTWWrappers_MKL_LIBRARIES "${MKL_INTEL}")
else()
  if(NOT FFTWWrappers_FIND_QUIETLY)
    message( "FFTWWrappers was not able to find ${MKL_INTERFACE_LIBNAME} in ${MKL_ROOT}")
  endif()
endif()

find_library(
  MKL_INTEL_THREAD
  NAMES "mkl_intel_thread"
  PATHS ${MKL_ROOT}
  PATH_SUFFIXES "lib" "lib/${MKL_INTERFACE_LIBDIR}_lin" "lib/${MKL_INTERFACE_LIBDIR}_mac" "lib/${MKL_INTERFACE_LIBDIR}_win" "lib/${MKL_INTERFACE_LIBDIR}"
  NO_DEFAULT_PATH
  )

if(EXISTS ${MKL_INTEL_THREAD})
  list(APPEND FFTWWrappers_MKL_LIBRARIES "${MKL_INTEL_THREAD}")
else()
  if(NOT FFTWWrappers_FIND_QUIETLY)
    message( "FFTWWrappers was not able to find mkl_intel_thread in ${MKL_ROOT}")
  endif()
endif()

find_library(
  MKL_CORE
  NAMES "mkl_core"
  PATHS ${MKL_ROOT}
  PATH_SUFFIXES "lib" "lib/${MKL_INTERFACE_LIBDIR}_lin" "lib/${MKL_INTERFACE_LIBDIR}_mac" "lib/${MKL_INTERFACE_LIBDIR}_win" "lib/${MKL_INTERFACE_LIBDIR}"
  NO_DEFAULT_PATH
  )

if(EXISTS ${MKL_CORE})
  list(APPEND FFTWWrappers_MKL_LIBRARIES "${MKL_CORE}")
else()
  if(NOT FFTWWrappers_FIND_QUIETLY)
    message("FFTWWrappers was not able to find mkl_core in ${MKL_ROOT}")
  endif()
endif()



find_library(
  MKL_IOMP5
  NAMES "iomp5" "iomp5md"
  PATHS ${MKL_ROOT} ${MKL_ROOT}/../compiler ${MKL_ROOT}/../tbb
  PATH_SUFFIXES "lib" "lib/${MKL_INTERFACE_LIBDIR}_lin" "lib/${MKL_INTERFACE_LIBDIR}_mac" "lib/${MKL_INTERFACE_LIBDIR}_win" "lib/${MKL_INTERFACE_LIBDIR}" "lib/${MKL_INTERFACE_LIBDIR}/gcc4.7" "lib/${MKL_INTERFACE_LIBDIR}/gcc4.4" "lib/${MKL_INTERFACE_LIBDIR}/gcc4.1"
  NO_DEFAULT_PATH
  )

if(EXISTS ${MKL_IOMP5})
  list(APPEND FFTWWrappers_MKL_LIBRARIES "${MKL_IOMP5}")
else()
  if(NOT FFTWWrappers_FIND_QUIETLY)
    message("FFTWWrappers was not able to find iomp5")
  endif()
endif()


if(NOT FFTWWrappers_FIND_QUIETLY)
  message("++ FindFFTWWrappers")
  message("++ FFTWWrappers_ROOT             : ${FFTWWrappers_ROOT}")
  message("++ MKL_ROOT                      : ${MKL_ROOT}")
  message("++ FFTWWrappers_GNU_LIBRARIES    : ${FFTWWrappers_GNU_LIBRARIES}")
  message("++ FFTWWrappers_INTEL_LIBRARIES  : ${FFTWWrappers_INTEL_LIBRARIES}")
  message("++ FFTWWrappers_MSVS_LIBRARIES   : ${FFTWWrappers_MSVS_LIBRARIES}")
  message("++ FFTWWrappers_LIBRARY_DIR      : ${FFTWWrappers_LIBRARY_DIR}")
  message("++ FFTWWrappers_MKL_LIBRARIES    : ${FFTWWrappers_MKL_LIBRARIES}")
  message("++ FFTWWrappers_MKL_INCLUDE_DIR  : ${FFTWWrappers_MKL_INCLUDE_DIR}")
endif()


######################################### EXPORTS #####################################
#revert CMAKE_FIND_LIBRARY_SUFFIXES
set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV} )

include(FindPackageHandleStandardArgs)

if(FFTWWrappers_GNU_LIBRARIES)
  find_package_handle_standard_args(FFTWWrappers
    REQUIRED_VARS FFTWWrappers_GNU_LIBRARIES
    REQUIRED_VARS FFTWWrappers_MKL_LIBRARIES
    REQUIRED_VARS FFTWWrappers_MKL_INCLUDE_DIR
    )
  set(FFTWWrappers_LIBRARIES "${FFTWWrappers_GNU_LIBRARIES}")
elseif(FFTWWrappers_INTEL_LIBRARIES)
  find_package_handle_standard_args(FFTWWrappers
    REQUIRED_VARS FFTWWrappers_INTEL_LIBRARIES
    REQUIRED_VARS FFTWWrappers_MKL_LIBRARIES
    REQUIRED_VARS FFTWWrappers_MKL_INCLUDE_DIR
    )
  set(FFTWWrappers_LIBRARIES "${FFTWWrappers_INTEL_LIBRARIES}")
elseif(FFTWWrappers_MSVS_LIBRARIES)
  find_package_handle_standard_args(FFTWWrappers
    REQUIRED_VARS FFTWWrappers_MSVS_LIBRARIES
    REQUIRED_VARS FFTWWrappers_MKL_LIBRARIES
    REQUIRED_VARS FFTWWrappers_MKL_INCLUDE_DIR
    )
  set(FFTWWrappers_LIBRARIES "${FFTWWrappers_MSVS_LIBRARIES}")
endif()


mark_as_advanced(
  FFTWWrappers_LIBRARIES
  FFTWWrappers_GNU_LIBRARIES
  FFTWWrappers_INTEL_LIBRARIES
  FFTWWrappers_MSVS_LIBRARIES
  FFTWWrappers_LIBRARY_DIR
  FFTWWrappers_MKL_LIBRARIES
  FFTWWrappers_MKL_LIBRARY_DIR
  FFTWWrappers_MKL_INCLUDE_DIR
  )
