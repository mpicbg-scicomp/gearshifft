# - Find the FFTW library
#
# Usage:
#   find_package(FFTWWrappers [REQUIRED] [QUIET])
#
# It sets the following variables:
#   FFTWWrappers_FOUND          ... true if fftw is found on the system
#   FFTWWrappers_LIBRARIES      ... list of library identifiers that were found (will be filled with serial/openmp/pthreads enabled file names if present, only stubs will be filled in not full paths)
#   FFTWWrappers_MKL_LIBRARIES  ... list of library in the MKL that need to be linked to (intel64)
#   FFTWWrappers_MKL_INCLUDE_DIR  ... folder containing fftw3.h
#   FFTWWrappers_MKL_LIBRARY_DIRS  ... folder containing the libraries in FFTWWrappers_MKL_LIBRARIES
#   FFTWWrappers_LIBRARY_DIR	... fftw library directory
#
# The following variables will be checked by the function
#   FFTWWrappers_ROOT           ... if set, the libraries are exclusively searched
#                                   under this path
#   MKLROOT                     ... take the MKL libraries from here
#

#If environment variable FFTWWrappers_ROOT is defined, it has the same effect as the cmake variable
if( NOT FFTWWrappers_ROOT AND DEFINED ENV{FFTWWrappers_ROOT} )
  if( EXISTS "$ENV{FFTWWrappers_ROOT}/" )
    set( FFTWWrappers_ROOT $ENV{FFTWWrappers_ROOT} )
  else()
    message( "FFTWWrappers_ROOT set to ${FFTWWrappers_ROOT}, but folder does not exist")
  endif()
endif()

#initialize library variables
find_library(
  FFTWWrappers_LIBRARIES
  NAMES libfftw3xc_gnu libfftw3xc_gnu.a
  PATHS ${FFTWWrappers_ROOT}
  PATH_SUFFIXES "lib" "lib64"
  NO_DEFAULT_PATH
  )


if(EXISTS ${FFTWWrappers_LIBRARIES})
  get_filename_component(FFTWWrappers_LIBRARY_DIR ${FFTWWrappers_LIBRARIES} DIRECTORY)
endif()

######################################### MKL related #####################################

if( NOT MKLROOT AND DEFINED ENV{MKLROOT})
  if( EXISTS "$ENV{MKLROOT}/" )
    set( MKLROOT $ENV{MKLROOT} )
  else()
    message( "MKLROOT set to ${MKLROOT}, but folder does not exist")
  endif()
endif()

find_file(
  FFTWWrapper_include_file
  NAMES fftw3.h
  PATHS ${MKLROOT} ${MKLROOT}/include/fftw
  PATH_SUFFIXES "include" "include/fftw"
  NO_DEFAULT_PATH
  )

if(EXISTS ${FFTWWrapper_include_file})
  get_filename_component(FFTWWrappers_MKL_INCLUDE_DIR ${FFTWWrapper_include_file} DIRECTORY)
else()
  message( "FFTWWrappers was not able to find fftw3.h in ${MKLROOT}")
endif()

find_library(
  MKL_INTEL_LP64
  NAMES libmkl_intel_lp64 libmkl_intel_lp64.a
  PATHS ${MKLROOT}
  PATH_SUFFIXES "lib/intel64_lin"
  NO_DEFAULT_PATH
  )

if(EXISTS ${MKL_INTEL_LP64})
  list(APPEND FFTWWrappers_MKL_LIBRARIES "${MKL_INTEL_LP64}")
else()
  message( "FFTWWrappers was not able to find libmkl_intel_lp64.a in ${MKLROOT}")
endif()

find_library(
  MKL_INTEL_THREAD
  NAMES libmkl_intel_thread libmkl_intel_thread.a
  PATHS ${MKLROOT}
  PATH_SUFFIXES "lib/intel64_lin"
  NO_DEFAULT_PATH
  )

if(EXISTS ${MKL_INTEL_THREAD})
  list(APPEND FFTWWrappers_MKL_LIBRARIES "${MKL_INTEL_THREAD}")
else()
  message( "FFTWWrappers was not able to find libmkl_intel_thread.a in ${MKLROOT}")
endif()

find_library(
  MKL_CORE
  NAMES libmkl_core libmkl_core.a
  PATHS ${MKLROOT}
  PATH_SUFFIXES "lib/intel64_lin"
  NO_DEFAULT_PATH
  )

if(EXISTS ${MKL_CORE})
  list(APPEND FFTWWrappers_MKL_LIBRARIES "${MKL_CORE}")
else()
  message("FFTWWrappers was not able to find libmkl_core.a in ${MKLROOT}")
endif()

list(APPEND FFTWWrappers_MKL_LIBRARY_DIRS "${MKLROOT}/../compiler/lib/intel64")
list(APPEND FFTWWrappers_MKL_LIBRARY_DIRS "${MKLROOT}/../tbb/lib/intel64/gcc4.4")

find_library(
  MKL_IOMP5
  NAMES iomp5 libiomp5 libiomp5.a
  PATHS ${MKLROOT} ${MKLROOT}/../compiler/lib/intel64 ${MKLROOT}/../tbb/lib/intel64/gcc4.4
  NO_DEFAULT_PATH
  )

if(EXISTS ${MKL_IOMP5})
  list(APPEND FFTWWrappers_MKL_LIBRARIES "${MKL_IOMP5}")
else()

endif()

list(APPEND FFTWWrappers_MKL_LIBRARIES "m;pthread;dl")

if(NOT FFTWWrappers_VERBOSE_FIND)
  message("++ FindFFTWWrappers")
  message("++ FFTWWrappers_LIBRARIES        : ${FFTWWrappers_LIBRARIES}")
  message("++ FFTWWrappers_LIBRARY_DIR      : ${FFTWWrappers_LIBRARY_DIR}")
  message("++ FFTWWrappers_MKL_LIBRARIES    : ${FFTWWrappers_MKL_LIBRARIES}")
  message("++ FFTWWrappers_MKL_LIBRARY_DIRS : ${FFTWWrappers_MKL_LIBRARY_DIRS}")
  message("++ FFTWWrappers_MKL_INCLUDE_DIR  : ${FFTWWrappers_MKL_INCLUDE_DIR}")
endif()


######################################### EXPORTS #####################################

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTWWrappers DEFAULT_MSG
  FFTWWrappers_LIBRARIES
#  HANDLE_COMPONENTS
  )

mark_as_advanced(
  FFTWWrappers_LIBRARIES
  FFTWWrappers_LIBRARY_DIR
  FFTWWrappers_MKL_LIBRARIES
  FFTWWrappers_MKL_LIBRARY_DIR
  FFTWWrappers_MKL_LIBRARY_DIR
)
