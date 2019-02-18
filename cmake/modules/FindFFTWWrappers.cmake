# - Find the FFTW library
#
# Usage:
#   find_package(FFTWWrappers [REQUIRED] [QUIET])
#
# It sets the following variables:
#   FFTWWrappers_FOUND          ... true if fftw is found on the system
#   FFTWWrappers_GNU_LIBRARIES  ... list of library that were build with the GNU binary layout
#   FFTWWrappers_INTEL_LIBRARIES  ... list of library that were build with the intel binary layout
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

#If environment variable MKLROOT is defined, it has the same effect as the cmake variable
if( NOT MKLROOT AND DEFINED ENV{MKLROOT})
  if( EXISTS "$ENV{MKLROOT}/" )
    set( MKLROOT $ENV{MKLROOT} )
  else()
    message( "MKLROOT set to ${MKLROOT}, but folder does not exist")
  endif()
endif()

################################### FFTWWrappers related ##################################

#initialize library variables
find_library(
  FFTWWrappers_GNU_LIBRARIES
  NAMES libfftw3xc_gnu libfftw3xc_gnu.a
  PATHS ${FFTWWrappers_ROOT}
  PATH_SUFFIXES "lib" "lib64"
  NO_DEFAULT_PATH
  )
find_library(
  FFTWWrappers_GNU_LIBRARIES
  NAMES libfftw3xc_gnu libfftw3xc_gnu.a
  PATHS ${MKLROOT}
  PATH_SUFFIXES "lib/intel64"
  NO_DEFAULT_PATH
  )
find_library(
  FFTWWrappers_GNU_LIBRARIES
  NAMES libfftw3xc_gnu libfftw3xc_gnu.a
  )

if(EXISTS ${FFTWWrappers_GNU_LIBRARIES})
  get_filename_component(FFTWWrappers_LIBRARY_DIR ${FFTWWrappers_GNU_LIBRARIES} DIRECTORY)
endif()

find_library(
  FFTWWrappers_INTEL_LIBRARIES
  NAMES libfftw3xc_intel libfftw3xc_intel.a
  PATHS ${FFTWWrappers_ROOT}
  PATH_SUFFIXES "lib" "lib64"
  NO_DEFAULT_PATH
  )
find_library(
  FFTWWrappers_INTEL_LIBRARIES
  NAMES libfftw3xc_intel libfftw3xc_intel.a
  PATHS ${MKLROOT}
  PATH_SUFFIXES "lib/intel64_lin"
  NO_DEFAULT_PATH
)
find_library(
  FFTWWrappers_INTEL_LIBRARIES
  NAMES libfftw3xc_intel libfftw3xc_intel.a
)

if(EXISTS ${FFTWWrappers_INTEL_LIBRARIES})
  get_filename_component(FFTWWrappers_LIBRARY_DIR ${FFTWWrappers_INTEL_LIBRARIES} DIRECTORY)
endif()

######################################### MKL related #####################################

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
  PATH_SUFFIXES "lib/intel64_lin" "lib/intel64"
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
  PATH_SUFFIXES "lib/intel64_lin" "lib/intel64"
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
  PATH_SUFFIXES "lib/intel64_lin" "lib/intel64"
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
  PATHS ${MKLROOT} ${MKLROOT}/../compiler ${MKLROOT}/../tbb
  PATH_SUFFIXES "lib/intel64_lin" "lib/intel64" "lib/intel64/gcc4.4"
  NO_DEFAULT_PATH
  )

if(EXISTS ${MKL_IOMP5})
  list(APPEND FFTWWrappers_MKL_LIBRARIES "${MKL_IOMP5}")
else()

endif()

list(APPEND FFTWWrappers_MKL_LIBRARIES "m;dl")

if(NOT FFTWWrappers_FIND_QUIETLY)
  message("++ FindFFTWWrappers")
  message("++ FFTWWrappers_GNU_LIBRARIES    : ${FFTWWrappers_GNU_LIBRARIES}")
  message("++ FFTWWrappers_INTEL_LIBRARIES  : ${FFTWWrappers_INTEL_LIBRARIES}")
  message("++ FFTWWrappers_LIBRARY_DIR      : ${FFTWWrappers_LIBRARY_DIR}")
  message("++ FFTWWrappers_MKL_LIBRARIES    : ${FFTWWrappers_MKL_LIBRARIES}")
  message("++ FFTWWrappers_MKL_LIBRARY_DIRS : ${FFTWWrappers_MKL_LIBRARY_DIRS}")
  message("++ FFTWWrappers_MKL_INCLUDE_DIR  : ${FFTWWrappers_MKL_INCLUDE_DIR}")
endif()


######################################### EXPORTS #####################################

include(FindPackageHandleStandardArgs)

if(FFTWWrappers_GNU_LIBRARIES)
  find_package_handle_standard_args(FFTWWrappers
    REQUIRED_VARS FFTWWrappers_GNU_LIBRARIES
    REQUIRED_VARS FFTWWrappers_MKL_LIBRARIES
    REQUIRED_VARS FFTWWrappers_MKL_INCLUDE_DIR
    )
  set(FFTWWrappers_LIBRARIES "${FFTWWrappers_GNU_LIBRARIES}")

else()
  find_package_handle_standard_args(FFTWWrappers
    REQUIRED_VARS FFTWWrappers_INTEL_LIBRARIES
    REQUIRED_VARS FFTWWrappers_MKL_LIBRARIES
    REQUIRED_VARS FFTWWrappers_MKL_INCLUDE_DIR
    )
  set(FFTWWrappers_LIBRARIES "${FFTWWrappers_INTEL_LIBRARIES}")

endif()


mark_as_advanced(
  FFTWWrappers_LIBRARIES
  FFTWWrappers_GNU_LIBRARIES
  FFTWWrappers_INTEL_LIBRARIES
  FFTWWrappers_LIBRARY_DIR
  FFTWWrappers_MKL_LIBRARIES
  FFTWWrappers_MKL_LIBRARY_DIR
  FFTWWrappers_MKL_INCLUDE_DIR
)
