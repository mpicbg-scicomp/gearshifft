# This module only handles the ESSL SMP library and the FFTW3-style wrapper.

set(ESSL_ROOT "/opt/ibmmath/essl/6.2")

#------------------------------------------------------------------------------
# Find ESSL libraries
#------------------------------------------------------------------------------

# library for shared memory parallelism
find_library(
  ESSL_SMP
  NAMES libesslsmp.so
  PATHS ${ESSL_ROOT}
  PATH_SUFFIXES "lib64" "lib"
  NO_DEFAULT_PATHS
)

# ESSL header
find_path(
  ESSL_BASE_INCLUDE_DIR
  NAMES "essl.h"
  PATHS ${ESSL_ROOT}
  PATH_SUFFIXES "include"
  NO_DEFAULT_PATHS
)

#------------------------------------------------------------------------------
# Find FFTW3-style wrapper libraries for ESSL
#------------------------------------------------------------------------------

# FFTW3-style wrapper library
find_library(
  ESSL_FFTW
  NAMES libfftw3_essl.a
  PATHS ${ESSL_ROOT}
  PATH_SUFFIXES "lib64" "lib"
  NO_DEFAULT_PATHS
)

# header for wrapper library
find_path(
  ESSL_FFTW_INCLUDE_DIR
  NAMES "fftw3_essl.h"
  PATHS ${ESSL_ROOT}
  PATH_SUFFIXES "include"
  NO_DEFAULT_PATHS
)

#------------------------------------------------------------------------------
# Add library targets
#------------------------------------------------------------------------------

if(ESSL_BASE_INCLUDE_DIR)
  set(ESSL_INCLUDE_DIRS ${ESSL_BASE_INCLUDE_DIR})
endif()

if(ESSL_FFTW_INCLUDE_DIR)
  set(ESSL_INCLUDE_DIRS ${ESSL_INCLUDE_DIRS} ${ESSL_FFTW_INCLUDE_DIR})
endif()

if(ESSL_SMP)
  set(ESSL_SMP_FOUND TRUE)
  set(ESSL_LIBRARIES ${ESSL_LIBRARIES} ${ESSL_SMP})
  add_library(ESSL::SMP INTERFACE IMPORTED)
  set_target_properties(ESSL::SMP
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ESSL_INCLUDE_DIRS}"
               INTERFACE_LINK_LIBRARIES "${ESSL_SMP}"
  )
else()
  set(ESSL_SMP_FOUND FALSE)
endif()

if(ESSL_FFTW)
  set(ESSL_FFTW_FOUND TRUE)
  set(ESSL_LIBRARIES ${ESSL_LIBRARIES} ${ESSL_FFTW})
  add_library(ESSL::FFTW INTERFACE IMPORTED)
  set_target_properties(ESSL::FFTW
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ESSL_INCLUDE_DIRS}"
               INTERFACE_LINK_LIBRARIES "${ESSL_FFTW};ESSL::SMP"
  )
else()
  set(ESSL_FFTW_FOUND FALSE)
endif()

#------------------------------------------------------------------------------
# Packaging
#------------------------------------------------------------------------------

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  ESSL
  REQUIRED_VARS ESSL_INCLUDE_DIRS ESSL_SMP ESSL_FFTW ESSL_LIBRARIES
)

mark_as_advanced(
  ESSL_INCLUDE_DIRS
  ESSL_SMP
  ESSL_FFTW
  ESSL_LIBRARIES
)
