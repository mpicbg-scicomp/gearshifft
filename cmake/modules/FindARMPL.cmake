# According to https://gerrit.gromacs.org/c/gromacs/+/8621 a FindARMPL module
# shipped by ARM was announced. I can't find a different source, though...

if(NOT ARMPL_DIR AND DEFINED ENV{ARMPL_DIR})
  if(EXISTS "$ENV{ARMPL_DIR}")
    set(ARMPL_DIR $ENV{ARMPL_DIR})
  else()
    message("ARMPL_DIR is set to $ENV{ARMPL_DIR} but the directory does not exist")
  endif()
endif()

#------------------------------------------------------------------------------
# Find ARM PL
#------------------------------------------------------------------------------

find_library(
  ARMPL_LIBRARIES
  NAMES libarmpl_lp64_mp.so
  PATHS $ENV{ARMPL_LIBRARIES} ${ARMPL_DIR}
  PATH_SUFFIXES "lib"
  NO_DEFAULT_PATHS
)

# FFTW-style header
find_path(
  ARMPL_INCLUDE_DIRS
  NAMES "fftw3.h"
  PATHS $ENV{ARMPL_INCLUDES} ${ARMPL_DIR}
  PATH_SUFFIXES "include"
  NO_DEFAULT_PATHS
)

#------------------------------------------------------------------------------
# Add library target
#------------------------------------------------------------------------------

find_package(OpenMP)

add_library(ARMPL::ARMPL INTERFACE IMPORTED)
set_target_properties(ARMPL::ARMPL
  PROPERTIES INTERFACE_COMPILE_DEFINITIONS INTEGER32
             INTERFACE_INCLUDE_DIRECTORIES "${ARMPL_INCLUDE_DIRS}"
             INTERFACE_LINK_LIBRARIES "${ARMPL_LIBRARIES};$<$<CXX_COMPILER_ID:GNU>:gfortran>;OpenMP::OpenMP_CXX"
)

#------------------------------------------------------------------------------
# Packaging
#------------------------------------------------------------------------------

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  ARMPL
  REQUIRED_VARS OPENMP_FOUND ARMPL_INCLUDE_DIRS ARMPL_LIBRARIES
)

mark_as_advanced(
  ARMPL_INCLUDE_DIRS
  ARMPL_LIBRARIES
)
