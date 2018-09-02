# - Find clFFT, AMD's OpenCL FFT library
#
# Usage:
#   find_package(clFFT [REQUIRED] [QUIET])
#
# It sets the following variables:
# clFFT_INCLUDE_DIRS    - Location of clFFT's include directory.
# clFFT_LIBRARIES       - Location of clFFT's libraries
# clFFT_FOUND           - True if clFFT has been located
#
# It also defines the imported target clFFT::clFFT.
#
# If your clFFT installation is not in a standard installation directory, you
# may provide a hint to where it may be found. Simply set the value CLFFT_ROOT
# to the directory containing 'include/clFFT.h" prior to calling this script.


find_path(CLFFT_ROOT_DIR
  NAMES include/clFFT.h
  PATHS CLFFT_ROOT
  PATHS ENV CLFFT_ROOT
  DOC "clFFT root directory.")

find_path(CLFFT_ROOT_DIR
  NAMES include/clFFT.h
  DOC "clFFT root directory.")

find_path(_CLFFT_INCLUDE_DIRS
  NAMES clFFT.h
  PATHS "${CLFFT_ROOT_DIR}" "/usr"
  PATH_SUFFIXES "include"
  DOC "clFFT include directory")

find_path(_CLFFT_INCLUDE_DIRS
  NAMES clFFT.h
  DOC "clFFT include directory")

find_library(_CLFFT_LIBRARY
  NAMES clFFT
  PATHS "${CLFFT_ROOT_DIR}" "/usr"
  PATH_SUFFIXES "lib" "lib64"
  DOC "clFFT library directory")

find_library(_CLFFT_LIBRARY
  NAMES clFFT
  DOC "clFFT library directory")


set(clFFT_INCLUDE_DIRS ${_CLFFT_INCLUDE_DIRS})
set(clFFT_LIBRARIES ${_CLFFT_LIBRARY})

add_library(clFFT::clFFT IMPORTED INTERFACE)
set_target_properties(clFFT::clFFT PROPERTIES
  INTERFACE_LINK_LIBRARIES "${clFFT_LIBRARIES}"
  INTERFACE_INCLUDE_DIRECTORIES "${clFFT_INCLUDE_DIRS}"
  )

# handle the QUIETLY and REQUIRED arguments and set clFFT_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(clFFT DEFAULT_MSG clFFT_LIBRARIES clFFT_INCLUDE_DIRS)

if(NOT clFFT_FIND_QUIETLY)
  message("++ FindclFFT")
  message("++ clFFT_INCLUDES    : ${clFFT_INCLUDE_DIRS}")
  message("++ clFFT_LIBRARIES   : ${clFFT_LIBRARIES}")
endif()

mark_as_advanced(
  clFFT_LIBRARIES
  clFFT_INCLUDE_DIRS
  )
