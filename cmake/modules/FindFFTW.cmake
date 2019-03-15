# - Find the FFTW library
#
# Usage:
#   find_package(FFTW [REQUIRED] [QUIET] [COMPONENTS ...])
#
# It sets the following variables:
#   FFTW_FOUND          ... true if fftw is found on the system
#   FFTW_LIBRARIES      ... list of library identifiers that were found (will be filled with serial/openmp/pthreads enabled file names if present, only stubs will be filled in not full paths)
#   FFTW_INCLUDES       ... fftw include directory
#   FFTW_SERIAL_LIBS  ... list of full path(s) to serial fftw library(ies)
#   FFTW_THREADS_LIBS ... list of full path(s) to pthreads fftw library(ies)
#   FFTW_OPENMP_LIBS  ... list of full path(s) to openmp fftw library(ies)
#   FFTW_INCLUDE_DIR  ... fftw include directory
#   FFTW_LIBRARY_DIR  ... fftw library directory
#
# The following variables will be checked by the function
#   FFTW_USE_STATIC_LIBS    ... if true, only static libraries are found
#   FFTW_ROOT               ... if set, the libraries are exclusively searched
#                               under this path
#   FFTW_LIBRARY            ... fftw library to use
#   FFTW_INCLUDE_DIR        ... fftw include directory
#   FFTW_VERBOSE_FIND       ... print extra messages
#
# Besides these command line defines, the environment is searched for FFTWDIR and FFTW_ROOT to be filled
# if these variables are defined, it's assumed that they point to the FFTW installation root path


#sanity check of components
if(NOT FFTW_FIND_COMPONENTS)
  set(FFTW_FIND_COMPONENTS "float;double")
endif()

set(FFTW_ALLOWED_COMPONENTS "single;float;double;long;quad")
foreach(_COMPONENT IN LISTS FFTW_FIND_COMPONENTS)
  list(FIND FFTW_ALLOWED_COMPONENTS ${_COMPONENT}  ALLOWED_INDEX)
  if(${ALLOWED_INDEX} LESS 0)
    if(NOT FFTW_FIND_QUIETLY)
      message("-- [FindFFTW.cmake] ignoring unknown component ${_COMPONENT}, allowed values are ${FFTW_ALLOWED_COMPONENTS}")
    endif()
    list(REMOVE_ITEM FFTW_FIND_COMPONENTS ${_COMPONENT})
  endif()
endforeach()

#prepare fftw library precision stub
set(FFTW_LIBRARY_STUBS)
foreach(_COMPONENT IN LISTS FFTW_FIND_COMPONENTS)
  if(_COMPONENT MATCHES "float" OR _COMPONENT MATCHES "single")
    list(APPEND FFTW_LIBRARY_STUBS "fftw3f")
  endif()
  if(_COMPONENT MATCHES "double")
    list(APPEND FFTW_LIBRARY_STUBS "fftw3")
  endif()
  if(_COMPONENT MATCHES "long")
    list(APPEND FFTW_LIBRARY_STUBS "fftw3l")
  endif()
  if(_COMPONENT MATCHES "quad")
    list(APPEND FFTW_LIBRARY_STUBS "fftw3q")
  endif()
endforeach()

#If environment variable FFTWDIR is specified, it has same effect as FFTW_ROOT
if( NOT FFTW_ROOT AND (DEFINED ENV{FFTWDIR} OR DEFINED ENV{FFTW_ROOT}))
  if(EXISTS "$ENV{FFTWDIR}/")
    set( FFTW_ROOT $ENV{FFTWDIR} )
  endif()
  if( EXISTS "$ENV{FFTW_ROOT}/" )
    set( FFTW_ROOT $ENV{FFTW_ROOT} )
  endif()
endif()

# Check if we can use PkgConfig
find_package(PkgConfig)

#Determine from PKG
if( PKG_CONFIG_FOUND AND NOT FFTW_ROOT )
  pkg_check_modules( PKG_FFTW QUIET "fftw3" )
endif()

#Check whether to search static or dynamic libs
set( CMAKE_FIND_LIBRARY_SUFFIXES_SAV ${CMAKE_FIND_LIBRARY_SUFFIXES} )

if( FFTW_USE_STATIC_LIBS ) # find static libs only
  set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX} )
endif()

#initialize library variables
foreach(_LIBSTUB IN LISTS FFTW_LIBRARY_STUBS)

  if( FFTW_ROOT )

    find_library(
      FFTW_SERIAL_STUBLIB
      NAMES ${_LIBSTUB} lib${_LIBSTUB}
      PATHS ${FFTW_ROOT}
      PATH_SUFFIXES "lib" "lib64"
      NO_DEFAULT_PATH
      )

    find_library(
      FFTW_OPENMP_STUBLIB
      NAMES ${_LIBSTUB}_omp lib${_LIBSTUB}_omp
      PATHS ${FFTW_ROOT}
      PATH_SUFFIXES "lib" "lib64"
      NO_DEFAULT_PATH
      )

    find_library(
      FFTW_THREADS_STUBLIB
      NAMES ${_LIBSTUB}_threads lib${_LIBSTUB}_threads
      PATHS ${FFTW_ROOT}
      PATH_SUFFIXES "lib" "lib64"
      NO_DEFAULT_PATH
      )

  else( FFTW_ROOT )
    find_library(
      FFTW_SERIAL_STUBLIB
      NAMES ${_LIBSTUB} lib${_LIBSTUB}
      PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
      )

    find_library(
      FFTW_OPENMP_STUBLIB
      NAMES ${_LIBSTUB}_omp lib${_LIBSTUB}_omp
      PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
      )


    find_library(
      FFTW_THREADS_STUBLIB
      NAMES ${_LIBSTUB}_threads lib${_LIBSTUB}_threads
      PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
      )

  endif( FFTW_ROOT )

  # manually build shared libs from static libs (if only static libs were found)
  if(EXISTS ${FFTW_SERIAL_STUBLIB})

    if( NOT CMAKE_FIND_LIBRARY_SUFFIXES MATCHES ${CMAKE_STATIC_LIBRARY_SUFFIX} )
      add_library(${_LIBSTUB} STATIC IMPORTED GLOBAL)
    else()
      add_library(${_LIBSTUB} SHARED IMPORTED GLOBAL)
    endif()

    set_target_properties(${_LIBSTUB} PROPERTIES IMPORTED_LOCATION ${FFTW_SERIAL_STUBLIB} )

    list(APPEND FFTW_SERIAL_LIBS_ABSPATH ${FFTW_SERIAL_STUBLIB})
    list(APPEND FFTW_SERIAL_LIBS ${_LIBSTUB})

    list(APPEND FFTW_LIBRARIES ${_LIBSTUB})
  else()
    if(NOT FFTW_FIND_QUIETLY)
      message("FFTW_SERIAL_STUBLIB empty for ${_LIBSTUB} .. :${FFTW_SERIAL_STUBLIB}:")
    endif()
  endif()

  if(EXISTS ${FFTW_OPENMP_STUBLIB})

    if( NOT CMAKE_FIND_LIBRARY_SUFFIXES MATCHES ${CMAKE_STATIC_LIBRARY_SUFFIX} )
      add_library(${_LIBSTUB}_omp STATIC IMPORTED GLOBAL)
    else()
      add_library(${_LIBSTUB}_omp SHARED IMPORTED GLOBAL)
    endif()

    set_target_properties(${_LIBSTUB}_omp PROPERTIES IMPORTED_LOCATION ${FFTW_OPENMP_STUBLIB} )

    list(APPEND FFTW_OPENMP_LIBS_ABSPATH ${FFTW_OPENMP_STUBLIB})
    list(APPEND FFTW_OPENMP_LIBS ${_LIBSTUB}_omp)

    list(APPEND FFTW_LIBRARIES ${_LIBSTUB}_omp)

  else()
    if(NOT FFTW_FIND_QUIETLY)
      message("FFTW_OPENMP_STUBLIB empty for ${_LIBSTUB} .. :${FFTW_OPENMP_STUBLIB}:")
    endif()
  endif()

  if(EXISTS ${FFTW_THREADS_STUBLIB})
    if( NOT CMAKE_FIND_LIBRARY_SUFFIXES MATCHES ${CMAKE_STATIC_LIBRARY_SUFFIX} )
      add_library(${_LIBSTUB}_threads STATIC IMPORTED GLOBAL)
    else()
      add_library(${_LIBSTUB}_threads SHARED IMPORTED GLOBAL)
    endif()
    set_target_properties(${_LIBSTUB}_threads PROPERTIES IMPORTED_LOCATION ${FFTW_THREADS_STUBLIB} )

    list(APPEND FFTW_THREADS_LIBS_ABSPATH ${FFTW_THREADS_STUBLIB})
    list(APPEND FFTW_THREADS_LIBS ${_LIBSTUB}_threads)

    list(APPEND FFTW_LIBRARIES ${_LIBSTUB}_threads)
  else()
    if(NOT FFTW_FIND_QUIETLY)
      message("FFTW_OPENMP_STUBLIB empty for ${_LIBSTUB} .. :${FFTW_THREADS_STUBLIB}:")
    endif()
  endif()

  unset(FFTW_SERIAL_STUBLIB CACHE)
  unset(FFTW_THREADS_STUBLIB CACHE)
  unset(FFTW_OPENMP_STUBLIB CACHE)

endforeach()

#find includes
if( FFTW_ROOT )

  find_path(
    FFTW_INCLUDES
    NAMES "fftw3.h"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
    )

else()

  find_path(
    FFTW_INCLUDES
    NAMES "fftw3.h"
    PATHS ${PKG_FFTW_INCLUDE_DIRS} ${INCLUDE_INSTALL_DIR}
    )

endif( FFTW_ROOT )

if(EXISTS ${FFTW_INCLUDES})
  set(FFTW_INCLUDE_DIR ${FFTW_INCLUDES})
endif()

# fill library dir
set(FFTW_LIBFILES_LIST)

foreach(_ITEM IN LISTS FFTW_SERIAL_LIBS_ABSPATH FFTW_THREADS_LIBS_ABSPATH FFTW_OPENMP_LIBS_ABSPATH )
  if(EXISTS ${_ITEM})
    list(APPEND FFTW_LIBFILES_LIST ${_ITEM})
  endif()
endforeach()

if(FFTW_LIBRARIES)
  list(GET FFTW_LIBFILES_LIST 0 FFTW_LIBFILES_ITEM0)
  get_filename_component(FFTW_LIBRARY_DIR ${FFTW_LIBFILES_ITEM0} DIRECTORY)
endif()

#revert CMAKE_FIND_LIBRARY_SUFFIXES
set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW DEFAULT_MSG
  FFTW_INCLUDES
  FFTW_LIBRARIES
  #  HANDLE_COMPONENTS
  )

if(NOT FFTW_FIND_QUIETLY)
  message("++ FindFFTW")
  message("++ FFTW_INCLUDES    : ${FFTW_INCLUDES}")
  message("++ FFTW_LIBRARIES   : ${FFTW_LIBRARIES}")
  message("++ FFTW_SERIAL_LIBS : ${FFTW_SERIAL_LIBS_ABSPATH}")
  message("++ FFTW_THREADS_LIBS: ${FFTW_THREADS_LIBS_ABSPATH}")
  message("++ FFTW_OPENMP_LIBS : ${FFTW_OPENMP_LIBS_ABSPATH}")
  message("++ FFTW_INCLUDE_DIR : ${FFTW_INCLUDE_DIR}")
  message("++ FFTW_LIBRARY_DIR : ${FFTW_LIBRARY_DIR}")
endif()

mark_as_advanced(
  FFTW_INCLUDES #include directory
  FFTW_LIBRARIES
  FFTW_SERIAL_LIBS #individual serial libs
  FFTW_THREADS_LIBS #pthreads libraries
  FFTW_OPENMP_LIBS #openmp libraries
  FFTW_INCLUDE_DIR
  FFTW_LIBRARY_DIR
  )
