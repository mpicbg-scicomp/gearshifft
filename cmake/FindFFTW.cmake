# - Find the FFTW library
#
# Usage:
#   find_package(FFTW [REQUIRED] [QUIET] [COMPONENTS ...])
#     
# It sets the following variables:
#   FFTW_FOUND          ... true if fftw is found on the system
#   FFTW_LIBRARIES      ... full path to fftw library (will be filled with serial/openmp/pthreads enabled file names if present)
#   FFTW_INCLUDES       ... fftw include directory
#   FFTW_SERIAL_LIBS	... full path to serial fftw library
#   FFTW_THREADS_LIBS	... full path to pthreads fftw library
#   FFTW_OPENMP_LIBS	... full path to openmp fftw library
#
# The following variables will be checked by the function
#   FFTW_USE_STATIC_LIBS    ... if true, only static libraries are found
#   FFTW_ROOT               ... if set, the libraries are exclusively searched
#                               under this path
#   FFTW_LIBRARY            ... fftw library to use
#   FFTW_INCLUDE_DIR        ... fftw include directory
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
if( NOT FFTW_ROOT AND (ENV{FFTWDIR} OR ENV{FFTW_ROOT}))
  if(EXISTS ENV{FFTWDIR})
    set( FFTW_ROOT $ENV{FFTWDIR} )
  endif()
  if(EXISTS ENV{FFTW_ROOT})
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

if( ${FFTW_USE_STATIC_LIBS} )
  set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX} )
else()
  set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX} )
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

  if(EXISTS ${FFTW_SERIAL_STUBLIB})
    list(APPEND FFTW_SERIAL_LIBS ${FFTW_SERIAL_STUBLIB})
  else()
    if(NOT FFTW_FIND_QUIETLY)
      message("FFTW_SERIAL_STUBLIB empty for ${_LIBSTUB} .. :${FFTW_SERIAL_STUBLIB}:")
    endif()
  endif()

  if(EXISTS ${FFTW_OPENMP_STUBLIB})
    list(APPEND FFTW_OPENMP_LIBS ${FFTW_OPENMP_STUBLIB})
  else()
    if(NOT FFTW_FIND_QUIETLY)
      message("FFTW_OPENMP_STUBLIB empty for ${_LIBSTUB} .. :${FFTW_OPENMP_STUBLIB}:")
    endif()
  endif()

  if(EXISTS ${FFTW_THREADS_STUBLIB})
    list(APPEND FFTW_THREADS_LIBS ${FFTW_THREADS_STUBLIB})
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

# fill libraries
set(FFTW_LIBRARIES)
foreach(_ITEM IN LISTS FFTW_SERIAL_LIBS FFTW_THREADS_LIBS FFTW_OPENMP_LIBS )
  if(EXISTS ${_ITEM})
    set(FFTW_LIBRARIES "${FFTW_LIBRARIES} ${_ITEM}")
  endif()
endforeach()


#revert CMAKE_FIND_LIBRARY_SUFFIXES
set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW DEFAULT_MSG
  FFTW_INCLUDES
  FFTW_LIBRARIES
  )

mark_as_advanced(FFTW_INCLUDES #include directory
  FFTW_LIBRARIES
  FFTW_SERIAL_LIBS #individual serial libs
  FFTW_THREADS_LIBS #pthreads libraries
  FFTW_OPENMP_LIBS #openmp libraries
  )

