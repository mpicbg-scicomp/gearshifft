#-------------------------------------------------------------------------------

option(GEARSHIFFT_SUPERBUILD_EXT_INBUILD "Builds external libraries in build directory (otherwise it is built in ext/)" ON)
set( GEARSHIFFT_SUPERBUILD_EXT_DIR "${GEARSHIFFT_ROOT}/ext" CACHE STRING "Install directory for external dependencies")

if(GEARSHIFFT_SUPERBUILD_EXT_INBUILD)
  set( GEARSHIFFT_SUPERBUILD_EXT_DIR "${CMAKE_BINARY_DIR}/ext" CACHE STRING "" FORCE)
endif()
set( GEARSHIFFT_SUPERBUILD_EXT_SOURCE_DIR "${GEARSHIFFT_SUPERBUILD_EXT_DIR}/src" CACHE STRING "Source directory for external dependencies")


# superbuild is aimed for static build (otherwise env must be set accordingly, so the libs are found)
#set(GEARSHIFFT_USE_STATIC_LIBS ON CACHE BOOL "" FORCE)

include(ExternalProject)

#------------------------------------------------------------------------------
# Boost
#------------------------------------------------------------------------------
option(GEARSHIFFT_SUPERBUILD_EXT_DOWNLOAD_Boost "Download and build Boost" OFF )

if(GEARSHIFFT_SUPERBUILD_EXT_DOWNLOAD_Boost)
  # looks in /gearshifft/ext/boost, and may download+build it
  # sets BOOST_ROOT
  include(ext/Boost)
else()
  set(BOOST_ROOT "")
endif()

#------------------------------------------------------------------------------
# FFTW
#------------------------------------------------------------------------------
option(GEARSHIFFT_SUPERBUILD_EXT_DOWNLOAD_FFTW "Download and build FFTW" OFF )

if(GEARSHIFFT_SUPERBUILD_EXT_DOWNLOAD_FFTW)
  # looks in /gearshifft/ext/fftw, and may download+build it
  # sets FFTW_ROOT
  include(ext/FFTW)
else()
  set(FFTW_ROOT "")
endif()

#------------------------------------------------------------------------------
# clFFT
#------------------------------------------------------------------------------
option(GEARSHIFFT_SUPERBUILD_EXT_DOWNLOAD_CLFFT "Download and build clFFT" OFF )

if(GEARSHIFFT_SUPERBUILD_EXT_DOWNLOAD_CLFFT)
  # looks in /gearshifft/ext/clfft, and may download+build it
  # sets CLFFT_ROOT
  include(ext/clfft)
else()
  set(CLFFT_ROOT "")
endif()

#------------------------------------------------------------------------------
# ROCM+rocFFT
#------------------------------------------------------------------------------
option(GEARSHIFFT_SUPERBUILD_EXT_DOWNLOAD_ROCFFT "Download and build rocfft" OFF )

if(GEARSHIFFT_SUPERBUILD_EXT_DOWNLOAD_ROCFFT)
  # looks in /gearshifft/ext/rocfft, and may download+build it
  # sets ROCFFT_ROOT
  include(ext/rocfft)
else()
  set(ROCFFT_ROOT "")
endif()


#------------------------------------------------------------------------------
# gearshifft
#------------------------------------------------------------------------------

# to get BUILD_TESTING option, which is just forwarded to gearshifft
include(CTest)
if(GEARSHIFFT_USE_STATIC_LIBS AND GEARSHIFFT_SUPERBUILD_EXT_DOWNLOAD_Boost)
  # TODO: not supported yet
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
endif()


include(get_gearshifft_options)
get_gearshifft_options(cmake_pass_args ";-D") # ; to convert to a list
set(SEPARATE_BACKENDS)

macro(append_args VAR)
  foreach(ARG IN ITEMS ${ARGN})
    if(${ARG})
      string(APPEND ${VAR} ";-D${ARG}=${${ARG}}")
    endif()
  endforeach()
endmacro()

# builds specific backend, changes to cmake_pass_args remains local (as it is a cmake function)
function(build_gearshifft BACKEND)
  if(BACKEND MATCHES "rocfft")
    if(NOT CMAKE_CXX_COMPILER MATCHES "hcc$")
      find_program(PROGRAM_HCC NAMES "hcc")
      if(NOT PROGRAM_HCC)
        message(WARNING "HC compiler not found. rocFFT backend will not be build.")
        return()
      endif()
      set(CMAKE_CXX_COMPILER ${PROGRAM_HCC})
    endif()
    append_args(cmake_pass_args
      BOOST_ROOT
      ROCFFT_ROOT
      CMAKE_CXX_COMPILER)
    set(cmake_pass_args "${cmake_pass_args};-DGEARSHIFFT_BACKEND_ROCFFT_ONLY=ON")
  else()
    append_args(cmake_pass_args
      BOOST_ROOT
      FFTW_ROOT
      CLFFT_ROOT)
    set(cmake_pass_args "${cmake_pass_args};-DGEARSHIFFT_BACKEND_ROCFFT=OFF")
  endif()

  if(BACKEND)
    set(BACKEND "-${BACKEND}")
  else()
    set(BACKEND "")
  endif()

  ExternalProject_Add( gearshifft${BACKEND}
    DEPENDS ${gearshifft_DEPENDENCIES}
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    SOURCE_DIR ${CMAKE_SOURCE_DIR}
    BINARY_DIR ${CMAKE_BINARY_DIR}/gearshifft${BACKEND}-build
    CMAKE_GENERATOR ${EP_CMAKE_GENERATOR}
    CMAKE_ARGS
    ${cmake_pass_args}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DBUILD_TESTING:BOOL=${BUILD_TESTING}
    -DGEARSHIFFT_USE_SUPERBUILD:BOOL=OFF
    -DGEARSHIFFT_SUPERBUILD_DIR:PATH=${CMAKE_BINARY_DIR}
    -DSEPARATE_BACKENDS:LIST=${SEPARATE_BACKENDS}
    INSTALL_COMMAND ""
    )
  set(SEPARATE_BACKENDS "gearshifft${BACKEND};${SEPARATE_BACKENDS}" PARENT_SCOPE)
endfunction()

# always as separate build due to probably different compiler chain (eg HCC, g++, ..)
if(GEARSHIFFT_BACKEND_ROCFFT)
  build_gearshifft("rocfft")
endif()

# actual build (must be the last target, i.e., after all backend-specific builds)
build_gearshifft("")

ExternalProject_Add_Step(gearshifft package
  DEPENDS ${SEPARATE_BACKENDS}
  COMMAND ${CMAKE_COMMAND} --build gearshifft-build --target package
  DEPENDEES build #steps we depend on
  ALWAYS YES # always appear out of date
  EXCLUDE_FROM_MAIN YES # external project’s main target does not depend on the custom step.
)
ExternalProject_Add_StepTargets(gearshifft package)
