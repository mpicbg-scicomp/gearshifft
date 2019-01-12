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

function(append_args ARGS)
  foreach(ARG in ${ARGS})
    if(ARG)
      set(cmake_pass_args "${cmake_pass_args};-D${ARG}=${${ARG}}")
    endif()
  endforeach()
endfunction()

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
    append_args(ROCFFT_ROOT)
    # TODO: disable other backends
  else()
    set(cmake_pass_args "${cmake_pass_args};-DGEARSHIFFT_BACKEND_ROCFFT=OFF")
    append_args(BOOST_ROOT
      FFTW_ROOT
      CLFFT_ROOT)
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
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DBUILD_TESTING:BOOL=${BUILD_TESTING}
    -DGEARSHIFFT_USE_SUPERBUILD:BOOL=OFF
    -DCPACK_INSTALL_CMAKE_PROJECTS=${CPACK_INSTALL_CMAKE_PROJECTS}
    INSTALL_COMMAND ""
    )
endfunction()

if(GEARSHIFFT_BACKEND_ROCFFT)
  build_gearshifft("rocfft")
  if(EXISTS "${CMAKE_BINARY_DIR}/gearshifft-rocfft-build/gearshifft/gearshifft-rocfft")
    message("Adds gearshifft-rocfft to later package.")
    list(APPEND CPACK_INSTALL_CMAKE_PROJECTS "${CMAKE_BINARY_DIR}/gearshifft-rocfft-build/gearshifft/;gearshifft;gearshifft-rocfft;/")
  endif()
endif()

# actual build
build_gearshifft("")
ExternalProject_Add_Step(gearshifft package
  COMMAND ${CMAKE_COMMAND} --build gearshifft-build --target package
  DEPENDEES build #steps we depend on
  ALWAYS YES # always appear out of date
  EXCLUDE_FROM_MAIN YES # external project’s main target does not depend on the custom step.
)
ExternalProject_Add_StepTargets(gearshifft package)
