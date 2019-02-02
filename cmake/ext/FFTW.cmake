#---------------------------------------------------------------------------
# Get and build fftw (double+single), if it is not already present

set(GEARSHIFFT_EXT_FFTW_VERSION "3.3.8" CACHE STRING "FFTW version to be built.")
set_property(CACHE GEARSHIFFT_EXT_FFTW_VERSION PROPERTY STRINGS "3.3.8")

set(FFTW_ROOT ${GEARSHIFFT_SUPERBUILD_EXT_DIR}/fftw CACHE PATH "'FFTW' root directory")

find_path(FFTW_INCLUDE_DIR
  NAMES fftw3.h
  PATHS ${FFTW_ROOT}/include
  NO_DEFAULT_PATH)

find_library(FFTW_LIBRARY_DIR
  NAMES fftw3
  PATHS ${FFTW_ROOT}/lib
  NO_DEFAULT_PATH)

find_library(FFTWF_LIBRARY_DIR
  NAMES fftw3f
  PATHS ${FFTW_ROOT}/lib
  NO_DEFAULT_PATH)

if((NOT FFTW_INCLUDE_DIR) OR (NOT EXISTS ${FFTW_INCLUDE_DIR})
    OR
    (NOT FFTW_LIBRARY_DIR) OR (NOT EXISTS ${FFTW_LIBRARY_DIR})
    OR
    (NOT FFTWF_LIBRARY_DIR) OR (NOT EXISTS ${FFTWF_LIBRARY_DIR})
    )

  message("'fftw' and 'fftwf' library could not be found, so [make] will download and build it.")

  # md5 hashes of tar.gz files
  if( UNIX OR APPLE )
    if(GEARSHIFFT_EXT_FFTW_VERSION MATCHES "3.3.8")
      set(FFTW_hash "8aac833c943d8e90d51b697b27d4384d")
    else()
      set(FFTW_hash "")
    endif()

    set( FFTW_url "http://www.fftw.org/fftw-${GEARSHIFFT_EXT_FFTW_VERSION}.tar.gz")
  else()
    message(FATAL_ERROR "Windows is not supported yet.")
  endif()

  if(GEARSHIFFT_USE_STATIC_LIBS)
    set(BUILD_LIBS "static")
  else()
    set(BUILD_LIBS "shared")
  endif()
  find_package(OpenMP QUIET)
  if(OpenMP_FOUND)
    set(FFTW_OPENMP_FLAG "--enable-openmp")
  else()
    message(WARNING "Build FFTW without OpenMP support (could not detect OpenMP for ${CMAKE_CXX_COMPILER}).")
    set(FFTW_OPENMP_FLAG "--enable-openmp=no")
  endif()
  #FFTW does not distinguish between "release" and "debug"
  set(FFTW_BUILD_FLAGS ${FFTW_OPENMP_FLAG} --enable-sse2 -q --enable-${BUILD_LIBS}=yes --with-gnu-ld  --enable-silent-rules --with-pic)
  ExternalProject_Add(fftw
    BUILD_IN_SOURCE 1
    URL ${FFTW_url}
    URL_MD5 ${FFTW_hash}
    PREFIX ${GEARSHIFFT_SUPERBUILD_EXT_DIR}
    SOURCE_DIR ${GEARSHIFFT_SUPERBUILD_EXT_SOURCE_DIR}/fftw
    DOWNLOAD_DIR ${GEARSHIFFT_ROOT}/ext/downloads/
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND CC=${CMAKE_C_COMPILER}; CXX=${CMAKE_CXX_COMPILER} ./configure --prefix=${FFTW_ROOT} ${FFTW_BUILD_FLAGS}
    BUILD_COMMAND make -j 4
    INSTALL_COMMAND make install
    )
  list(APPEND gearshifft_DEPENDENCIES fftw)

  ExternalProject_Add(fftwf
    BUILD_IN_SOURCE 1
    URL ${FFTW_url}
    URL_MD5 ${FFTW_hash}
    PREFIX ${GEARSHIFFT_SUPERBUILD_EXT_DIR}
    SOURCE_DIR ${GEARSHIFFT_SUPERBUILD_EXT_SOURCE_DIR}/fftwf
    DOWNLOAD_DIR ${GEARSHIFFT_ROOT}/ext/downloads/
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND CC=${CMAKE_C_COMPILER}; CXX=${CMAKE_CXX_COMPILER} ./configure --prefix=${FFTW_ROOT} --enable-float ${FFTW_BUILD_FLAGS}
    BUILD_COMMAND make -j 4
    INSTALL_COMMAND make install
    )
  list(APPEND gearshifft_DEPENDENCIES fftwf)

endif()
