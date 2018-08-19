cmake_minimum_required(VERSION 3.7)

set(GEARSHIFFT_HALF_VERSION "1.12.0")

find_path(half_INCLUDE_DIR
  NAMES half-${GEARSHIFFT_HALF_VERSION}.zip
  PATHS ${CMAKE_SOURCE_DIR}/ext/half/src)

if((NOT half_INCLUDE_DIR) OR (NOT EXISTS ${half_INCLUDE_DIR}))
  # we couldn't find the header files for FOO or they don't exist
  message("Unable to find dependency 'half'.")

  include(ExternalProject)

  # half-code is a header-only library, so no build and install required
  ExternalProject_Add(ext_half
    PREFIX ${CMAKE_SOURCE_DIR}/ext/half
    URL https://downloads.sourceforge.net/project/half/half/${GEARSHIFFT_HALF_VERSION}/half-${GEARSHIFFT_HALF_VERSION}.zip
    URL_MD5 86d023c0729abf3465bcd55665a39013
    INSTALL_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    LOG_DOWNLOAD ON
    LOG_CONFIGURE OFF
    LOG_BUILD OFF
    )
  ExternalProject_Get_Property(ext_half DOWNLOAD_DIR)

  set(half_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/ext/half/src
    CACHE PATH "'half' include directory")
endif()

# for convenience setup a target
add_library(half INTERFACE)
target_include_directories(half INTERFACE
  $<BUILD_INTERFACE:${half_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}>)

# need to export target as well
install(TARGETS half EXPORT gearshifftExport DESTINATION lib)
