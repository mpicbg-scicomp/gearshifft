#---------------------------------------------------------------------------
# Get and build half, if it is not already present

set(GEARSHIFFT_HALF_VERSION "1.12.0" CACHE STRING "'half' version to be built.")
set_property(CACHE GEARSHIFFT_HALF_VERSION PROPERTY STRINGS "1.12.0")

if(NOT GEARSHIFFT_SUPERBUILD_EXT_DIR)
  set(half_BASE_DIR ${GEARSHIFFT_ROOT}/ext/half)
else()
  set(half_BASE_DIR ${GEARSHIFFT_SUPERBUILD_EXT_DIR}/half)
endif()

set(half_INCLUDE_DIR ${half_BASE_DIR}/src/ExtHalf/include
  CACHE PATH "'half' include directory")

# for convenience setup a target
add_library(half INTERFACE)
target_include_directories(half INTERFACE ${half_INCLUDE_DIR})

find_path(_half_INCLUDE_DIR
  NAMES half.hpp
  PATHS ${half_INCLUDE_DIR}
  NO_DEFAULT_PATH)

if((NOT _half_INCLUDE_DIR) OR (NOT EXISTS ${half_INCLUDE_DIR}))

  message("'half' library could not be found, so [make] will download it.")

  include(ExternalProject)

  # half-code is a header-only library, so no build and install required
  ExternalProject_Add(ExtHalf
    PREFIX ${half_BASE_DIR}
    URL https://downloads.sourceforge.net/project/half/half/${GEARSHIFFT_HALF_VERSION}/half-${GEARSHIFFT_HALF_VERSION}.zip
    URL_MD5 86d023c0729abf3465bcd55665a39013
    INSTALL_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    LOG_DOWNLOAD ON
    LOG_CONFIGURE OFF
    LOG_BUILD OFF
    )

  # required to trigger download
  add_dependencies(half ExtHalf)
endif()
