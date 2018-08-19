cmake_minimum_required(VERSION 3.7)

project(gearshifft
  VERSION 0.4.0
  LANGUAGES C CXX)

#-------------------------------------------------------------------------------

# guard against in-source builds and bad build-type strings
include(cmake/safeguards.cmake)

#-------------------------------------------------------------------------------

# cmake module path with additional finders
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/)

#-------------------------------------------------------------------------------

add_subdirectory(src)

include(CTest) # defines: BUILD_TESTING, default is ON

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

#-------------------------------------------------------------------------------

# installs config extents files
#file(GLOB cfg_files "${CMAKE_CURRENT_SOURCE_DIR}/../share/gearshifft/*.conf")
install(
  DIRECTORY share/gearshifft/
  DESTINATION ${GEARSHIFFT_INSTALL_CONFIG_PATH}
  FILES_MATCHING
  PATTERN *.conf
  )
install(
  FILES README.md LICENSE
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  )
