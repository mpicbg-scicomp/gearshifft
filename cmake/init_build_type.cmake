# Default build type to use if none was specified
if(NOT DEFINED GEARSHIFFT_DEFAULT_BUILD_TYPE)
  set(GEARSHIFFT_DEFAULT_BUILD_TYPE "Release")
endif()

set(CMAKE_BUILD_TYPE ${GEARSHIFFT_DEFAULT_BUILD_TYPE} CACHE STRING "Build type")
# Set the possible values of build type for cmake-gui
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
  "Debug"
  "Release"
  "MinSizeRel"
  "RelWithDebInfo"
  )

# sets default build type if none was specified
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to ${GEARSHIFFT_DEFAULT_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE ${GEARSHIFFT_DEFAULT_BUILD_TYPE} CACHE STRING "Build type" FORCE)
endif()
