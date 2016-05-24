CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(helper CXX)

set(CMAKE_CXX_STANDARD 14)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "--std=gnu++14" ${CMAKE_CXX_FLAGS})
endif ()
find_package(CUDA)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if (NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gdwarf-2")
  endif()
endif()

set(MY_INCLUDES ${PROJECT_SOURCE_DIR}/../inc)
include_directories(${MY_INCLUDES} ${CUDA_INCLUDE_DIRS})
set(SOURCES ${SOURCES} statistics.cpp)

add_library(${PROJECT_NAME} STATIC ${SOURCES})

# Required on Unix OS family to be able to be linked into shared libraries.
set_target_properties(${PROJECT_NAME}
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${PROJECT_NAME} rt)

# Expose B's public includes to other
# subprojects through cache variable.
set(${PROJECT_NAME}_INCLUDE_DIRS "${MY_INCLUDES}" "${CUDA_INCLUDE_DIRS}"
    CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)