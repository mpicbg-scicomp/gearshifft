cmake_minimum_required(VERSION 2.8)

project(cufft_test CXX)

set(CMAKE_CXX_STANDARD 14)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if (NOT (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0
        OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0))
    message(FATAL_ERROR "${PROJECT_NAME} requires g++ 5.0 or greater.")
  endif()
  set(CMAKE_CXX_FLAGS "--std=gnu++14" ${CMAKE_CXX_FLAGS})
  #gcc4.8+ uses dwarf-4. If you have gdb <7.5 then use this line
  #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -gdwarf-2")
  #gdb 7.0-7.5 may require
  #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fvar-tracking-assignments")
endif()

#------------------------------------------------------------------------------
# CUDA+CUFFT
#------------------------------------------------------------------------------
find_package(CUDA)
if(CUDA_FOUND)
  include(FindCUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  list(APPEND FFTLIBS "cufft")
  message(">> cuFFT -> " ${CUDA_CUFFT_LIBRARIES})
endif()

set(LIBS ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
add_executable(cufft_time_r2c time_cufft_r2c.cpp)
target_link_libraries(cufft_time_r2c ${LIBS})
