cmake_minimum_required(VERSION 2.8)

project(gearshifft CXX)

list(APPEND CMAKE_MODULE_PATH "../cmake")

set(CMAKE_CXX_STANDARD 14)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if (NOT (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0
        OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0))
    message(FATAL_ERROR "${PROJECT_NAME} requires g++ 5.0 or greater.")
  endif()
  set(CMAKE_CXX_FLAGS "--std=gnu++14" ${CMAKE_CXX_FLAGS})
  #gcc4.8+ uses dwarf-4. If you have gdb <7.0 then use this line
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG} -gdwarf-2")
  #gdb 7.0-7.5 may require
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG} -fvar-tracking-assignments")
endif()

find_package(Boost 1.56 QUIET COMPONENTS unit_test_framework REQUIRED)
if(Boost_FOUND)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
endif()

set(PRECISION "" CACHE STRING "Specify precision [float,double].")
#------------------------------------------------------------------------------
# CUDA+CUFFT
#------------------------------------------------------------------------------
set(CUDA_CC "" CACHE STRING "Specify your CUDA ComputeCapability")
find_package(CUDA)
if(CUDA_FOUND)
  include(FindCUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    #for gcc5.x and cuda
    list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")
  endif()
  # --- CUDA device architectures ---
  if(CUDA_CC)
    message(">> Set CUDA gencode for ComputeCapability ${CUDA_CC}")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${CUDA_CC},code=sm_${CUDA_CC}")
  else()
    message(">> All CUDA gencodes are created")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=sm_20")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=sm_30")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_32,code=sm_32")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_35,code=sm_35")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_50,code=sm_50")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_53,code=sm_53")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_53,code=compute_53")
  endif(CUDA_CC)
  list(APPEND FFTLIBS "cufft")
  message(">> cuFFT -> " ${CUDA_CUFFT_LIBRARIES})
else()
  message("cuFFT benchmark disabled.")
endif()

#------------------------------------------------------------------------------
# OPENCL+CLFFT
#------------------------------------------------------------------------------
find_package(OpenCL)
if(OPENCL_FOUND)
  find_package(clFFT)
  if(CLFFT_FOUND)
    include_directories(${CLFFT_INCLUDE_DIRS})
    list(APPEND FFTLIBS "clfft")
    message(">> clFFT -> " ${CLFFT_LIBRARIES})
  endif()
endif()

if(NOT CLFFT_FOUND)
  message("clFFT benchmark disabled.")
endif()
if(NOT CLFFT_FOUND AND NOT CUDA_FOUND)
  message(FATAL_ERROR "-> No FFT library for benchmark found !!!")
endif()

#------------------------------------------------------------------------------
# Helper function to add specific FFT library benchmark
#------------------------------------------------------------------------------
function(add_exec Tlib Ttype)
  message(">> Target: ${Tlib} ${Ttype}")
  set(PROJECT_EXEC ${PROJECT_NAME}_${Tlib}_${Ttype})
  set(PROJECT_EXEC_FLAGS "-DBENCH_PRECISION=${Ttype} -DDEFAULT_RESULT_FILE=${PROJECT_EXEC}")
  add_executable(${PROJECT_EXEC} ${SOURCES})
  if (Tlib STREQUAL "cufft")
    set_target_properties(${PROJECT_EXEC}
      PROPERTIES COMPILE_FLAGS "-DCUDA_ENABLED ${PROJECT_EXEC_FLAGS}")
    set(LIBS ${LIBS} ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
  endif()
  if (Tlib STREQUAL "clfft")
    set_target_properties(${PROJECT_EXEC}
      PROPERTIES COMPILE_FLAGS "-DOPENCL_ENABLED ${PROJECT_EXEC_FLAGS}")
    set(LIBS ${LIBS} ${CLFFT_LIBRARIES} ${OPENCL_LIBRARIES})
  endif()
  target_link_libraries(${PROJECT_EXEC} ${Boost_LIBRARIES} ${LIBS})
  add_test(NAME ${PROJECT_EXEC} COMMAND ${PROJECT_EXEC})
endfunction()

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
include_directories(../inc)
set(SOURCES benchmark.cpp)

enable_testing()
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --verbose)

foreach(FFTLIB ${FFTLIBS})
  if(PRECISION)
    add_exec(${FFTLIB} ${PRECISION})
  else()
    add_exec(${FFTLIB} "float")
    add_exec(${FFTLIB} "double")
  endif()
endforeach()
