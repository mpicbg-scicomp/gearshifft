cmake_minimum_required(VERSION 2.8)

project(gearshifft CXX)

include(CheckCXXCompilerFlag)

if (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|Intel)$")
  CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)
  if (HAS_CPP14_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  else()
    message(FATAL_ERROR "gearshifft requires C++14 support!")
  endif()
endif()

#------------------------------------------------------------------------------
# Boost
#------------------------------------------------------------------------------

find_package(Boost 1.56 QUIET COMPONENTS system unit_test_framework program_options REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
endif()

#------------------------------------------------------------------------------
# CUDA+CUFFT
#------------------------------------------------------------------------------

find_package(CUDA)
if(CUDA_FOUND AND GEARSHIFFT_CUFFT)
  include(FindCUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  list(APPEND FFTLIBS "cufft")
  message(">> cuFFT -> " ${CUDA_CUFFT_LIBRARIES})
else()
  message("<< cuFFT benchmark disabled.")
endif()

#------------------------------------------------------------------------------
# OPENCL+CLFFT
#------------------------------------------------------------------------------

find_package(OpenCL)
if(OPENCL_FOUND AND GEARSHIFFT_CLFFT)
  find_package(clFFT)
  if(CLFFT_FOUND)
    include_directories(${CLFFT_INCLUDE_DIRS})
    list(APPEND FFTLIBS "clfft")
    string(REPLACE ";" " " local_clfft_libraries "${CLFFT_LIBRARIES}")
    message(">> clFFT -> " ${local_clfft_libraries})
  endif()
endif()

if(NOT CLFFT_FOUND)
  message("<< clFFT benchmark disabled.")
endif()

#------------------------------------------------------------------------------
# ROCM+HCFFT
#------------------------------------------------------------------------------

find_package(hcFFT)
if(HCFFT_FOUND AND GEARSHIFFT_HCFFT)
  include_directories(${HCFFT_INCLUDE_DIRS})
  list(APPEND FFTLIBS "hcfft")
  message(">> hcFFT -> " ${HCFFT_LIBRARIES} " " ${HCFFT_INCLUDE_DIRS})
  message(">> Warning: no runner for hcfft implemented yet")
endif()

if(NOT HCFFT_FOUND)
  message("<< hcFFT benchmark disabled.")
endif()

#------------------------------------------------------------------------------
# FFTW
#------------------------------------------------------------------------------

find_package(FFTW COMPONENTS float double)
if(FFTW_FOUND AND GEARSHIFFT_FFTW)
  include_directories(${FFTW_INCLUDE_DIR})
  link_directories(${FFTW_LIBRARY_DIR})
  list(APPEND FFTLIBS fftw)
  find_package(OpenMP)

  foreach(_LIBSTEM IN LISTS FFTW_SERIAL_LIBS)
    list(APPEND FFTW_LIBS_TO_USE ${_LIBSTEM})
  endforeach()

  if(GEARSHIFFT_FFTW_OPENMP AND OPENMP_FOUND AND "${FFTW_LIBRARIES}" MATCHES ".*_omp.*")
    foreach(_LIBSTEM IN LISTS FFTW_OPENMP_LIBS)
      list(APPEND FFTW_LIBS_TO_USE ${_LIBSTEM})
    endforeach()
  endif()

  if(GEARSHIFFT_FFTW_PTHREADS AND "${FFTW_LIBRARIES}" MATCHES ".*_threads.*")
    foreach(_LIBSTEM IN LISTS FFTW_THREADS_LIBS)
      list(APPEND FFTW_LIBS_TO_USE ${_LIBSTEM})
    endforeach()

  endif()

  set(FFTW_LIBRARIES ${FFTW_LIBS_TO_USE})
  message(">> FFTW -> ${FFTW_LIBRARIES} ${FFTW_INCLUDES}")
else()
  message("<< FFTW benchmark disabled.")
endif()


if(NOT FFTLIBS)
  message(FATAL_ERROR ">> No FFT library for benchmark found !!!")
endif()

#------------------------------------------------------------------------------
# Helper function to add specific FFT library benchmark
#------------------------------------------------------------------------------

function(add_exec Tlib)
  set(PROJECT_EXEC gearshifft_${Tlib})
  add_executable(${PROJECT_EXEC} ${SOURCES})
  target_compile_definitions(${PROJECT_EXEC} PUBLIC
    -DGEARSHIFFT_NUMBER_WARM_RUNS=${GEARSHIFFT_NUMBER_WARM_RUNS}
    -DGEARSHIFFT_NUMBER_WARMUPS=${GEARSHIFFT_NUMBER_WARMUPS}
    -DGEARSHIFFT_ERROR_BOUND=${GEARSHIFFT_ERROR_BOUND}
    )
  if (Tlib STREQUAL "cufft" OR DEV_TESTS STREQUAL "cufft")
    target_compile_definitions(${PROJECT_EXEC} PUBLIC -DCUDA_ENABLED)
    set(LIBS ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
  elseif(Tlib STREQUAL "clfft" OR DEV_TESTS STREQUAL "clfft")
    target_compile_definitions(${PROJECT_EXEC} PUBLIC -DOPENCL_ENABLED)
    set(LIBS ${CLFFT_LIBRARIES} ${OPENCL_LIBRARIES})
  endif()
  if (Tlib STREQUAL "fftw")
    target_compile_definitions(${PROJECT_EXEC} PUBLIC -DFFTW_ENABLED)
    set(LIBS ${LIBS} ${FFTW_LIBRARIES})
  endif()

  target_link_libraries(${PROJECT_EXEC} ${Boost_LIBRARIES} ${LIBS})
  add_test(NAME ${PROJECT_EXEC} COMMAND ${PROJECT_EXEC})
endfunction()

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../inc)

enable_testing()
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --verbose)

if(DEV_TESTS)
  set(SOURCES ${TESTS})
  add_exec("test")
else()
  set(SOURCES benchmark.cpp options.cpp)
  foreach(FFTLIB ${FFTLIBS})
    add_exec(${FFTLIB})
  endforeach()
endif()
