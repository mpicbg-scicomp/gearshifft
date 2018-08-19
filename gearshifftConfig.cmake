cmake_minimum_required(VERSION 3.7)

#-------------------------------------------------------------------------------
# gearshifft cmake options

option(GEARSHIFFT_VERBOSE "Verbose output during build generation." ON)
option(GEARSHIFFT_CXX11_ABI "enable _GLIBCXX_USE_CXX11_ABI in GCC 5.0+" ON)
option(GEARSHIFFT_FFTW_OPENMP "use OpenMP parallel fftw libraries if found" ON)
option(GEARSHIFFT_FFTW_PTHREADS "use pthreads parallel fftw libraries if found" OFF)
option(GEARSHIFFT_CUFFT "Compile gearshifft_cufft if available?" ON)
option(GEARSHIFFT_CLFFT "Compile gearshifft_clfft if available?" ON)
option(GEARSHIFFT_FFTW  "Compile gearshifft_fftw if available?" ON)
option(GEARSHIFFT_HCFFT "< Not implemented yet >" OFF)

set(GEARSHIFFT_FLOAT16_SUPPORT "0" CACHE STRING "Enable float16 data type (cufft and ComputeCapability 5.3+ only)")
set_property(CACHE GEARSHIFFT_FLOAT16_SUPPORT PROPERTY STRINGS "0;1")

set(GEARSHIFFT_NUMBER_WARM_RUNS "10" CACHE STRING "Number of repetitions of an FFT benchmark after a warmup.")
set(GEARSHIFFT_NUMBER_WARMUPS "2" CACHE STRING "Number of warmups of an FFT benchmark.")
set(GEARSHIFFT_ERROR_BOUND "0.00001" CACHE STRING "Error-bound for FFT benchmarks (<0 for dynamic error bound).")
set(GEARSHIFFT_INSTALL_CONFIG_PATH "${CMAKE_INSTALL_PREFIX}/share/gearshifft/" CACHE STRING "Default install path of config files.")

#-------------------------------------------------------------------------------

if(GEARSHIFFT_HCFFT)
  message(FATAL_ERROR "HCFFT is not implemented yet.")
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

  if (NOT (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0
        OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0))
    message(FATAL_ERROR "gearshifft requires g++ 5.0 or greater.")
  endif()
  #gcc4.8+ uses dwarf-4. If you have gdb <7.5 then use this line
  #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -gdwarf-2")
  #gdb 7.0-7.5 may require
  #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fvar-tracking-assignments")
endif()

#-------------------------------------------------------------------------------

# setting language requirements for all targets
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#------------------------------------------------------------------------------
# half-code for half precision data type
#------------------------------------------------------------------------------

if(GEARSHIFFT_FLOAT16_SUPPORT)
  # message(STATUS "half-precision support enabled.")
  # message(STATUS " 'make' will check out half-code library for half-precision data types.")
  # message(STATUS " If 'make' or 'make half-code' fails, then download half-code: 'http://sourceforge.net/projects/half/files/latest/download' to 'third-party/src/'.")

  include(half)
endif()

#------------------------------------------------------------------------------
# Threads
#------------------------------------------------------------------------------

find_package(Threads)

#------------------------------------------------------------------------------
# Boost
#------------------------------------------------------------------------------

find_package(Boost 1.59 REQUIRED
  COMPONENTS system unit_test_framework program_options)

#------------------------------------------------------------------------------
# CUDA+CUFFT
#------------------------------------------------------------------------------

find_package(CUDA)

#------------------------------------------------------------------------------
# OPENCL+CLFFT
#------------------------------------------------------------------------------

find_package(OpenCL)
if(OpenCL_FOUND AND GEARSHIFFT_CLFFT)
  find_package(clFFT)
endif()

#------------------------------------------------------------------------------
# ROCM+HCFFT
#------------------------------------------------------------------------------

find_package(hcFFT)

#------------------------------------------------------------------------------
# FFTW
#------------------------------------------------------------------------------

find_package(FFTW COMPONENTS float double)
if(FFTW_FOUND AND GEARSHIFFT_FFTW)
  find_package(OpenMP)

  foreach(_LIBSTEM IN LISTS FFTW_SERIAL_LIBS)
    list(APPEND FFTW_LIBS_TO_USE ${_LIBSTEM})
  endforeach()

  set(GEARSHIFFT_FFTW_THREADS 0)
  if(GEARSHIFFT_FFTW_OPENMP AND OPENMP_FOUND AND "${FFTW_LIBRARIES}" MATCHES ".*_omp.*")
    foreach(_LIBSTEM IN LISTS FFTW_OPENMP_LIBS)
      list(APPEND FFTW_LIBS_TO_USE ${_LIBSTEM})
    endforeach()
    set(GEARSHIFFT_FFTW_THREADS 1)
  endif()

  if(GEARSHIFFT_FFTW_PTHREADS AND "${FFTW_LIBRARIES}" MATCHES ".*_threads.*")
    foreach(_LIBSTEM IN LISTS FFTW_THREADS_LIBS)
      list(APPEND FFTW_LIBS_TO_USE ${_LIBSTEM})
    endforeach()
    set(GEARSHIFFT_FFTW_THREADS 1)
  endif()
endif()

#------------------------------------------------------------------------------
# gearshifft
#------------------------------------------------------------------------------

# Generating version information for gearshifft sources.
# As this information is built as own library, targets only have to relink, when version changes.
configure_file(gearshifft_version.cpp.in gearshifft_version.cpp @ONLY)

add_library(gearshifft_version STATIC ${CMAKE_CURRENT_BINARY_DIR}/gearshifft_version.cpp)
if(GEARSHIFFT_CXX11_ABI AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_definitions(
    gearshifft_version PUBLIC -D_GLIBCXX_USE_CXX11_ABI=1)
endif()
target_include_directories(
  gearshifft_version PUBLIC ${CMAKE_SOURCE_DIR}/inc/)


# Helper function to add specific FFT library benchmark
function(gearshifft_add_executable Tlib TARGET)

  cmake_parse_arguments(
    ARG
    "INSTALL"
    "TARGET"
    "SOURCES"
    ${ARGN})
  set(Sources ${ARG_SOURCES})

  add_executable(${TARGET} ${Sources})


  # common properties
  target_include_directories(
    ${TARGET} PRIVATE
    ${Boost_INCLUDE_DIR}
    )
  target_compile_definitions(
    ${TARGET} PRIVATE
    -DGEARSHIFFT_NUMBER_WARM_RUNS=${GEARSHIFFT_NUMBER_WARM_RUNS}
    -DGEARSHIFFT_NUMBER_WARMUPS=${GEARSHIFFT_NUMBER_WARMUPS}
    -DGEARSHIFFT_ERROR_BOUND=${GEARSHIFFT_ERROR_BOUND}
    -DGEARSHIFFT_INSTALL_CONFIG_FILE=${GEARSHIFFT_INSTALL_CONFIG_PATH}/extents.conf
    -DGEARSHIFFT_FLOAT16_SUPPORT=${GEARSHIFFT_FLOAT16_SUPPORT}
    )
  target_link_libraries(
    ${TARGET} PRIVATE
    gearshifft_version
    ${Boost_LIBRARIES}
    Threads::Threads
    )

  if (Tlib STREQUAL "cufft")
    # cufft target

    target_include_directories(
      ${TARGET} PRIVATE
      ${CUDA_INCLUDE_DIRS}
      )
    target_compile_definitions(
      ${TARGET} PRIVATE
      -DCUDA_ENABLED
      )
    target_link_libraries(
      ${TARGET} PRIVATE
      ${CUDA_LIBRARIES}
      ${CUDA_CUFFT_LIBRARIES}
      )

    if(GEARSHIFFT_FLOAT16_SUPPORT)
      add_dependencies(${TARGET} half)
    endif()

  elseif (Tlib STREQUAL "clfft")
    # clfft target

    target_compile_definitions(
      ${TARGET} PRIVATE
      -DOPENCL_ENABLED
      )
    target_link_libraries(
      ${TARGET} PRIVATE
      ${CLFFT_LIBRARIES} ${OpenCL_LIBRARIES}
      )

  elseif (Tlib STREQUAL "fftw")
    # fftw target

    target_include_directories(
      ${TARGET} PRIVATE
      ${FFTW_INCLUDE_DIR}
      )
    target_compile_definitions(
      ${TARGET} PRIVATE
      -DFFTW_ENABLED
      -DGEARSHIFFT_FFTW_THREADS=${GEARSHIFFT_FFTW_THREADS}
      )
    target_link_libraries(
      ${TARGET} PRIVATE
      ${FFTW_LIBRARIES}
      )

  elseif (Tlib STREQUAL "hcfft")
    message(FATAL_ERROR "hcfft not supported.")

  endif()

  #  target_link_libraries(${TARGET} ${CMAKE_THREAD_LIBS_INIT})

  #  add_test(NAME ${TARGET} COMMAND ${TARGET} -e 64)
  if(ARG_INSTALL AND CMAKE_INSTALL_PREFIX)
    install(
      TARGETS ${TARGET}
      EXPORT gearshifft
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/static
      COMPONENT gearshifft
      )
  endif()
endfunction()

# lib based add-executable functions

function(gearshifft_add_cufft_executable TARGET)
  gearshifft_add_executable(cufft ${TARGET} ${ARGN})
endfunction()

function(gearshifft_add_clfft_executable TARGET)
  gearshifft_add_executable(clfft ${TARGET} ${ARGN})
endfunction()

function(gearshifft_add_fftw_executable TARGET)
  gearshifft_add_executable(fftw  ${TARGET} ${ARGN})
endfunction()

function(gearshifft_add_hcfft_executable TARGET)
  gearshifft_add_executable(hcfft ${TARGET} ${ARGN})
endfunction()
