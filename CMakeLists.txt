cmake_minimum_required(VERSION 3.7) # better language support, more features
# https://cliutils.gitlab.io/modern-cmake/chapters/intro/newcmake.html
# cmake configurations are inspired by VexCL
# See: https://github.com/ddemidov/vexcl/

if(${CMAKE_VERSION} VERSION_LESS 3.12)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.12)
endif()

#-------------------------------------------------------------------------------

set(GEARSHIFFT_ROOT "${CMAKE_SOURCE_DIR}")

# guard against in-source builds and default build-type
include(cmake/safeguards.cmake)

# cmake module path with additional finders
list(APPEND CMAKE_MODULE_PATH ${GEARSHIFFT_ROOT}/cmake/modules)
#-------------------------------------------------------------------------------

project(gearshifft
  VERSION 0.4.0
  LANGUAGES C CXX)

# gearshifft cmake options (requires GEARSHIFFT_ROOT)
include(${GEARSHIFFT_ROOT}/cmake/gearshifft-options.cmake)

# half-code for half precision data type
if(GEARSHIFFT_FLOAT16_SUPPORT)
  include(${GEARSHIFFT_ROOT}/cmake/half.cmake)
endif()

include(${GEARSHIFFT_ROOT}/cmake/openmp-helper.cmake)

#------------------------------------------------------------------------------

if(GEARSHIFFT_VERBOSE)
  set(_QUIET "")
else()
  set(_QUIET "QUIET")
endif()

#------------------------------------------------------------------------------
# Boost
#------------------------------------------------------------------------------

set(Boost_USE_STATIC_LIBS ${GEARSHIFFT_USE_STATIC_LIBS})
find_package(Boost 1.59 REQUIRED ${_QUIET} COMPONENTS
  system
  unit_test_framework
  program_options
  )

include(${GEARSHIFFT_ROOT}/cmake/BoostTargets.cmake)

#----------------------------------------------------------------------------
# Generic target
#----------------------------------------------------------------------------

add_library(Common INTERFACE)
add_library(gearshifft::Common ALIAS Common)

# setting language requirements for all targets
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# TODO: in cmake 3.8+ use target_compile_features(myTarget PUBLIC cxx_std_11)

target_include_directories(Common INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/inc>
#  $<INSTALL_INTERFACE:include> # include dirs are not installed
#  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> # include dirs are not installed
  )

# do not export this interface (only for internal usage requirements)
target_compile_options(Common INTERFACE
  # g++
  $<$<CXX_COMPILER_ID:GNU>:$<BUILD_INTERFACE:-Wall>>
  $<$<CXX_COMPILER_ID:GNU>:$<BUILD_INTERFACE:-Wextra>>
  $<$<CXX_COMPILER_ID:GNU>:$<BUILD_INTERFACE:-Wold-style-cast>>
#  $<$<CXX_COMPILER_ID:GNU>:$<BUILD_INTERFACE:-Wshadow>> # maybe too many false positives
  $<$<CXX_COMPILER_ID:GNU>:-Wno-missing-braces>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated-declarations>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-ignored-attributes>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-local-typedefs>
  # Clang
  $<$<CXX_COMPILER_ID:Clang>:$<BUILD_INTERFACE:-Wall>>
  $<$<CXX_COMPILER_ID:Clang>:$<BUILD_INTERFACE:-Wextra>>
  $<$<CXX_COMPILER_ID:Clang>:$<BUILD_INTERFACE:-Wshadow>> # see above, does better job than gcc
  $<$<CXX_COMPILER_ID:Clang>:-Wno-missing-braces>
  $<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-declarations>
  $<$<CXX_COMPILER_ID:Clang>:-Wno-ignored-attributes>
  # MSVC
  $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
  $<$<CXX_COMPILER_ID:MSVC>:/wd4003>
  $<$<CXX_COMPILER_ID:MSVC>:/wd4996>
  )

# gearshifft version.
# As this information is built as own library, targets only have to relink,
# when version changes.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gearshifft_version.cpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/gearshifft_version.cpp"
  @ONLY)

add_library(gearshifft_version STATIC
  "${CMAKE_CURRENT_BINARY_DIR}/gearshifft_version.cpp")

target_link_libraries(gearshifft_version PUBLIC Common)

# continue with Common interface used by all gearshifft targets
target_compile_definitions(Common INTERFACE
  $<$<CXX_COMPILER_ID:GNU>:_GLIBCXX_USE_CXX11_ABI=${GEARSHIFFT_CXX11_ABI}>
  $<BUILD_INTERFACE:GEARSHIFFT_INSTALL_CONFIG_FILE=${CMAKE_SOURCE_DIR}/share/gearshifft/extents.conf>
  $<INSTALL_INTERFACE:GEARSHIFFT_INSTALL_CONFIG_FILE=${GEARSHIFFT_INSTALL_CONFIG_PATH}/extents.conf>
  GEARSHIFFT_FLOAT16_SUPPORT=${GEARSHIFFT_FLOAT16_SUPPORT}
  )
#TODO: check paths

target_link_libraries(Common INTERFACE
  gearshifft_version
  Boost::system
  Boost::unit_test_framework
  Boost::program_options
  Boost::thread
  ${CMAKE_DL_LIBS}
  )

#------------------------------------------------------------------------------
# Find gearshifft back-ends
#------------------------------------------------------------------------------

# CUDA+CUFFT
if(GEARSHIFFT_BACKEND_CUFFT)
  find_package(CUDA ${_QUIET}) # no imported targets, will be language in cmake >= 3.10
  if(CUDA_FOUND)

    add_library(CUFFT INTERFACE)
    add_library(gearshifft::CUFFT ALIAS CUFFT)

    target_include_directories(CUFFT INTERFACE ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(CUFFT INTERFACE CUFFT_ENABLED)
    target_link_libraries(CUFFT INTERFACE
      Common
      ${CUDA_LIBRARIES}
      ${CUDA_CUFFT_LIBRARIES}) # TODO: add static cufft library if gearshifft requires it

    if(GEARSHIFFT_FLOAT16_SUPPORT)
      add_dependencies(CUFFT half)
    endif()

    message(STATUS " gearshifft::CUFFT enabled.")
  endif()
endif()

#------------------------------------------------------------------------------
# OPENCL+CLFFT
#------------------------------------------------------------------------------

if(GEARSHIFFT_BACKEND_CLFFT)
  find_package(OpenCL ${_QUIET}) # cmake >= 3.7 provides imported targets
  if(OpenCL_FOUND)
    find_package(clFFT ${_QUIET})

    if(clFFT_FOUND)

      add_library(CLFFT INTERFACE)
      add_library(gearshifft::CLFFT ALIAS CLFFT)

      find_and_add_openmp(CLFFT ${_QUIET})

      target_compile_definitions(CLFFT INTERFACE CLFFT_ENABLED)
      target_link_libraries(CLFFT INTERFACE
        Common
        clFFT::clFFT
        OpenCL::OpenCL
        ${CMAKE_DL_LIBS}
        )

      message(STATUS " gearshifft::CLFFT enabled")
    endif()
  endif()
endif()

#------------------------------------------------------------------------------
# ROCM+HCFFT
#------------------------------------------------------------------------------

#find_package(hcFFT)

#------------------------------------------------------------------------------
# FFTW
#------------------------------------------------------------------------------

if(GEARSHIFFT_BACKEND_FFTW)
  set(FFTW_USE_STATIC_LIBS ${GEARSHIFFT_USE_STATIC_LIBS})
  find_package(FFTW ${_QUIET} COMPONENTS float double) #TODO: only optional float/double in benchmark

  if(FFTW_FOUND)

    add_library(FFTW INTERFACE)
    add_library(gearshifft::FFTW ALIAS FFTW)

    set(GEARSHIFFT_BACKEND_FFTW_THREADS 0)

    if(GEARSHIFFT_BACKEND_FFTW_OPENMP)

      if(FFTW_OPENMP_LIBS)

        find_and_add_openmp(FFTW ${_QUIET})

        if(GEARSHIFFT_OPENMP_FOUND)
          set(GEARSHIFFT_BACKEND_FFTW_THREADS 1)
        else()
          message(WARNING "Could not find OpenMP")
          set(GEARSHIFFT_BACKEND_FFTW_OPENMP OFF)
        endif()
      else()

        message(WARNING "FFTW libraries for OpenMP not found")
        set(GEARSHIFFT_BACKEND_FFTW_OPENMP OFF)
      endif() # FFTW_OPENMP_LIBS
    endif() # GEARSHIFFT_BACKEND_FFTW_OPENMP

    if(GEARSHIFFT_BACKEND_FFTW_PTHREADS)
      if(FFTW_THREADS_LIBS)
        find_package(Threads)
        set(GEARSHIFFT_BACKEND_FFTW_THREADS 1)
      else()
        message(WARNING "FFTW libraries for pthreads not found")
        set(GEARSHIFFT_BACKEND_FFTW_PTHREADS OFF)
      endif()
    endif()

    if(GEARSHIFFT_BACKEND_FFTW_PTHREADS AND GEARSHIFFT_BACKEND_FFTW_OPENMP)
      message(FATAL_ERROR "FFTW: Cannot use both OpenMP and pthreads, please disable one")
    endif()

    if(GEARSHIFFT_BACKEND_FFTW_PTHREADS
        OR GEARSHIFFT_BACKEND_FFTW_OPENMP
        OR FFTW_SERIAL_LIBS)

      target_include_directories(FFTW INTERFACE ${FFTW_INCLUDE_DIR})
      target_compile_definitions(FFTW INTERFACE
        FFTW_ENABLED
        GEARSHIFFT_USE_FFTW_THREADS=${GEARSHIFFT_USE_FFTW_THREADS})
      target_link_libraries(FFTW INTERFACE
        Common
        $<$<STREQUAL:${GEARSHIFFT_BACKEND_FFTW_OPENMP},ON>:${FFTW_OPENMP_LIBS}>
        $<$<STREQUAL:${GEARSHIFFT_BACKEND_FFTW_PTHREADS},ON>:${FFTW_THREADS_LIBS}>
        ${FFTW_SERIAL_LIBS}
        $<$<BOOL:${Threads_FOUND}>:Threads::Threads>
        )

      message(STATUS " gearshifft::FFTW enabled.")
    endif()
  endif()
endif()

#------------------------------------------------------------------------------
# FFTW Wrappers of MKL
#------------------------------------------------------------------------------

# TODO: add backend cmake option for FFTWWrappers
find_package(FFTWWrappers ${_QUIET})
if(FFTWWrappers_FOUND) # found FFTWWrappers for GNU or Intel

  if(FFTWWrappers_GNU_LIBRARIES AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU"
      AND FFTWWrappers_INTEL_LIBRARIES AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    message(WARNING "Compiler does not match found FFTWWrappers")
  else()

    add_library(FFTWWrappers INTERFACE)
    add_library(gearshifft::FFTWWrappers ALIAS FFTWWrappers)

    # if compiler supports OpenMP then add flag, as FFTWWrappers might need it
    find_and_add_openmp(FFTWWrappers ${_QUIET})

    if(GEARSHIFFT_OPENMP_FOUND)
      set(GEARSHIFFT_BACKEND_FFTW_THREADS 1)
    endif()

    target_include_directories(FFTWWrappers INTERFACE ${FFTWWrappers_MKL_INCLUDE_DIR})
    target_compile_definitions(FFTWWrappers INTERFACE
      FFTW_ENABLED
      GEARSHIFFT_USE_FFTW_THREADS=${GEARSHIFFT_USE_FFTW_THREADS})
    target_link_libraries(FFTWWrappers INTERFACE
      Common
#      $<$<BOOL:${Threads_FOUND}>:Threads::Threads>
      )

    if(FFTWWrappers_GNU_LIBRARIES)
      target_link_libraries(FFTWWrappers INTERFACE
        $<$<CXX_COMPILER_ID:GNU>:${FFTWWrappers_GNU_LIBRARIES}>
        $<$<CXX_COMPILER_ID:Clang>:${FFTWWrappers_GNU_LIBRARIES}>
        )
    endif()

    if(FFTWWrappers_INTEL_LIBRARIES)
      target_link_libraries(FFTWWrappers INTERFACE
        $<$<CXX_COMPILER_ID:Intel>:${FFTWWrappers_INTEL_LIBRARIES}>
        )
    endif()

    target_link_libraries(FFTWWrappers INTERFACE
      -Wl,--start-group ${FFTWWrappers_MKL_LIBRARIES} -Wl,--end-group
      )

    message(STATUS " gearshifft::FFTWWrappers enabled.")

  endif() # compiler match fftwwrappers

endif()


#-------------------------------------------------------------------------------

if(NOT TARGET gearshifft::CUFFT
    AND NOT TARGET gearshifft::CLFFT
    AND NOT TARGET gearshifft::FFTW
    AND NOT TARGET gearshifft::FFTWWrappers
    )
  message(WARNING "No FFT backend found.")
endif()

#-------------------------------------------------------------------------------

add_subdirectory(gearshifft)

include(CTest) # defines: BUILD_TESTING, default is ON

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

#-------------------------------------------------------------------------------

# installs config extents files

install(
  DIRECTORY share/gearshifft
  DESTINATION share
  COMPONENT config
  FILES_MATCHING
  PATTERN *.conf
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ
              WORLD_READ
  )
install(
  FILES README.md
  DESTINATION doc
  COMPONENT doc
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ
              WORLD_READ
  )

#-------------------------------------------------------------------------------

# packaging
# check that we are in top-level CMakeLists
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_subdirectory(packaging)
  # include(CPack) will happen in here
endif()
