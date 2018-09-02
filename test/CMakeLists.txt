
function(gearshifft_add_test TARGET BACKEND)

  if(TARGET ${BACKEND})
    set(SOURCES ${ARGN})

    add_executable(${TARGET} ${SOURCES})
    if (NOT Boost_USE_STATIC_LIBS)
      target_compile_definitions(${TARGET} PRIVATE BOOST_TEST_DYN_LINK)
    endif()

    target_link_libraries(${TARGET} ${BACKEND})
    add_test(NAME ${TARGET} COMMAND ${TARGET})
  endif()
endfunction()

if(NOT GEARSHIFFT_TESTS_ADD_CPU_ONLY)
  gearshifft_add_test(test_cufft gearshifft::CUFFT test_cufft.cpp)
endif()
gearshifft_add_test(test_clfft gearshifft::CLFFT test_clfft.cpp)
gearshifft_add_test(test_clfft_global gearshifft::CLFFT test_clfft_global_fixture.cpp)
gearshifft_add_test(test_fftw gearshifft::FFTW test_fftw.cpp)

# test the gearshifft back-end with a small 32-point FFT

if(TARGET gearshifft::FFTW)
  add_test(NAME gearshifft_fftw_list
    COMMAND ${CMAKE_BINARY_DIR}/gearshifft/gearshifft_fftw -l)
  add_test(NAME gearshifft_fftw
    COMMAND ${CMAKE_BINARY_DIR}/gearshifft/gearshifft_fftw -e 32)
endif()

if(TARGET gearshifft::CLFFT)
  add_test(NAME gearshifft_clfft_list
    COMMAND ${CMAKE_BINARY_DIR}/gearshifft/gearshifft_clfft -l)
  add_test(NAME gearshifft_clfft
    COMMAND ${CMAKE_BINARY_DIR}/gearshifft/gearshifft_clfft -e 32)
endif()

if(TARGET gearshifft::FFTWWrappers)
  add_test(NAME gearshifft_fftwwrappers_list
    COMMAND ${CMAKE_BINARY_DIR}/gearshifft/gearshifft_fftwwrappers -l)
  add_test(NAME gearshifft_fftwwrappers
    COMMAND ${CMAKE_BINARY_DIR}/gearshifft/gearshifft_fftwwrappers -e 32)
endif()

if(TARGET gearshifft::CUFFT AND NOT GEARSHIFFT_TESTS_ADD_CPU_ONLY)
  add_test(NAME gearshifft_cufft_list
    COMMAND ${CMAKE_BINARY_DIR}/gearshifft/gearshifft_cufft -l)
  add_test(NAME gearshifft_cufft
    COMMAND ${CMAKE_BINARY_DIR}/gearshifft/gearshifft_cufft -e 32)
endif()
