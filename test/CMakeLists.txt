
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

gearshifft_add_test(test_cufft gearshifft::CUFFT test_cufft.cpp)
gearshifft_add_test(test_clfft gearshifft::CLFFT test_clfft.cpp)
gearshifft_add_test(test_clfft_global gearshifft::CLFFT test_clfft_global_fixture.cpp)
gearshifft_add_test(test_fftw gearshifft::FFTW test_fftw.cpp)
