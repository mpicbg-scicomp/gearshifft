
function(gearshifft_add_executable TARGET BACKEND)

  if(TARGET ${BACKEND})

    add_executable(${TARGET} benchmark.cpp options.cpp)

    target_compile_definitions(${TARGET} PRIVATE
      GEARSHIFFT_NUMBER_WARM_RUNS=${GEARSHIFFT_NUMBER_WARM_RUNS}
      GEARSHIFFT_NUMBER_WARMUPS=${GEARSHIFFT_NUMBER_WARMUPS}
      GEARSHIFFT_ERROR_BOUND=${GEARSHIFFT_ERROR_BOUND}
      )
    target_link_libraries(${TARGET} PRIVATE ${BACKEND})

    # test the gearshifft back-end with a small 64-point FFT
    # add_test(NAME ${TARGET} COMMAND ${TARGET} -e 64)

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

gearshifft_add_executable(gearshifft_cufft gearshifft::CUFFT)
gearshifft_add_executable(gearshifft_clfft gearshifft::CLFFT)
gearshifft_add_executable(gearshifft_fftw  gearshifft::FFTW)
gearshifft_add_executable(gearshifft_fftwwrappers  gearshifft::FFTWWrappers)
#gearshifft_add_executable(gearshifft_hcfft gearshifft::HCFFT)
