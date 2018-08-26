
function(gearshifft_add_executable TARGET BACKEND)

  if(TARGET ${BACKEND})

    add_executable(${TARGET} benchmark.cpp options.cpp)
    target_link_libraries(${TARGET} PRIVATE ${BACKEND})

    #  add_test(NAME ${TARGET} COMMAND ${TARGET} -e 64)

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
