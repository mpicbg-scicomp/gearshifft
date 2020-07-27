
set(GEARSHIFFT_COMPONENTS "" CACHE STRING "Contains gearshifft targets")
mark_as_advanced(GEARSHIFFT_COMPONENTS)

macro(gearshifft_add_executable TARGET BACKEND)

  if(TARGET ${BACKEND})

    add_executable(${TARGET} benchmark.cpp options.cpp)

    target_compile_definitions(${TARGET} PRIVATE
      GEARSHIFFT_NUMBER_WARM_RUNS=${GEARSHIFFT_NUMBER_WARM_RUNS}
      GEARSHIFFT_NUMBER_WARMUPS=${GEARSHIFFT_NUMBER_WARMUPS}
      GEARSHIFFT_ERROR_BOUND=${GEARSHIFFT_ERROR_BOUND}
      )
    target_link_libraries(${TARGET} PRIVATE ${BACKEND})

    install(
      TARGETS ${TARGET}
      RUNTIME DESTINATION bin
      COMPONENT ${TARGET}
      )
    list(APPEND GEARSHIFFT_COMPONENTS ${TARGET})

  endif()
endmacro()

gearshifft_add_executable(gearshifft_cufft gearshifft::CUFFT)
gearshifft_add_executable(gearshifft_clfft gearshifft::CLFFT)
gearshifft_add_executable(gearshifft_fftw  gearshifft::FFTW)
gearshifft_add_executable(gearshifft_fftwwrappers  gearshifft::FFTWWrappers)
gearshifft_add_executable(gearshifft_rocfft gearshifft::ROCFFT)
gearshifft_add_executable(gearshifft_esslfftw gearshifft::ESSL)
gearshifft_add_executable(gearshifft_armplfftw gearshifft::ARMPL)
