cmake_minimum_required(VERSION 3.1)

#------------------------------------------------------------------------------
# Helper function to add specific FFT library benchmark
#------------------------------------------------------------------------------

function(add_exec Tlib)
  set(PROJECT_EXEC gearshifft_${Tlib})

  add_executable(${PROJECT_EXEC} benchmark.cpp options.cpp)

  target_include_directories(${PROJECT_EXEC} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../inc")
  target_compile_definitions(${PROJECT_EXEC} PUBLIC
    -DGEARSHIFFT_NUMBER_WARM_RUNS=${GEARSHIFFT_NUMBER_WARM_RUNS}
    -DGEARSHIFFT_NUMBER_WARMUPS=${GEARSHIFFT_NUMBER_WARMUPS}
    -DGEARSHIFFT_ERROR_BOUND=${GEARSHIFFT_ERROR_BOUND}
    -DGEARSHIFFT_INSTALL_CONFIG_FILE=${GEARSHIFFT_INSTALL_CONFIG_PATH}/extents.conf
    )
  if (Tlib STREQUAL "cufft")
    target_compile_definitions(${PROJECT_EXEC} PUBLIC -DCUDA_ENABLED)
    set(LIBS ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
    if(GEARSHIFFT_FLOAT16_SUPPORT)
      add_dependencies(${PROJECT_EXEC} half-code)
    endif()
  elseif(Tlib STREQUAL "clfft")
    target_compile_definitions(${PROJECT_EXEC} PUBLIC -DOPENCL_ENABLED)
    set(LIBS ${CLFFT_LIBRARIES} ${OpenCL_LIBRARIES})
  endif()
  if (Tlib STREQUAL "fftw")
    target_compile_definitions(${PROJECT_EXEC} PUBLIC -DFFTW_ENABLED -DGEARSHIFFT_FFTW_USE_THREADS=${GEARSHIFFT_FFTW_USE_THREADS})
    set(LIBS ${FFTW_LIBRARIES})
  endif()
  if (Tlib STREQUAL "fftwwrappers")
    target_compile_definitions(${PROJECT_EXEC} PUBLIC -DFFTW_ENABLED -DGEARSHIFFT_FFTW_USE_THREADS=${GEARSHIFFT_FFTW_USE_THREADS})

    target_include_directories(${PROJECT_EXEC} PUBLIC ${FFTWWrappers_MKL_INCLUDE_DIR})

    set(LIBS ${LIBS} -Wl,--start-group ${FFTWWrappers_MKL_LIBRARIES} -Wl,--end-group ${FFTWWrappers_LIBRARIES})
  endif()

  if (Tlib STREQUAL "rocfft")

    target_compile_definitions(${PROJECT_EXEC} PUBLIC -DROCFFT_ENABLED)
    target_include_directories(${PROJECT_EXEC} PRIVATE ${rocfft_INCLUDE_DIR})
    target_include_directories(${PROJECT_EXEC} PRIVATE $<TARGET_PROPERTY:hip::hip_hcc,INTERFACE_INCLUDE_DIRECTORIES>)
    set(LIBS roc::rocfft hip::hip_hcc)
    if(HAS_NO_WIGNORE_UNUSED_CLI_FLAG)
      target_compile_options( ${PROJECT_EXEC} PRIVATE -Wno-unused-command-line-argument )
    endif()
    set_target_properties( ${PROJECT_EXEC} PROPERTIES CXX_EXTENSIONS OFF)

  endif()

  target_link_libraries(${PROJECT_EXEC} ${Boost_LIBRARIES} ${LIBS} ${CMAKE_THREAD_LIBS_INIT})
  add_test(NAME ${PROJECT_EXEC} COMMAND ${PROJECT_EXEC} -e 64)

  install(TARGETS ${PROJECT_EXEC} DESTINATION bin)
endfunction()

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

# FFT lib clients
foreach(FFTLIB ${FFTLIBS})
  add_exec(${FFTLIB})
endforeach()

# installs config extents files
file(GLOB cfg_files "${CMAKE_CURRENT_SOURCE_DIR}/../share/gearshifft/*.conf")
install(FILES ${cfg_files}  DESTINATION ${GEARSHIFFT_INSTALL_CONFIG_PATH})
