cmake_minimum_required(VERSION 3.1)

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
    target_compile_definitions(${PROJECT_EXEC} PUBLIC -DFFTW_ENABLED -DGEARSHIFFT_FFTW_THREADS=${GEARSHIFFT_FFTW_THREADS})
    set(LIBS ${FFTW_LIBRARIES})
  endif()

  target_link_libraries(${PROJECT_EXEC} ${Boost_LIBRARIES} ${LIBS})
  add_test(NAME ${PROJECT_EXEC} COMMAND ${PROJECT_EXEC} -e 64)

  install(TARGETS ${PROJECT_EXEC} DESTINATION bin)
endfunction()

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../inc)
set(SOURCES benchmark.cpp options.cpp)

# FFT lib clients
foreach(FFTLIB ${FFTLIBS})
  add_exec(${FFTLIB})
endforeach()

# installs config extents files
file(GLOB cfg_files "${CMAKE_CURRENT_SOURCE_DIR}/../share/gearshifft/*.conf")
install(FILES ${cfg_files}  DESTINATION ${GEARSHIFFT_INSTALL_CONFIG_PATH})
