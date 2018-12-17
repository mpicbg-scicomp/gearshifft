
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../inc)

if(CUDA_FOUND)
  set(TEST_EXEC test_cufft)
  add_executable(${TEST_EXEC} test_cufft.cpp)
  set(LIBS ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
  target_compile_definitions(${TEST_EXEC} PUBLIC -DBOOST_TEST_DYN_LINK)
  target_link_libraries(${TEST_EXEC} ${Boost_LIBRARIES} ${LIBS})
  add_test(NAME ${TEST_EXEC} COMMAND ${TEST_EXEC})
endif()

if(CLFFT_FOUND)
  set(TEST_EXEC test_clfft)
  add_executable(${TEST_EXEC} test_clfft.cpp)
  set(LIBS ${CLFFT_LIBRARIES} ${OpenCL_LIBRARIES})
  target_link_libraries(${TEST_EXEC} ${Boost_LIBRARIES} ${LIBS})
  target_compile_definitions(${TEST_EXEC} PUBLIC -DBOOST_TEST_DYN_LINK)
  add_test(NAME ${TEST_EXEC} COMMAND ${TEST_EXEC})

  set(TEST_EXEC test_clfft_global_fixture)
  add_executable(${TEST_EXEC} test_clfft_global_fixture.cpp)
  set(LIBS ${CLFFT_LIBRARIES} ${OpenCL_LIBRARIES})
  target_link_libraries(${TEST_EXEC} ${Boost_LIBRARIES} ${LIBS})
  target_compile_definitions(${TEST_EXEC} PUBLIC -DBOOST_TEST_DYN_LINK)
  add_test(NAME ${TEST_EXEC} COMMAND ${TEST_EXEC})
endif()

if(FFTW_FOUND)
  set(TEST_EXEC test_fftw)
  add_executable(${TEST_EXEC} test_fftw.cpp)
  set(LIBS ${FFTW_LIBRARIES})
  target_link_libraries(${TEST_EXEC} ${LIBS} ${Boost_LIBRARIES})
  target_compile_definitions(${TEST_EXEC} PUBLIC -DBOOST_TEST_DYN_LINK -DGEARSHIFFT_FFTW_USE_THREADS=${GEARSHIFFT_FFTW_USE_THREADS})
  add_test(NAME ${TEST_EXEC} COMMAND ${TEST_EXEC})
endif()

if(rocfft_FOUND)
  #https://github.com/ROCmSoftwarePlatform/rocFFT/wiki/Example-cpp-code-calling-rocFFT-routine-with--gnu-compiler

  function(add_hip_test_exec exec_name exec_src)
    find_package(hip)
    if(hip_FOUND OR EXISTS hip::hip_hcc)
      add_executable(${exec_name} ${exec_src})
      target_link_libraries(${exec_name} PUBLIC
        roc::rocfft
        hip::hip_hcc
        ${Boost_LIBRARIES})
      target_compile_definitions(${exec_name} PRIVATE -DBOOST_TEST_DYN_LINK)
      target_include_directories(${exec_name} PRIVATE ${rocfft_INCLUDE_DIR})
      target_include_directories(${exec_name} PRIVATE $<TARGET_PROPERTY:hip::hip_hcc,INTERFACE_INCLUDE_DIRECTORIES>)
      add_test(NAME ${exec_name} COMMAND ${exec_name})

      include(CheckCCompilerFlag)
      include(CheckCXXCompilerFlag)
      check_cxx_compiler_flag(-Wno-unused-command-line-argument HAS_NO_WIGNORE_UNUSED_CLI_FLAG)
      if(HAS_NO_WIGNORE_UNUSED_CLI_FLAG)
        target_compile_options( ${exec_name} PRIVATE -Wno-unused-command-line-argument )
      endif()
      set_target_properties( ${exec_name} PROPERTIES CXX_EXTENSIONS OFF)

    else()
      message(FATAL_ERROR "unable to locate hip_hcc required by rocfft")
    endif()
  endfunction()

  add_hip_test_exec(test_rocfft_helper test_rocfft_helper.cpp)
  add_hip_test_exec(test_rocfft test_rocfft.cpp)

endif()
