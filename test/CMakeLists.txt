
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
  set(TEST_EXEC test_rocfft)
  find_package(hip)
  if(hip_FOUND OR EXISTS hip::hip_hcc)
    add_executable(${TEST_EXEC} test_rocfft.cpp)
    target_link_libraries(${TEST_EXEC} PUBLIC
      roc::rocfft
      hip::hip_hcc
      ${Boost_LIBRARIES})
    target_compile_definitions(${TEST_EXEC} PRIVATE -DBOOST_TEST_DYN_LINK)
    target_include_directories(${TEST_EXEC} PRIVATE ${rocfft_INCLUDE_DIR})
    target_include_directories(${TEST_EXEC} PRIVATE $<TARGET_PROPERTY:hip::hip_hcc,INTERFACE_INCLUDE_DIRECTORIES>)
    add_test(NAME ${TEST_EXEC} COMMAND ${TEST_EXEC})

    include(CheckCCompilerFlag)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-Wno-unused-command-line-argument HAS_NO_WIGNORE_UNUSED_CLI_FLAG)
    if(HAS_NO_WIGNORE_UNUSED_CLI_FLAG)
      target_compile_options( ${TEST_EXEC} PRIVATE -Wno-unused-command-line-argument )
    endif()
    set_target_properties( ${TEST_EXEC} PROPERTIES CXX_EXTENSIONS OFF)

  else()
    message(FATAL_ERROR "unable to locate hip_hcc required by rocfft")
  endif()
endif()
