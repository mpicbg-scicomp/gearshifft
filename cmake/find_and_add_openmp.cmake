macro(find_and_add_openmp TARGET)
  set(_QUIET ${ARGN})

  find_package(OpenMP ${_QUIET})

  # Have to check several OPENMP_FOUND due to bug in
  # one version of CMake and the docs (fixed in patch release)
  # OpenMP is missing on macOS llvm default, for example
  if(OpenMP_FOUND OR OPENMP_FOUND OR OpenMP_CXX_FOUND)

    # CMake 3.9 FindOpenMP allows correct linking with Clang in more cases
    if(NOT TARGET OpenMP::OpenMP_CXX)
      find_package(Threads REQUIRED)
      add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
      # Clang may need -fopenmp=libiomp5 instead, can't be detected here without CMake 3.9
      set_property(TARGET OpenMP::OpenMP_CXX
        PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
      # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
      set_property(TARGET OpenMP::OpenMP_CXX
        PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
    endif()
    target_link_libraries(${TARGET} INTERFACE OpenMP::OpenMP_CXX)
  endif() # OpenMP_FOUND

endmacro()
