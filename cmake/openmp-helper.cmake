macro(find_and_add_openmp TARGET)
  set(_QUIET ${ARGN})

  find_package(OpenMP ${_QUIET})

  # Have to check several OPENMP_FOUND due to bug in
  # one version of CMake and the docs (fixed in patch release)
  # OpenMP is missing on macOS llvm default, for example
  if(OpenMP_FOUND OR OPENMP_FOUND OR OpenMP_CXX_FOUND)

    find_package(Threads REQUIRED)
    if(Threads_FOUND)
      # CMake 3.9 FindOpenMP allows correct linking with Clang in more cases
      if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(${TARGET} INTERFACE OpenMP::OpenMP_CXX)
      else()
        # Clang may need -fopenmp=libiomp5 instead, can't be detected here without CMake 3.9
        set_property(TARGET ${TARGET}
          PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
        # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
        set_property(TARGET ${TARGET}
          PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
      endif()
    endif()

  endif() # OpenMP_FOUND

endmacro()
