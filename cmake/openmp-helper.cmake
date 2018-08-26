
set(GEARSHIFFT_OPENMP_FOUND OFF CACHE BOOL "OpenMP has been found, do not find again.")
mark_as_advanced(GEARSHIFFT_OPENMP_FOUND)

function(find_and_add_openmp TARGET)
  set(_QUIET ${ARGN})

  if(NOT GEARSHIFFT_OPENMP_FOUND)

    find_package(OpenMP ${_QUIET})

    # Have to check several OPENMP_FOUND due to bug in
    # one version of CMake and the docs (fixed in patch release)
    # OpenMP is missing on macOS llvm default, for example
    if(OpenMP_FOUND OR OPENMP_FOUND OR OpenMP_CXX_FOUND)

      set(GEARSHIFFT_OPENMP_FOUND ON PARENT_SCOPE)

      # CMake 3.9 FindOpenMP allows correct linking with Clang in more cases
      if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(${TARGET} INTERFACE OpenMP::OpenMP_CXX)
      else()
        # Clang may need -fopenmp=libiomp5 instead, can't be detected here without CMake 3.9
        target_link_libraries(${TARGET} INTERFACE
          $<$<CXX_COMPILER_ID:GNU>:${OpenMP_CXX_FLAGS}>
          $<$<CXX_COMPILER_ID:Clang>:${OpenMP_CXX_FLAGS}>
          $<$<CXX_COMPILER_ID:Intel>:${OpenMP_CXX_FLAGS}>
          )
        target_compile_options(${TARGET} INTERFACE ${OpenMP_CXX_FLAGS})
      endif()

    endif() # OpenMP_FOUND

  endif()

endfunction()
