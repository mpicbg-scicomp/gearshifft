
set(GEARSHIFFT_LINUX_DISTRO OFF CACHE BOOL "Name of linux distribution.")
mark_as_advanced(GEARSHIFFT_LINUX_DISTRO)

macro(get_linux_distro_name)

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")

    find_program(LSB_RELEASE_EXEC lsb_release)

    if(LSB_RELEASE_EXEC)
      execute_process(COMMAND ${LSB_RELEASE_EXEC} -is
        OUTPUT_VARIABLE GEARSHIFFT_LINUX_DISTRO
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()
    mark_as_advanced(LSB_RELEASE_EXEC)
  endif()

endmacro()
