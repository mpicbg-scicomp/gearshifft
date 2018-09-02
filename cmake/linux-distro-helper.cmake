
set(GEARSHIFFT_LINUX_DISTRO OFF CACHE BOOL "Name of linux distribution.")
mark_as_advanced(GEARSHIFFT_LINUX_DISTRO)

macro(get_linux_distro_name)

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")

    find_program(LSB_RELEASE_EXEC lsb_release)
    mark_as_advanced(LSB_RELEASE_EXEC)

    if(LSB_RELEASE_EXEC)
      execute_process(COMMAND ${LSB_RELEASE_EXEC} -is
        OUTPUT_VARIABLE GEARSHIFFT_LINUX_DISTRO
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    else()
      find_path(OS_RELEASE
        NAMES "os-release"
        PATHS /etc
        NO_DEFAULT_PATH)
      if(OS_RELEASE)
        execute_process(COMMAND "awk -F= '/^NAME/{print $2}' /etc/os-release"
          OUTPUT_VARIABLE GEARSHIFFT_LINUX_DISTRO
          OUTPUT_STRIP_TRAILING_WHITESPACE
          )
      endif()
    endif()
  endif()

endmacro()
