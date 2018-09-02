cmake_minimum_required(VERSION 3.7)

set(CPACK_PACKAGE_NAME gearshifft)
set(CPACK_PACKAGE_VENDOR mpicbg-scicomp)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "FFT library benchmark suite for cuFFT, clFFT and FFTW")

set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
set(CPACK_PACKAGE_VERSION_MAJOR     ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR     ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH     ${PROJECT_VERSION_PATCH})
# escapes content that is written to cpack files
set(CPACK_VERBATIM_VARIABLES        YES)
set(CPACK_PACKAGE_DESCRIPTION_FILE  ${CMAKE_CURRENT_LIST_DIR}/Description.txt)
set(CPACK_RESOURCE_FILE_WELCOME     ${CMAKE_CURRENT_LIST_DIR}/Welcome.txt)
set(CPACK_RESOURCE_FILE_LICENSE     ${CMAKE_CURRENT_LIST_DIR}/License.txt)
set(CPACK_RESOURCE_FILE_README      ${CMAKE_CURRENT_LIST_DIR}/Readme.txt)

if(WIN32)

  set(CPACK_GENERATOR ZIP WIX)

elseif(APPLE)

  set(CPACK_GENERATOR TGZ productbuild)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")

  # always generate tar
  set(CPACK_GENERATOR TGZ)

  # distro specific packages
  include(linux-distro-helper)
  get_linux_distro_name()

  if(GEARSHIFFT_LINUX_DISTRO)
    message(STATUS "Packaging for Linux: ${GEARSHIFFT_LINUX_DISTRO}")

    if("${GEARSHIFFT_LINUX_DISTRO}" MATCHES "Manjaro")
      # TODO:
      # set(CPACK_GENERATOR TGZ)
    endif()

  endif()

else() # unknown system
  set(CPACK_GENERATOR TGZ)
endif()

include(CPack)
