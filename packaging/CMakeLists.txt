cmake_minimum_required(VERSION 3.7)

set(CPACK_PACKAGE_NAME gearshifft)
set(CPACK_PACKAGE_VENDOR mpicbg-scicomp)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "FFT library benchmark suite for cuFFT, clFFT and FFTW")

set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
set(CPACK_PACKAGE_VERSION_MAJOR     ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR     ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH     ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION           ${PROJECT_VERSION})
# escapes content that is written to cpack files
set(CPACK_VERBATIM_VARIABLES        YES)
set(CPACK_PACKAGE_DESCRIPTION_FILE  ${CMAKE_CURRENT_LIST_DIR}/Description.txt)
set(CPACK_RESOURCE_FILE_WELCOME     ${CMAKE_CURRENT_LIST_DIR}/Welcome.txt)
set(CPACK_RESOURCE_FILE_LICENSE     ${CMAKE_CURRENT_LIST_DIR}/License.txt)
set(CPACK_RESOURCE_FILE_README      ${CMAKE_CURRENT_LIST_DIR}/Readme.txt)
set(CPACK_PACKAGE_FILE_NAME         "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_PACKAGING_INSTALL_PREFIX  ${CMAKE_INSTALL_PREFIX})

if(WIN32)

  set(CPACK_GENERATOR ZIP WIX)

elseif(APPLE)

  set(CPACK_GENERATOR TGZ productbuild)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")

  # always generate tar
  set(CPACK_GENERATOR TGZ)

  # distro specific packages
  include(${GEARSHIFFT_ROOT}/cmake/linux-distro-helper.cmake)
  get_linux_distro_name()

  if(GEARSHIFFT_LINUX_DISTRO)
    message(STATUS "Packaging for Linux: ${GEARSHIFFT_LINUX_DISTRO}")

    if("${GEARSHIFFT_LINUX_DISTRO}" MATCHES "Manjaro")
      # TODO:
      # set(CPACK_GENERATOR TGZ)
    elseif("${GEARSHIFFT_LINUX_DISTRO}" MATCHES "Fedora")
      set(CPACK_GENERATOR RPM)
    endif()

  endif()

else() # unknown system
  set(CPACK_GENERATOR TGZ)
endif()

include(CPack)
