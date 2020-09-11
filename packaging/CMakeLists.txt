
set(CPACK_PACKAGE_NAME gearshifft)
set(CPACK_PACKAGE_VENDOR mpicbg-scicomp)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "FFT library benchmark suite for cuFFT, clFFT, FFTW and rocFFT(hcc)")

set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
set(CPACK_PACKAGE_VERSION_MAJOR     ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR     ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH     ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION           ${PROJECT_VERSION})
set(CPACK_PACKAGE_RELEASE 1)
# escapes content that is written to cpack files
set(CPACK_VERBATIM_VARIABLES        YES)
set(CPACK_PACKAGE_DESCRIPTION_FILE  ${CMAKE_CURRENT_LIST_DIR}/Description.txt)
set(CPACK_RESOURCE_FILE_WELCOME     ${CMAKE_CURRENT_LIST_DIR}/Welcome.txt)
set(CPACK_RESOURCE_FILE_LICENSE     ${CMAKE_CURRENT_LIST_DIR}/../LICENSE)
set(CPACK_RESOURCE_FILE_README      ${CMAKE_CURRENT_LIST_DIR}/Readme.txt)
set(CPACK_PACKAGING_INSTALL_PREFIX  ${CMAKE_INSTALL_PREFIX})

# get components (gearshifft targets)
get_directory_property(GEARSHIFFT_COMPONENTS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src DEFINITION "GEARSHIFFT_COMPONENTS")

set(CPACK_COMPONENTS_ALL ${GEARSHIFFT_COMPONENTS} config)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)

# looking for specific backend builds from superbuild

if(SEPARATE_BACKENDS MATCHES "gearshifft-rocfft")
  set(CPACK_INSTALL_CMAKE_PROJECTS
    "${GEARSHIFFT_SUPERBUILD_BINARY_DIR}/gearshifft-rocfft-build;gearshifft;gearshifft_rocfft;/")
endif()

set(CPACK_INSTALL_CMAKE_PROJECTS
  "${CPACK_INSTALL_CMAKE_PROJECTS}"
  "${CMAKE_BINARY_DIR};gearshifft;ALL;/")

# default system name by cmake
set(SYSTEM_NAME "${CMAKE_SYSTEM_NAME}")

if(WIN32)

  set(CPACK_GENERATOR ZIP WIX)

elseif(APPLE)

  set(CPACK_GENERATOR TGZ productbuild)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")

  include(gearshifft_linux_distro)
  set(SYSTEM_NAME "${CMAKE_SYSTEM_NAME}-${GEARSHIFFT_LINUX_DISTRO}") # overwrite to include distro

  # always generate tar
  set(CPACK_GENERATOR TGZ)

  find_program(HELPER_RPMBUILD_EXEC rpmbuild)
  if(HELPER_RPMBUILD_EXEC)
    list(APPEND CPACK_GENERATOR RPM)
  endif()

  find_program(HELPER_DPKG_EXEC dpkg)
  if(HELPER_DPKG_EXEC)
    list(APPEND CPACK_GENERATOR DEB)

    set(CPACK_DEBIAN_PACKAGE_PROVIDES ${PROJECT_NAME})
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Matthias Werner <11235813tdd@posteo.net>") #required
    set(CPACK_DEBIAN_PACKAGE_SECTION "science")
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
  endif()

else() # unknown system
  set(CPACK_GENERATOR TGZ)
endif()

# proper package file name gearshifft-<version>-<pkg-release>-<system>-<arch>
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}-${SYSTEM_NAME}.${CMAKE_SYSTEM_PROCESSOR}")


include(CPack)

foreach(comp ${GEARSHIFFT_COMPONENTS})
  cpack_add_component(${comp} DESCRIPTION "gearshifft benchmark binary")
  cpack_add_component(doc DESCRIPTION "gearshifft - doc files")
  cpack_add_component(config DESCRIPTION "gearshifft - FFT configuration files")
endforeach()
