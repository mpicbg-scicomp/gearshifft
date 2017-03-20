# - Find clFFT, AMD's OpenCL FFT library

# This script defines the following variables:
# CLFFT_INCLUDE_DIRS    - Location of clFFT's include directory.
# CLFFT_LIBRARIES       - Location of clFFT's libraries
# CLFFT_FOUND           - True if clFFT has been located
#
# If your clFFT installation is not in a standard installation directory, you
# may provide a hint to where it may be found. Simply set the value CLFFT_ROOT
# to the directory containing 'include/clFFT.h" prior to calling this script.
#
# By default this script will attempt to find the 32-bit version of clFFT.
# If you desire to use the 64-bit version instead, set
#   set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS ON)
# prior to calling this script.
#
#=============================================================================
# Copyright 2014 Brian Kloppenborg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
# 
if( (NOT DEFINED CLFFT_ROOT) AND DEFINED ENV{CLFFT_ROOT} )
  set( CLFFT_ROOT $ENV{CLFFT_ROOT} )
endif()

find_path(CLFFT_INCLUDE_DIRS
    NAMES "clFFT.h"
    PATHS ${CLFFT_ROOT}
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
    )

find_library(CLFFT_LIBRARIES
    NAMES "clFFT"
    PATHS ${CLFFT_ROOT}
    PATH_SUFFIXES "lib" "lib64" "lib64/import"
    NO_DEFAULT_PATH
    )

if(CLFFT_LIBRARIES MATCHES ".*.a")
  set(CLFFT_LIBRARIES ${CLFFT_LIBRARIES};dl)
endif()
# handle the QUIETLY and REQUIRED arguments and set CLFFT_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CLFFT DEFAULT_MSG CLFFT_LIBRARIES CLFFT_INCLUDE_DIRS)
MARK_AS_ADVANCED(CLFFT_LIBRARIES CLFFT_INCLUDE_DIRS)


