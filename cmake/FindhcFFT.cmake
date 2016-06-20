# - Find hcFFT, HSA's hcFFT library

# This script defines the following variables:
# HCFFT_INCLUDE_DIRS    - Location of hcFFT's include directory.
# HCFFT_LIBRARIES       - Location of hcFFT's libraries
# HCFFT_FOUND           - True if hcFFT has been located
#
# If your hcFFT installation is not in a standard installation directory, you
# may provide a hint to where it may be found. Simply set the value HCFFT_ROOT
# to the directory containing 'include/hcfft.h" prior to calling this script.
# Defining an environment variable called HCFFT_ROOT would also work
#
# By default this script will attempt to find the 32-bit version of hcFFT.
# If you desire to use the 64-bit version instead, set
#   set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS ON)
# prior to calling this script.
#
#=============================================================================
# Copyright 2016 Peter Steinbach
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
if( (NOT DEFINED HCFFT_ROOT) AND DEFINED ENV{HCFFT_ROOT} )
  set( HCFFT_ROOT $ENV{HCFFT_ROOT} )
endif()

find_path(HCFFT_INCLUDE_DIRS
    NAMES "hcfft.h"
    PATHS ${HCFFT_ROOT}
    PATH_SUFFIXES "include"
    )    
find_library(HCFFT_LIBRARIES
    NAMES "hcfft"
    PATHS ${HCFFT_ROOT}
    PATH_SUFFIXES "lib" "lib64"
    )
# handle the QUIETLY and REQUIRED arguments and set HCFFT_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(HCFFT DEFAULT_MSG HCFFT_LIBRARIES HCFFT_INCLUDE_DIRS)
MARK_AS_ADVANCED(HCFFT_LIBRARIES HCFFT_INCLUDE_DIRS)


