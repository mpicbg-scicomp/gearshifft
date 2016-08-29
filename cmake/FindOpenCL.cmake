# This comes from:
# https://github.com/elhigu/cmake-findopencl
# @todo may integrate opencl finder from cmake

# Find OpenCL
#
# To set manually the paths, define these environment variables:
# OpenCL_INCPATH    - Include path (e.g. OpenCL_INCPATH=/opt/cuda/4.0/cuda/include)
# OpenCL_LIBPATH    - Library path (e.g. OpenCL_LIBPATH=/usr/lib64/nvidia)
#
# Once done this will define
#  OPENCL_FOUND            - system has OpenCL
#  OPENCL_INCLUDE_DIRS     - the OpenCL include directory
#  OPENCL_LIBRARIES        - link these to use OpenCL
#  OPENCL_HAS_CPP_BINDINGS - system has also cl.hpp

# 0.2.0 added variables for our cluster environment
# 0.3.0 further paths to find OpenCL

FIND_PACKAGE(PackageHandleStandardArgs)

IF (APPLE)

	# IF OpenCL_LIBPATH is given use it and don't use default path
	IF (DEFINED ENV{OpenCL_LIBPATH})
		FIND_LIBRARY(OPENCL_LIBRARIES OpenCL PATHS ENV OpenCL_LIBPATH NO_DEFAULT_PATH)
	ELSE ()
		FIND_LIBRARY(OPENCL_LIBRARIES OpenCL DOC "OpenCL lib for OSX")
	ENDIF ()

	# IF OpenCL_INCPATH is given use it and find for CL/cl.h and OpenCL/cl.h do not try to find default paths
	IF (DEFINED ENV{OpenCL_INCPATH})
		FIND_PATH(OPENCL_INCLUDE_DIRS CL/cl.h OpenCL/cl.h PATHS ENV OpenCL_INCPATH NO_DEFAULT_PATH)
		FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp OpenCL/cl.hpp PATHS ${OPENCL_INCLUDE_DIRS} NO_DEFAULT_PATH)
	ELSE ()
		FIND_PATH(OPENCL_INCLUDE_DIRS OpenCL/cl.h DOC "Include for OpenCL on OSX")
		FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS OpenCL/cl.hpp DOC "Include for OpenCL CPP bindings on OSX")
	ENDIF ()

ELSE (APPLE)

	IF (WIN32)

		# Find OpenCL includes and libraries from environment variables provided by vendor
		SET(OPENCL_INCLUDE_SEARCH_PATHS)
		SET(OPENCL_LIBRARY_SEARCH_PATHS)
		SET(OPENCL_LIBRARY_64_SEARCH_PATHS)

		# Nvidia
		IF (DEFINED ENV{CUDA_INC_PATH})
			SET(OPENCL_INCLUDE_SEARCH_PATHS ${OPENCL_INCLUDE_SEARCH_PATHS} $ENV{CUDA_INC_PATH})
			SET(OPENCL_LIBRARY_64_SEARCH_PATHS ${OPENCL_LIBRARY_64_SEARCH_PATHS} $ENV{CUDA_LIB_PATH}/../lib64)
			SET(OPENCL_LIBRARY_SEARCH_PATHS ${OPENCL_LIBRARY_SEARCH_PATHS} $ENV{CUDA_LIB_PATH}/../lib)
		ENDIF()

		# Intel SDK
		IF (DEFINED ENV{INTELOCSDKROOT})
			SET(OPENCL_INCLUDE_SEARCH_PATHS ${OPENCL_INCLUDE_SEARCH_PATHS} $ENV{INTELOCSDKROOT}/include)
			SET(OPENCL_LIBRARY_64_SEARCH_PATHS ${OPENCL_LIBRARY_64_SEARCH_PATHS} $ENV{INTELOCSDKROOT}/lib/x64)
			SET(OPENCL_LIBRARY_SEARCH_PATHS ${OPENCL_LIBRARY_SEARCH_PATHS} $ENV{INTELOCSDKROOT}/lib/x86)
		ENDIF()

		# AMD SDK
		IF (DEFINED ENV{AMDAPPSDKROOT})
			SET(OPENCL_INCLUDE_SEARCH_PATHS ${OPENCL_INCLUDE_SEARCH_PATHS} $ENV{AMDAPPSDKROOT}/include)
			SET(OPENCL_LIBRARY_64_SEARCH_PATHS ${OPENCL_LIBRARY_64_SEARCH_PATHS} $ENV{AMDAPPSDKROOT}/lib/x86_64)
			SET(OPENCL_LIBRARY_SEARCH_PATHS ${OPENCL_LIBRARY_SEARCH_PATHS} $ENV{AMDAPPSDKROOT}/lib/x86)
		ENDIF()

		# Override search paths with OpenCL_INCPATH env variable
		IF (DEFINED ENV{OpenCL_INCPATH})
			SET(OPENCL_INCLUDE_SEARCH_PATHS $ENV{OpenCL_INCPATH})
		ENDIF ()

		# Override search paths with OpenCL_LIBPATH env variable
		IF (DEFINED ENV{OpenCL_INCPATH})
			SET(OPENCL_LIBRARY_SEARCH_PATHS $ENV{OpenCL_LIBPATH})
			SET(OPENCL_LIBRARY_64_SEARCH_PATHS $ENV{OpenCL_LIBPATH})
		ENDIF ()

		FIND_PATH(OPENCL_INCLUDE_DIRS CL/cl.h PATHS ${OPENCL_INCLUDE_SEARCH_PATHS})
		FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp PATHS ${OPENCL_INCLUDE_SEARCH_PATHS})
		
		FIND_LIBRARY(_OPENCL_32_LIBRARIES OpenCL.lib HINTS ${OPENCL_LIBRARY_SEARCH_PATHS} PATHS ${OPENCL_LIB_DIR} ENV PATH)
		FIND_LIBRARY(_OPENCL_64_LIBRARIES OpenCL.lib HINTS ${OPENCL_LIBRARY_64_SEARCH_PATHS} PATHS ${OPENCL_LIB_DIR} ENV PATH)

		# Check if 64bit or 32bit versions links fine
  		SET (_OPENCL_VERSION_SOURCE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/openclversion.c")
  		#SET (_OPENCL_VERSION_SOURCE "${CMAKE_BINARY_DIR}/test.c")
		FILE (WRITE "${_OPENCL_VERSION_SOURCE}"
			"
			#if __APPLE__
			#include <OpenCL/cl.h>
			#else /* !__APPLE__ */
			#include <CL/cl.h>
			#endif /* __APPLE__ */
			int main()
			{	
			    cl_int result;
			    cl_platform_id id;
			    result = clGetPlatformIDs(1, &id, NULL);
			    return result != CL_SUCCESS;
			}
			")

  		TRY_COMPILE(_OPENCL_64_COMPILE_SUCCESS ${CMAKE_BINARY_DIR} "${_OPENCL_VERSION_SOURCE}"
			CMAKE_FLAGS
			"-DINCLUDE_DIRECTORIES:STRING=${OPENCL_INCLUDE_DIRS}"
			CMAKE_FLAGS
			"-DLINK_LIBRARIES:STRING=${_OPENCL_64_LIBRARIES}"
  		)

		IF(_OPENCL_64_COMPILE_SUCCESS)
			message(STATUS "OpenCL 64bit lib found.")
			SET(OPENCL_LIBRARIES ${_OPENCL_64_LIBRARIES})
  		ELSE()
	  		TRY_COMPILE(_OPENCL_32_COMPILE_SUCCESS ${CMAKE_BINARY_DIR} "${_OPENCL_VERSION_SOURCE}"
				CMAKE_FLAGS
				"-DINCLUDE_DIRECTORIES:STRING=${OPENCL_INCLUDE_DIRS}"
				CMAKE_FLAGS
				"-DLINK_LIBRARIES:STRING=${_OPENCL_32_LIBRARIES}"
	  		)
			IF(_OPENCL_32_COMPILE_SUCCESS)
				message(STATUS "OpenCL 32bit lib found.")
				SET(OPENCL_LIBRARIES ${_OPENCL_32_LIBRARIES})
			ELSE()
				message(STATUS "Couldn't link opencl..")
			ENDIF()
		ENDIF()


	ELSE (WIN32) # linux
  
  		IF (CYGWIN)
                  SET (CMAKE_FIND_LIBRARY_SUFFIXES .lib)
                  SET (OCL_LIB_SUFFIX .lib)
  		ENDIF (CYGWIN)

                IF( NOT DEFINED OPENCL_ROOT AND DEFINED ENV{OPENCL_ROOT} )
                  SET(OPENCL_ROOT $ENV{OPENCL_ROOT})
                ENDIF()
                IF( NOT DEFINED OPENCL_INC AND DEFINED ENV{OPENCL_INC} )
                  SET(OPENCL_INC $ENV{OPENCL_INC})
                ENDIF()
                IF( NOT DEFINED OPENCL_LIB AND DEFINED ENV{OPENCL_LIB} )
                  SET(OPENCL_LIB $ENV{OPENCL_LIB})
                ENDIF()

		# Unix style platforms
		FIND_LIBRARY(OPENCL_LIBRARIES OpenCL${OCL_LIB_SUFFIX}
                  PATHS ENV LD_LIBRARY_PATH
                  PATHS ENV OpenCL_LIBPATH
                  OPENCL_LIB
		)

		GET_FILENAME_COMPONENT(OPENCL_LIB_DIR ${OPENCL_LIBRARIES} PATH)
		GET_FILENAME_COMPONENT(_OPENCL_INC_CAND ${OPENCL_LIB_DIR}/../../include ABSOLUTE)

		# The AMD SDK currently does not place its headers
		# in /usr/include, therefore also search relative
		# to the library
		FIND_PATH(OPENCL_INCLUDE_DIRS 
                  NAMES "cl.h"
                  PATHS ${_OPENCL_INC_CAND} 
                  PATHS "/usr/local/cuda/include"
                  PATHS "/usr/cuda/include"
                  PATHS "/opt/cuda/include"
                  PATHS "/opt/AMDAPP/include"
                  PATHS ENV OpenCL_INCPATH
                  PATHS ${OPENCL_INC}
                  PATH_SUFFIXES "CL")

	ENDIF (WIN32)

ENDIF (APPLE)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenCL DEFAULT_MSG OPENCL_LIBRARIES OPENCL_INCLUDE_DIRS)

IF(_OPENCL_CPP_INCLUDE_DIRS)
	SET( OPENCL_HAS_CPP_BINDINGS TRUE )
	LIST( APPEND OPENCL_INCLUDE_DIRS ${_OPENCL_CPP_INCLUDE_DIRS} )
	# This is often the same, so clean up
	LIST( REMOVE_DUPLICATES OPENCL_INCLUDE_DIRS )
ENDIF(_OPENCL_CPP_INCLUDE_DIRS)

MARK_AS_ADVANCED(
  OPENCL_INCLUDE_DIRS
)
