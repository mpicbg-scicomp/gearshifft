#ifndef ROCFFT_HELPER_HPP_
#define ROCFFT_HELPER_HPP_

#include "core/get_memory_size.h"

#include "hip/hip_runtime_api.h"
#include "rocfft.h"

#include <stdio.h>
#include <stdlib.h>
#include <rocfft.h>
#include <sstream>
#include <stdexcept>

#ifndef HIP_DISABLE_ERROR_CHECKING
#define CHECK_HIP(ans) gearshifft::Rocfft::check_hip((ans), "", #ans, __FILE__, __LINE__)
#define CHECK_LAST(msg) gearshifft::Rocfft::check_hip(hipGetLastError(), msg, "CHECK_LAST", __FILE__, __LINE__)
#else
#define CHECK_HIP(ans) {}
#define CHECK_LAST(msg) {}
#endif

namespace gearshifft {
namespace Rocfft {

    inline void rocfftGetVersion(int *version){
        version = 10000*rocfft_version_major + 100*rocfft_version_minor + rocfft_version_patch;
        return;
    }

  inline
  void throw_error(int code,
                   const char* error_string,
                   const char* msg,
                   const char* func,
                   const char* file,
                   int line) {
    throw std::runtime_error("HIP error "
                             +std::string(msg)
                             +" "+std::string(error_string)
                             +" ["+std::to_string(code)+"]"
                             +" "+std::string(file)
                             +":"+std::to_string(line)
                             +" "+std::string(func)
      );
  }

  static const char* rocfftResultToString(rocfft_status_e error) {
      switch (error) {
      case rocfft_status_success:
          return "ROCFFT_SUCCESS";

      case rocfft_status_failure:
          return "ROCFFT_FAILURE";

      case rocfft_status_invalid_arg_value:
          return "ROCFFT_INVALID_ARG_VALUE";

      case rocfft_status_invalid_dimensions:
          return "ROCFFT_INVALID_DIMENSIONS";

      case rocfft_status_invalid_array_type:
          return "ROCFFT_INVALID_ARRAY_TYPE";

      case rocfft_status_invalid_strides:
          return "ROCFFT_INVALID_STRIDES";

      case rocfft_status_invalid_distance:
          return "ROCFFT_INVALID_DISTANCE";

      case rocfft_status_invalid_offset:
          return "ROCFFT_INVALID_OFFSET";

      default:
          return "<unknown>";
      }

  }

  inline
  void check_hip(hipError_t code, const char* msg, const char *func, const char *file, int line) {
    if (code != hipSuccess) {
      throw_error(static_cast<int>(code),
                  hipGetErrorString(code), msg, func, file, line);
    }
  }
  inline
  void check_hip(rocfft_status_e code, const char* msg,  const char *func, const char *file, int line) {
    if (code != ROCFFT_SUCCESS) {
      throw_error(static_cast<int>(code),
                  rocfftResultToString(code), "rocfft", func, file, line);
    }
  }

  inline
  std::stringstream getHIPDeviceInformations(int dev) {
    std::stringstream info;
    hipDeviceProp prop;
    //int runtimeVersion = 0;
    int rocfftv = 0;
    size_t f=0, t=0;
    CHECK_HIP(rocfftGetVersion(&rocfftv));
    //CHECK_HIP( hipRuntimeGetVersion(&runtimeVersion) );
    CHECK_HIP(hipGetDeviceProperties(&prop, dev));
    CHECK_HIP(hipMemGetInfo(&f, &t));
    info << '"' << prop.name << '"'
         << ", \"CC\", " << prop.major << '.' << prop.minor
         << ", \"PCI Bus ID\", " << prop.pciBusID
         << ", \"PCI Device ID\", " << prop.pciDeviceID
         << ", \"Multiprocessors\", "<< prop.multiProcessorCount
         << ", \"Memory [MiB]\", "<< t/1048576
         << ", \"MemoryFree [MiB]\", " << f/1048576
         << ", \"HostMemory [MiB]\", "<< getMemorySize()/1048576
         << ", \"ECC enabled\", " << prop.ECCEnabled
         << ", \"MemClock [MHz]\", " << prop.memoryClockRate/1000
         << ", \"GPUClock [MHz]\", " << prop.clockRate/1000
        //<< ", \"HIP Runtime\", " << runtimeVersion
         << ", \"rocfft\", " << rocfftv
      ;
    return info;
  }

  std::stringstream listHipDevices() {
    std::stringstream info;
    int nrdev = 0;
    CHECK_HIP( hipGetDeviceCount( &nrdev ) );
    if(nrdev==0)
      throw std::runtime_error("No HIP capable device found");
    for(int i=0; i<nrdev; ++i)
      info << "\"ID\"," << i << "," << getHIPDeviceInformations(i).str() << std::endl;
    return info;
  }
} // Rocfft
} // gearshifft
#endif
