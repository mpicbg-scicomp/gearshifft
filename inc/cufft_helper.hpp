#ifndef CUFFT_HELPER_HPP_
#define CUFFT_HELPER_HPP_

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <sstream>
#include <stdexcept>

#ifndef CUDA_DISABLE_ERROR_CHECKING
#define CHECK_CUDA(ans) gearshifft::CuFFT::check_cuda((ans), "", #ans, __FILE__, __LINE__)
#define CHECK_CUFFT(ans) gearshifft::CuFFT::check_cufft((ans), #ans, __FILE__, __LINE__)
#define CHECK_LAST(msg) gearshifft::CuFFT::check_cuda(cudaGetLastError(), msg, __FILE__, __LINE__)
#else
#define CHECK_CUDA(ans) {}
#define CHECK_CUFFT(ans) {}
#define CHECK_LAST(msg) {}
#endif

namespace gearshifft {
namespace CuFFT {

  inline
  void throw_error(int code,
                   const char* error_string,
                   const char* msg,
                   const char* func,
                   const char* file,
                   int line) {
    cudaDeviceReset();
    throw std::runtime_error("CUDA error "
                             +std::string(msg)
                             +" "+std::string(error_string)
                             +" ["+std::to_string(code)+"]"
                             +" "+std::string(file)
                             +":"+std::to_string(line)
                             +" "+std::string(func)
      );
  }

  inline
  void check_cuda(cudaError_t code, const char* msg, const char *func, const char *file, int line) {
    if (code != cudaSuccess) {
      throw_error(static_cast<int>(code),
                  cudaGetErrorString(code), msg, func, file, line);
    }
  }

  static const char* cufftResultToString(cufftResult error) {
    switch (error) {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
    }
    return "<unknown>";
  }

  inline
  void check_cufft(cufftResult code, const char *func, const char *file, int line) {
    if(code) {
      throw_error(static_cast<int>(code),
                  cufftResultToString(code), "cuFFT", func, file, line);
    }
  }

  inline
  std::stringstream getCUDADeviceInformations(int dev) {
    std::stringstream info;
    cudaDeviceProp prop;
    int runtimeVersion = 0;
    size_t f=0, t=0;
    CHECK_CUDA( cudaRuntimeGetVersion(&runtimeVersion) );
    CHECK_CUDA( cudaGetDeviceProperties(&prop, dev) );
    CHECK_CUDA( cudaMemGetInfo(&f, &t) );
    info << '"' << prop.name << '"'
         << ", \"CC\", " << prop.major << '.' << prop.minor
         << ", \"Multiprocessors\", "<< prop.multiProcessorCount
         << ", \"Memory [MiB]\", "<< t/1048576
         << ", \"MemoryFree [MiB]\", " << f/1048576
         << ", \"MemClock [MHz]\", " << prop.memoryClockRate/1000
         << ", \"GPUClock [MHz]\", " << prop.clockRate/1000
         << ", \"CUDA Runtime\", " << runtimeVersion
      ;
    return info;
  }
} // CuFFT
} // gearshifft
#endif
