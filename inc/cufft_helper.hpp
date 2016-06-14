#ifndef CUFFT_HELPER_HPP_
#define CUFFT_HELPER_HPP_

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <sstream>

#ifndef CUDA_DISABLE_ERROR_CHECKING
#define CHECK_CUDA(ans) gearshifft::CuFFT::check_cuda((ans), #ans, __FILE__, __LINE__)
#define CHECK_CUFFT(ans) gearshifft::CuFFT::check_cufft((ans), #ans, __FILE__, __LINE__)
#define CHECK_LAST(msg) gearshifft::CuFFT::check_cuda_last(msg, __FILE__, __LINE__)
#else
#define CHECK_CUDA(ans) {}
#define CHECK_CUFFT(ans) {}
#define CHECK_LAST(msg) {}
#endif

namespace gearshifft {
namespace CuFFT {

  inline
  void check_cuda(cudaError_t code, const char *func, const char *file, int line)
  {
    if (code != cudaSuccess)
    {
      fprintf(stderr,"CUDA Error '%s' at %s:%d (%s)\n", cudaGetErrorString(code), file, line, func);
      cudaDeviceReset();
      exit(static_cast<unsigned int>(code));
    }
  }
  inline
  void check_cuda_last(const char *msg, const char *file, int line)
  {
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess)
    {
      fprintf(stderr,"CUDA Error '%s' at %s:%d (%s)\n", cudaGetErrorString(code), file, line, msg);

      cudaDeviceReset();
      exit(static_cast<unsigned int>(code));
    }
  }

  static const char* cufftResultToString(cufftResult error)
  {
    switch (error)
    {
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
  void check_cufft(cufftResult code, const char *func, const char *file, int line)
  {
    if (code)
    {
      fprintf(stderr,"CUDA Lib Error '%s' ('%d') at %s:%d (%s)\n", cufftResultToString(code), static_cast<unsigned int>(code), file, line, func);

      cudaDeviceReset();
      exit(static_cast<unsigned int>(code));
    }
  }
  inline
  std::stringstream getCUDADeviceInformations(int dev) {
    std::stringstream info;
    cudaDeviceProp prop;
    int runtimeVersion = 0;
    CHECK_CUDA( cudaRuntimeGetVersion(&runtimeVersion) );
    CHECK_CUDA( cudaGetDeviceProperties(&prop, dev) );
    info << '"' << prop.name << '"'
         << ',' << prop.major << '.' << prop.minor
         << ", \"MemClock [MHz]\", " << prop.memoryClockRate/1000
         << ",\"CUDA\", " << runtimeVersion
      ;
    return info;
  }
} // CuFFT
} // gearshifft
#endif
