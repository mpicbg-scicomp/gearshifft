#ifndef CUFFT_HELPER_HPP_
#define CUFFT_HELPER_HPP_

#include "helper.h"

#include <cufft.h>

#define CHECK_CUFFT(ans) gearshifft::CuFFT::check_error((ans), #ans, __FILE__, __LINE__)

namespace gearshifft {
namespace CuFFT {

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
  void check_error(cufftResult code, const char *func, const char *file, int line)
  {
    if (code)
    {
      fprintf(stderr,"CUDA Lib Error '%s' ('%d') at %s:%d (%s)\n", cufftResultToString(code), static_cast<unsigned int>(code), file, line, func);

      cudaDeviceReset();
      exit(static_cast<unsigned int>(code));
    }
  }
} // CuFFT
} // gearshifft
#endif
