#ifndef GLOBALS_CUDA_H_
#define GLOBALS_CUDA_H_

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/*---------------------------------------------------------------------------*/
#define CUDA_ERROR_CHECKING 1
#define LAST_CUDA_ERROR_CHECKING 1
/*---------------------------------------------------------------------------*/

#if CUDA_ERROR_CHECKING==1
  #define CHECK_ERROR(ans) globals_check_error((ans), #ans, __FILE__, __LINE__)
  #define CHECK_ERROR_LIB(ans) globals_check_error_lib((ans), #ans, __FILE__, __LINE__)
#else
  #define CHECK_ERROR(ans) {}
  #define CHECK_ERROR_LIB(ans) {}
#endif
#if CUDA_ERROR_CHECKING==1 && LAST_CUDA_ERROR_CHECKING==1
#define CHECK_LAST(msg) globals_check_error_last(msg, __FILE__, __LINE__)
#else
#define CHECK_LAST(msg) {}
#endif

inline
void globals_check_error(cudaError_t code, const char *func, const char *file, int line)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA Error '%s' at %s:%d (%s)\n", cudaGetErrorString(code), file, line, func);
      cudaDeviceReset();
      exit(static_cast<unsigned int>(code));
   }
}
inline
void globals_check_error_last(const char *msg, const char *file, int line)
{
  cudaError_t code = cudaGetLastError();
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA Error '%s' at %s:%d (%s)\n", cudaGetErrorString(code), file, line, msg);

      cudaDeviceReset();
      exit(static_cast<unsigned int>(code));
   }
}
template<typename T>
void globals_check_error_lib(T code, const char *func, const char *file, int line)
{
   if (code)
   {
      fprintf(stderr,"CUDA Lib Error '%d' at %s:%d (%s)\n", static_cast<unsigned int>(code), file, line, func);

      cudaDeviceReset();
      exit(static_cast<unsigned int>(code));
   }
}
/*---------------------------------------------------------------------------*/


#endif /* GLOBALS_CUDA_H_ */
