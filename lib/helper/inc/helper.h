#ifndef HELPER_H_
#define HELPER_H_

#include "statistics.h"
#include "timer.h"
#include "timestatistics.h"

typedef Timer<TimerCPU_> TimerCPU;

#if defined(CUDA_ENABLED) && defined(OPENCL_ENABLED)
#error "Either CUDA_ENABLED or OPENCL_ENABLED can be set, but not both."
#endif

#ifdef CUDA_ENABLED
#include "timer_cuda.h"
typedef Timer<TimerCUDA_> TimerGPU;
#endif

#ifdef OPENCL_ENABLED
#include "timer_opencl.h"
typedef Timer<TimerOpenCL_> TimerGPU;
#endif


#endif /* HELPER_H_ */
