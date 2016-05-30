#ifndef HELPER_H_
#define HELPER_H_

#include "statistics.h"
#include "timer.h"
#include "timestatistics.h"

#if defined(CUDA_ENABLED) && defined(OPENCL_ENABLED)
#error "Either CUDA_ENABLED or OPENCL_ENABLED can be set, but not both."
#endif

#ifdef CUDA_ENABLED
#include "timer_cuda.h"
#endif
#ifdef OPENCL_ENABLED
#include "timer_opencl.h"
#endif

namespace gearshifft {
namespace helper {

typedef Timer<TimerCPU_> TimerCPU;
#ifdef CUDA_ENABLED
typedef Timer<TimerCUDA_> TimerGPU;
#endif
#ifdef OPENCL_ENABLED
typedef Timer<TimerOpenCL_> TimerGPU;
#endif
} // helper
} // gearshifft

#endif /* HELPER_H_ */
