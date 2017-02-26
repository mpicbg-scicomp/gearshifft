#ifndef TIMER_CUDA_HPP_
#define TIMER_CUDA_HPP_

#include "timer.hpp"
#include "libraries/cufft/cufft_helper.hpp"
#include <cuda_runtime.h>

namespace gearshifft {

  struct TimerCUDA_ {
    double time = 0.0;
    cudaEvent_t gpustart = 0;
    cudaEvent_t gpustop = 0;

    TimerCUDA_() {
      CHECK_CUDA(cudaEventCreate(&gpustart));
      CHECK_CUDA(cudaEventCreate(&gpustop));
    }

    void startTimer() {
      // complete all issued CUDA calls
      CHECK_CUDA( cudaEventRecord(gpustart) );
    }

    double stopTimer() {
      float milliseconds = 0;
      CHECK_CUDA(cudaEventRecord(gpustop));
      CHECK_CUDA(cudaEventSynchronize(gpustop));
      CHECK_CUDA(cudaEventElapsedTime(&milliseconds, gpustart, gpustop));
      return (time = static_cast<double>(milliseconds));
    }

    ~TimerCUDA_() {
      if(gpustart){
        CHECK_CUDA(cudaEventDestroy(gpustart));
      }
      if(gpustop){
        CHECK_CUDA(cudaEventDestroy(gpustop));
      }
    }
  };

  typedef Timer<TimerCUDA_> TimerGPU;

} // gearshifft
#endif /* TIMER_CUDA_HPP_ */
