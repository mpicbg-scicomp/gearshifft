#ifndef TIMER_CUDA_H_
#define TIMER_CUDA_H_

#include "cufft_helper.hpp"
#include <cuda_runtime.h>

namespace gearshifft {

  struct TimerCUDA_ {
    double time = 0.0;
    cudaEvent_t gpustart = 0;
    cudaEvent_t gpustop = 0;
    void startTimer() {
      if(gpustart==0){
        CHECK_CUDA(cudaEventCreate(&gpustart));
      }
      if(gpustop==0){
        CHECK_CUDA(cudaEventCreate(&gpustop));
      }
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

} // gearshifft
#endif /* TIMER_CUDA_H_ */
