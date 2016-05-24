#ifndef TIMER_CUDA_H_
#define TIMER_CUDA_H_

#include "globals_cuda.h"
#include <cuda_runtime.h>

struct TimerCUDA_ {
  double time = 0.0;
  cudaEvent_t gpustart = 0;
  cudaEvent_t gpustop = 0;
  void startTimer(){
    if(gpustart==0){
      CHECK_ERROR(cudaEventCreate(&gpustart));
    }
    if(gpustop==0){
      CHECK_ERROR(cudaEventCreate(&gpustop));
    }
    CHECK_ERROR( cudaEventRecord(gpustart) );
  }
  double stopTimer(){
    float milliseconds = 0;
    CHECK_ERROR(cudaEventRecord(gpustop));
    CHECK_ERROR(cudaEventSynchronize(gpustop));
    CHECK_ERROR(cudaEventElapsedTime(&milliseconds, gpustart, gpustop));
    return (time = static_cast<double>(milliseconds));
  }
  ~TimerCUDA_(){
    if(gpustart){
      CHECK_ERROR(cudaEventDestroy(gpustart));
    }
    if(gpustop){
      CHECK_ERROR(cudaEventDestroy(gpustop));
    }
  }
};

#endif /* TIMER_CUDA_H_ */
