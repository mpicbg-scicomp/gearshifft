#ifndef TIMER_HIP_HPP_
#define TIMER_HIP_HPP_

#include "timer.hpp"
#include "libraries/rocfft/rocfft_helper.hpp"
#include <hip/hip_runtime.h>

namespace gearshifft {

  struct TimerHIP_ {
    double time = 0.0;
    hipEvent_t gpustart = 0;
    hipEvent_t gpustop = 0;

    TimerHIP_() {
      CHECK_HIP(hipEventCreate(&gpustart));
      CHECK_HIP(hipEventCreate(&gpustop));
    }

    void startTimer() {
      // complete all issued HIP calls
      CHECK_HIP( hipEventRecord(gpustart) );
    }

    double stopTimer() {
      float milliseconds = 0;
      CHECK_HIP(hipEventRecord(gpustop));
      CHECK_HIP(hipEventSynchronize(gpustop));
      CHECK_HIP(hipEventElapsedTime(&milliseconds, gpustart, gpustop));
      return (time = static_cast<double>(milliseconds));
    }

    ~TimerHIP_() {
      if(gpustart){
        CHECK_HIP(hipEventDestroy(gpustart));
      }
      if(gpustop){
        CHECK_HIP(hipEventDestroy(gpustop));
      }
    }
  };

  typedef Timer<TimerHIP_> TimerGPU;

} // gearshifft
#endif /* TIMER_HIP_HPP_ */
