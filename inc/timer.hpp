
#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <chrono>
#include <stdexcept>

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

  /**
   * Timer Base Class
   */
  template<typename TimerImpl>
    struct Timer : public TimerImpl {
    bool started = false;
    void startTimer() {
      TimerImpl::startTimer();
      started = true;
    }
    double stopTimer() {
      if(started==false)
        throw std::runtime_error("Timer must be started before.");
      started = false;
      return TimerImpl::stopTimer();
    }
  };

  /** CPU Wall timer
   */
  struct TimerCPU_ {
    typedef std::chrono::high_resolution_clock clock;

    clock::time_point start;
    double time = 0.0;

    void startTimer() {
      start = clock::now();
    }

    double stopTimer() {
      auto diff = clock::now() - start;
      return (time = std::chrono::duration<double, std::milli> (diff).count());
    }
  };

typedef Timer<TimerCPU_> TimerCPU;
#ifdef CUDA_ENABLED
typedef Timer<TimerCUDA_> TimerGPU;
#endif
#ifdef OPENCL_ENABLED
typedef Timer<TimerOpenCL_> TimerGPU;
#endif

} // gearshifft

#endif /* TIMER_HPP_ */
