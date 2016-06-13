#ifndef TIMER_OPENCL_H_
#define TIMER_OPENCL_H_

#include "clfft_helper.hpp"
#include <CL/cl.h>

namespace gearshifft {
  /**
   * @todo OpenCL Timer
   */
  struct TimerOpenCL_ {
    double time = 0.0;
    void startTimer(){

    }
    double stopTimer(){
      return time;
    }

  };
}
#endif /* TIMER_OPENCL_H_ */
