#ifndef TIMER_OPENCL_H_
#define TIMER_OPENCL_H_

#include <CL/cl.h>

struct TimerOpenCL_ {
  double time = 0.0;
  void startTimer(){

  }
  double stopTimer(){
    return time;
  }

};

#endif /* TIMER_OPENCL_H_ */
