#ifndef TIMER_OPENCL_HPP_
#define TIMER_OPENCL_HPP_

#include "application.hpp"
#include "libraries/clfft/clfft_helper.hpp"
#include <CL/cl.h>

namespace gearshifft {

  /**
   * OpenCL Timer
   */
  template<typename Context>
  struct TimerOpenCL_ {
    double time = 0.0;
    void startTimer(){
    }
    double stopTimer(){
      cl_ulong time_start, time_end;
      std::shared_ptr<cl_event> event = Application<Context>::getContext().event;
      if(event) {
        CHECK_CL(clWaitForEvents(1,event.get()));
        CHECK_CL(clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL));
        CHECK_CL(clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL));

        time = 1e-6*(time_end-time_start);
      }else
        time = -1.0;
      return time;
    }
  };

  template<typename Context>
  using TimerGPU = Timer<TimerOpenCL_<Context> >;

}
#endif /* TIMER_OPENCL_HPP_ */
