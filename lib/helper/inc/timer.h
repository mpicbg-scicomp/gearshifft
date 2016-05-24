
#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>
#include <stdexcept>

template<typename TimerImpl>
struct Timer : public TimerImpl {
  bool started = false;
  void startTimer(){
    TimerImpl::startTimer();
    started = true;
  }
  double stopTimer(){
    if(started==false)
      throw std::runtime_error("Timer must be started before.");
    started = false;
    return TimerImpl::stopTimer();
  }
};

// Wall time
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

#endif /* TIMER_H_ */
