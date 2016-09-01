
#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <chrono>
#include <stdexcept>

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

} // gearshifft

#endif /* TIMER_HPP_ */
