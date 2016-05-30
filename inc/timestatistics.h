#ifndef TIMESTATISTICS_H_
#define TIMESTATISTICS_H_

#include "statistics.h"
#include <string>
#include <stdexcept>

namespace gearshifft {
namespace helper {

  template<typename TTimer>
  class TimeStatistics final
  {
  public:
    explicit TimeStatistics(Statistics* stats=NULL);
    ~TimeStatistics();

    int add(const std::string& label, const std::string& unit="ms", bool invert=false, double factor=1.0);
    int append(const std::string& label, const std::string& unit="ms", bool invert=false, double factor=1.0);
    void setFactor(int index, double factor);
    void setFactorAll(double factor);
    void start(int index);
    void stop(int index);
    const Statistics* statistics() const;

  private:
    Statistics* _statistics;
    std::vector<TTimer> _mtimes;
    bool _delete_statistics;
  };


  template<typename TTimer>
  inline void
  TimeStatistics<TTimer>::setFactor(int index, double factor)
  {
    _statistics->setFactor(index, factor);
  }

  template<typename TTimer>
  inline void
  TimeStatistics<TTimer>::setFactorAll(double factor)
  {
    _statistics->setFactorAll(factor);
  }

  template<typename TTimer>
  inline const Statistics*
  TimeStatistics<TTimer>::statistics() const {
    return _statistics;
  }

  template<typename TTimer>
  TimeStatistics<TTimer>::TimeStatistics(Statistics* stats) {
    if(stats==nullptr){
      _statistics = new Statistics();
      _delete_statistics = true;
    }else{
      _delete_statistics = false;
      _statistics = stats;
    }
  }

  template<typename TTimer>
  TimeStatistics<TTimer>::~TimeStatistics() {
    if(_delete_statistics)
      delete _statistics;
  }

  template<typename TTimer>
  void
  TimeStatistics<TTimer>::start(int index) {
    _mtimes[index].startTimer();
  }

  template<typename TTimer>
  void
  TimeStatistics<TTimer>::stop(int index) {
    double eltime = _mtimes[index].stopTimer();
    _statistics->process( index, eltime );
  }

/**
 * Add a time metric for which statistics are generated after timer has stopped.
 * @param label Metric label/short description. If already exists metric data will be resetted.
 * @param unit Metric unit.
 * @param invert Inverts time metric, so we have 1/time-scale.
 * @param factor Scales the time by this number.
 * @return Unique index of metric.
 */
  template<typename TTimer>
  int
  TimeStatistics<TTimer>::add(const std::string& label, const std::string& unit, bool invert, double factor) {
    int index = _statistics->add(label, unit, invert, factor);
    _mtimes.resize(_statistics->getLength());
    return index;
  }

/**
 * Add a time metric for which statistics are generated after timer has stopped.
 * @param label Metric label/short description. If already exists metric data will be appended.
 * @param unit Metric unit.
 * @param invert Inverts time metric, so we have 1/time-scale.
 * @param factor Scales the time by this number.
 * @return Unique index of metric.
 */
  template<typename TTimer>
  int
  TimeStatistics<TTimer>::append(const std::string& label, const std::string& unit, bool invert, double factor) {
    int index = _statistics->append(label, unit, invert, factor);
    _mtimes.resize(_statistics->getLength());
    return index;
  }

} // helper
} // gearshifft

#endif /* TIMESTATISTICS_H_ */
