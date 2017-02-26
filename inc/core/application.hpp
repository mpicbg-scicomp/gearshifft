#ifndef APPLICATION_HPP_
#define APPLICATION_HPP_

#include "result_benchmark.hpp"
#include "result_all.hpp"
#include "timer_cpu.hpp"
#include "types.hpp"

#include <vector>
#include <array>

namespace gearshifft {

  template<typename T_Context>
  class Application {
  public:
    /// Number of benchmark runs after warmup
    static constexpr int NR_RUNS = 5;
    static constexpr int NR_WARMUP_RUNS = 1;
    static const int NR_RECORDS  = static_cast<int>(RecordType::_NrRecords);
    using ResultAllT = ResultAll<NR_RUNS, NR_RECORDS>;
    using ResultT    = ResultBenchmark<NR_RUNS, NR_RECORDS>;
    /// Boost tests will fail when deviation(iFFT(FFT(data)),data) returns a greater value
    static constexpr double ERROR_BOUND = 0.00001;

    static Application& getInstance() {
      static Application app;
      return app;
    }
    static T_Context& getContext() {
      return getInstance().context_;
    }

    bool isContextCreated() const {
      return context_created_;
    }

    void createContext() {
      TimerCPU timer;
      timer.startTimer();
      context_.create();
      timeContextCreate_ = timer.stopTimer();
      context_created_ = true;
    }

    void destroyContext() {
      TimerCPU timer;
      timer.startTimer();
      context_.destroy();
      timeContextDestroy_ = timer.stopTimer();
      context_created_ = false;
    }

    void addRecord(ResultT r) {
      resultAll_.add(r);
    }

    void dumpResults() {
      if(gearshifft::Options::getInstance().getVerbose()) {
        resultAll_.print(std::cout,
                         T_Context::title(),
                         context_.get_used_device_properties(),
                         timeContextCreate_,
                         timeContextDestroy_);
      }

      std::string fname = Options::getInstance().getOutputFile();
      resultAll_.sort();
      resultAll_.saveCSV(fname,
                         T_Context::title(),
                         context_.get_used_device_properties(),
                         timeContextCreate_,
                         timeContextDestroy_);
      std::cout << "Results dumped to "+fname << std::endl;
    }

  private:
    T_Context context_;
    bool context_created_ = false;
    ResultAllT resultAll_;
    ResultT result_;
    double timeContextCreate_ = 0.0;
    double timeContextDestroy_ = 0.0;

    Application() = default;
    ~Application() = default;
  };

} // gearshifft

#endif
