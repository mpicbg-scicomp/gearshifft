#ifndef APPLICATION_HPP_
#define APPLICATION_HPP_

#include "result_benchmark.hpp"
#include "result_all.hpp"
#include "timer_cpu.hpp"
#include "types.hpp"

#include <boost/asio.hpp>

#include <ctime>
#include <vector>
#include <array>
#include <sstream>


#ifndef GEARSHIFFT_NUMBER_WARM_RUNS
#define GEARSHIFFT_NUMBER_WARM_RUNS 10
#endif

#ifndef GEARSHIFFT_NUMBER_WARMUPS
#define GEARSHIFFT_NUMBER_WARMUPS 2
#endif

#ifndef GEARSHIFFT_NUMBER_RUNS
#define GEARSHIFFT_NUMBER_RUNS GEARSHIFFT_NUMBER_WARM_RUNS + GEARSHIFFT_NUMBER_WARMUPS
#endif

#ifndef GEARSHIFFT_ERROR_BOUND
#define GEARSHIFFT_ERROR_BOUND 0.00001
#endif

namespace gearshifft {

  inline
  std::string gearshifft_version() {
    return std::to_string(GEARSHIFFT_VERSION_MAJOR) + "." + std::to_string(GEARSHIFFT_VERSION_MINOR) + "." + std::to_string(GEARSHIFFT_VERSION_PATCH);
  }

  template<typename T_Context>
  class Application {
  public:
    /// Number of benchmark runs after warmup
    static constexpr int NR_RUNS = GEARSHIFFT_NUMBER_RUNS;
    static constexpr int NR_WARM_RUNS = GEARSHIFFT_NUMBER_WARM_RUNS;
    static constexpr int NR_WARMUP_RUNS = GEARSHIFFT_NUMBER_WARMUPS;
    static const int NR_RECORDS  = static_cast<int>(RecordType::_NrRecords);
    using ResultAllT = ResultAll<NR_RUNS, NR_WARMUP_RUNS, NR_RECORDS>;
    using ResultT    = ResultBenchmark<NR_RUNS, NR_RECORDS>;
    /// Boost tests will fail when deviation(iFFT(FFT(data)),data) returns a greater value
    static constexpr double ERROR_BOUND = GEARSHIFFT_ERROR_BOUND;

    static Application& getInstance() {
      static Application app;
      return app;
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
      std::time_t now = std::time(nullptr);
      std::stringstream meta_information;
      meta_information << context_.get_used_device_properties()
                       << ",\"NumberWarmups\"," << NR_WARMUP_RUNS
                       << ",\"NumberWarmRuns\"," << NR_WARM_RUNS
                       << ",\"NumberTotalRuns\"," << NR_RUNS
                       << ",\"ErrorBound\"," << ERROR_BOUND
                       << ",\"CurrentTime\"," << now
                       << ",\"CurrentTimeLocal\",\"" << strtok(ctime(&now), "\n") << "\""
                       << ",\"Hostname\",\"" << boost::asio::ip::host_name() << "\""
                       << ",\"gearshifft\",\"" << gearshifft_version() << "\""
        ;
      if(T_Context::options().getVerbose()) {
        resultAll_.print(std::cout,
                         T_Context::title(),
                         meta_information.str(),
                         timeContextCreate_,
                         timeContextDestroy_);
      }

      std::string fname = T_Context::options().getOutputFile();
      resultAll_.sort();
      resultAll_.saveCSV(fname,
                         T_Context::title(),
                         meta_information.str(),
                         timeContextCreate_,
                         timeContextDestroy_);
      std::cout << "Results dumped to " << fname << std::endl;
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
