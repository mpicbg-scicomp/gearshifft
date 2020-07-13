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

#ifndef GEARSHIFFT_DUMP_FREQUENCY
#define GEARSHIFFT_DUMP_FREQUENCY 1
#endif

namespace gearshifft {

  inline
  std::string version() {
    return std::to_string(GEARSHIFFT_VERSION_MAJOR) + "."
         + std::to_string(GEARSHIFFT_VERSION_MINOR) + "."
         + std::to_string(GEARSHIFFT_VERSION_PATCH)
#ifdef GEARSHIFFT_SCOREP_INSTRUMENTATION
         + "-scorep"
#endif
            ;
  }

  template<typename T_Context>
  class Application {
  public:
    /// Number of benchmark runs after warmup
    static constexpr int NR_RUNS = GEARSHIFFT_NUMBER_RUNS;
    static constexpr int NR_WARM_RUNS = GEARSHIFFT_NUMBER_WARM_RUNS;
    static constexpr int NR_WARMUP_RUNS = GEARSHIFFT_NUMBER_WARMUPS;
    static constexpr int NR_RECORDS  = static_cast<int>(RecordType::NrRecords_);
    static constexpr int DUMP_FREQUENCY = GEARSHIFFT_DUMP_FREQUENCY;
    using ResultAllT    = ResultAll<NR_RUNS, NR_WARMUP_RUNS, NR_RECORDS>;
    using ResultWriterT = ResultWriter<NR_RUNS, NR_WARMUP_RUNS, NR_RECORDS>;
    using ResultT       = ResultBenchmark<NR_RUNS, NR_RECORDS>;
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

    void addRecord(const ResultT& r) {
      resultAll_.add(r);

      if (resultAll_.size() % DUMP_FREQUENCY == 0) {
        resultWriter_.update();
      }
    }

    void startWriter() {

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
                       << ",\"gearshifft\",\"" << gearshifft::version() << "\"";

      resultWriter_.start(&resultAll_,
                          T_Context::options().getOutputFile(),
                          T_Context::title(),
                          meta_information.str(),
                          T_Context::options().getVerbose());
    }

    void stopWriter() {
      resultWriter_.stop(timeContextCreate_, timeContextDestroy_);
    }

  private:
    T_Context context_;
    bool context_created_ = false;
    ResultAllT resultAll_;
    ResultWriterT resultWriter_;
    double timeContextCreate_ = 0.0;
    double timeContextDestroy_ = 0.0;

    Application() = default;
    ~Application() = default;
  };

} // gearshifft

#endif
