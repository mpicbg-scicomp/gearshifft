#ifndef APPLICATION_HPP_
#define APPLICATION_HPP_

#include "fft_abstract.hpp"
#include "result_benchmark.hpp"
#include "result_all.hpp"
#include "timer.hpp"

#include <vector>
#include <array>

namespace gearshifft {

  class Settings {
  public:
    static Settings& getInstance() {
      static Settings settings;
      return settings;
    }
  private:
    Settings() {
      int argc = boost::unit_test::framework::master_test_suite().argc;
      char** argv = boost::unit_test::framework::master_test_suite().argv;
      int verbose = 0;
      for(int k=0; k<argc; ++k) {
        if(strcmp(argv[k],"v")==0) {
          verbose_ = true;
          break;
        }

      }
    }
  private:
    bool verbose_;
    std::string fileExtents_;
  };

  template<typename T_Context>
  class Application {
  public:
    /// Number of benchmark runs after warmup
    static constexpr int NR_RUNS    = 5;
    static const int NR_RECORDS = static_cast<int>(RecordType::_NrRecords);
    using ResultAllT = ResultAll<NR_RUNS, NR_RECORDS>;
    using ResultT    = ResultBenchmark<NR_RUNS, NR_RECORDS>;
    /// Boost tests will fail when deviation(iFFT(FFT(data)),data) returns a greater value
    static constexpr double ERROR_BOUND = 0.00001;
    using Extents1D = std::array<unsigned,1>;
    using Extents2D = std::array<unsigned,2>;
    using Extents3D = std::array<unsigned,3>;
    using Extents1DVec = std::vector< Extents1D >;
    using Extents2DVec = std::vector< Extents2D >;
    using Extents3DVec = std::vector< Extents3D >;

    static Application& getInstance() {
      static Application app;
      return app;
    }
    static T_Context& getContext() {
      return getInstance().context_;
    }

    void createContext() {
      TimerCPU timer;
      timer.startTimer();
      context_.create();
      timeContextCreate_ = timer.stopTimer();
    }

    void destroyContext() {
      TimerCPU timer;
      timer.startTimer();
      context_.destroy();
      timeContextDestroy_ = timer.stopTimer();
    }

    void addRecord(ResultT r) {
      resultAll_.add(r);
    }

    void dumpResults() {
      resultAll_.write(T_Context::title(),
                       context_.getDeviceInfos(),
                       timeContextCreate_,
                       timeContextDestroy_);
    }

    Extents1DVec getExtents1D() {
      Extents1D extents = {1024};
      Extents1DVec vector;
      vector.push_back(extents);
      return vector;
    }
    Extents2DVec getExtents2D() {
      Extents2D extents = {1024,1024};
      Extents2DVec vector;
      vector.push_back(extents);
      return vector;
    }
    Extents3DVec getExtents3D() {
      Extents3D extents = {64,64,64};
      Extents3DVec vector;
      vector.push_back(extents);
      return vector;
    }

  private:
    T_Context context_;
    ResultAllT resultAll_;
    ResultT result_;
    double timeContextCreate_ = 0.0;
    double timeContextDestroy_ = 0.0;

    Application() {
    }
    ~Application() {
    }
  };

  template<typename T_Context>
  struct FixtureApplication {
    using ApplicationT = Application<T_Context>;

    FixtureApplication() {
      ApplicationT::getInstance().createContext();
    }
    ~FixtureApplication() {
      ApplicationT::getInstance().destroyContext();
      ApplicationT::getInstance().dumpResults();
    }
  };

}

#endif
