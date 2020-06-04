#ifndef BENCHMARK_HPP_
#define BENCHMARK_HPP_
// we customize the main function of the unit test framework
#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "application.hpp"
#include "benchmark_suite.hpp"

// see https://www.boost.org/doc/libs/1_65_1/libs/test/doc/html/boost_test/usage_variants.html
// Single-header usage variant
// No BOOST_TEST_MODULE as entry point is customized
#include <boost/test/included/unit_test.hpp>

namespace gearshifft {

  /**
   * Benchmark API class for clients.
   *
   */
  template<typename Context>
  class Benchmark {
    using AppT = Application<Context>;

  public:
    Benchmark() = default;

    ~Benchmark() {
      if(AppT::getInstance().isContextCreated()) {
        AppT::getInstance().destroyContext();
      }
    }

    void configure(int argc, char* argv[]) {
      std::vector<char*> vargv(argv, argv+argc);
      boost_vargv_.clear();
      boost_vargv_.emplace_back(argv[0]); // [0] = name of application
      auto parseResult = Context::options().parse(vargv, boost_vargv_);
      switch (parseResult) {
        case 0:  break;
        case 1:  info_only_ = true; break;
        default: parsing_failed_ = true;
      }
    }

    template<typename T_FFT_Is_Normalized,
             typename T_FFTs,
             typename T_Precisions>
    int run() {
      if (parsing_failed_) {
        return 1;
      } else if (info_only_) {
        if(Context::options().getListDevices()) {
          std::cout << Context::get_device_list();
        } else if (Context::options().getVersion()) {
          std::cout << "gearshifft " << gearshifft::version() << '\n';
        } else if (Context::options().getHelp()) {
          std::cout << "gearshifft " << gearshifft::version() << '\n'
                    << Context::options().getDescription();
        }
        return 0;
      }

      AppT::getInstance().createContext();
      AppT::getInstance().startWriter();

      auto init_function = []() {
        BenchmarkSuite<Context, T_FFT_Is_Normalized, T_FFTs, T_Precisions> instance;
        ::boost::unit_test::framework::master_test_suite().add( instance() );
        return true;
      };

      int r = ::boost::unit_test::unit_test_main( init_function,
                                                  boost_vargv_.size(),
                                                  boost_vargv_.data() );

      AppT::getInstance().destroyContext();
      AppT::getInstance().stopWriter();
      return r;
    }

  private:
    bool info_only_ = false;
    bool parsing_failed_ = false;
    std::vector<char*> boost_vargv_;
  };

} // gearshifft

#endif // BENCHMARK_HPP_
