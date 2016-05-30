/*
 * Fixture class runs benchmark by a given Functor and test data from FFTFixtureData.
 * 
 */
#ifndef FIXTURE_BENCHMARK_HPP_
#define FIXTURE_BENCHMARK_HPP_

#include "helper.h"
#include "fixture_data.hpp"

#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <sstream>
#include <fstream>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/test/unit_test.hpp>

// compiler flags, set by cmake
#ifndef BENCH_PRECISION
 #define BENCH_PRECISION float
#endif
#ifndef DEFAULT_RESULT_FILE
 #define DEFAULT_RESULT_FILE results
#endif


namespace gearshifft
{
struct Results
  {
    helper::Statistics stats;
    size_t alloc_mem_in_bytes = 0;
    size_t plan_mem_in_bytes = 0;
  };

struct Output {
    // @todo output filename as parameter and verbosity flag
    template<typename TExtent>
    static void write(const Results& results, const TExtent& extents, const std::string& title, const std::string& test_suite_name) {
      std::stringstream ss;
      int i_stats = results.stats.getLength()-1;
      // output on console
      std::cout << test_suite_name << " " << title << " "
                << extents.size()
                << "D";
      for(const auto& e : extents)
        std::cout << ", "<<e;
      std::cout << ", DevAllocs " << results.alloc_mem_in_bytes / 1048576.0
                << '+' << results.plan_mem_in_bytes / 1048576.0
                << " MiB, "
                << results.stats.getLabel(i_stats) << " = "
                << results.stats.getAverage(i_stats) << results.stats.getUnit(i_stats)
                << std::endl;

      ss <<'\"'<< test_suite_name  << '\"'
         << ",\"" <<  title <<"_"<< extents.size() << "D" << '\"'
         << ",\"Precision\"," << BOOST_PP_STRINGIZE(BENCH_PRECISION)
         << ",\"AllocBuffer\"," <<  results.alloc_mem_in_bytes / 1048576.0 << ",\"MiB\""
         << ",\"AllocPlan\"," <<  results.plan_mem_in_bytes / 1048576.0 << ",\"MiB\""
         << ",\"Extent\"";
      for(const auto& e : extents)
        ss  << ',' << e;
      ss << std::endl;

      ss << results.stats << std::endl;
      write(ss.str());
    }
    static void write(const std::string& content) {
      // resets output file on first touch in application
      static bool reset_file=true;
      std::string fname = BOOST_PP_STRINGIZE(DEFAULT_RESULT_FILE)".csv";
      std::ofstream fp;
      fp.open(fname, reset_file ? std::ofstream::out : std::ofstream::out | std::ofstream::app);
      fp << content;
      fp.close();
      reset_file = false;
    }
  };

  template<typename TPrecision>
  struct FixtureBenchmark
  {
    /// Number of benchmark runs after warmup
    static constexpr int NR_RUNS = 5;
    /// Boost tests will fail when deviation(iFFT(FFT(data)),data) returns a greater value
    const double ERROR_BOUND = 0.00001;
    /// for benchmark statistics
    Results results;

    /**
     * Benchmark body with 1 warmup and NR_RUNS repetitions of iFFT(FFT()) implementation.
     * Implementation is given with TFunctor.
     * Depending on TFunctor::InputIsReal it uses RealType or ComplexType test data.
     * Statistics will be printed after the runs.
     * FFT output will be normalized and compared to original input, which is kept stored in FFTFixtureData.
     * Is called from auto test case function in fixture_test_suite.
     */
    template<typename TFunctor, bool NormalizeResult, size_t NDIM=3>
    void benchmark(const std::array<unsigned, NDIM>& extents)
      {
        using TVector = std::conditional_t<TFunctor::IsComplex,
                                           typename FixtureData<TPrecision,NDIM>::ComplexVector,
                                           typename FixtureData<TPrecision,NDIM>::RealVector>;
        static_assert(NDIM<=3,"NDIM<=3");
    
        auto data_set = FixtureData<TPrecision,NDIM>::getInstancePtr(extents);
        TVector* data_buffer_p = new TVector;
        TVector& data_buffer = *data_buffer_p; // workaround, stack ctor fails
        data_set->copyTo(data_buffer);
        assert(data_buffer.data());

        // warmup
        TFunctor()(data_buffer, extents, results);

        results.stats.resetAll();
        for(int r=0; r<NR_RUNS; ++r)
        {
          data_set->copyTo(data_buffer);
          TFunctor()(data_buffer, extents, results);
        }
        // get test suite name
        std::string test_suite_name = (boost::unit_test::framework::get<boost::unit_test::test_suite>(boost::unit_test::framework::current_test_case().p_parent_id)).p_name;
        Output::write(results, extents, TFunctor::Title, test_suite_name);

        auto deviation = data_set->template check_deviation<NormalizeResult>(data_buffer);
        BOOST_CHECK_LE( deviation, ERROR_BOUND );
        delete data_buffer_p;
      }
  };

  typedef FixtureBenchmark< BENCH_PRECISION > FixtureBenchmarkT;

} // gearshifft

#endif /* FIXTURE_BENCHMARK_HPP_ */
