#ifndef BENCHMARK_SUITE_HPP_
#define BENCHMARK_SUITE_HPP_

#include "application.hpp"
#include "options.hpp"
#include "benchmark_executor.hpp"
#include "traits.hpp"
#include "types.hpp"

#include <array>

#include <boost/bind.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/mpl/for_each.hpp>

namespace gearshifft {

  using boost::unit_test::test_suite;

  template<typename T_Context,
           typename T_FFT_Normalized,
           typename T_FFTs,
           typename T_Precision>
  struct BenchmarkFactory {
    /// Functor for mpl::for_each<T_FFTs>
    template<typename T_Extents>
    struct Apply {
      const T_Extents e_;
      test_suite* ts_;
      explicit Apply(const T_Extents& e) : e_(e) {
        std::stringstream ss;
        ss << e;
        ts_ = BOOST_TEST_SUITE( ss.str() );
      }
      template<typename FFT>
      void operator()(FFT) {
        static_assert( has_title<FFT>::value, "FFT Implementation has no static title method.");
        using BenchmarkExecutorT = BenchmarkExecutor<T_Context,
                                                     FFT,
                                                     T_FFT_Normalized,
                                                     T_Precision,
                                                     T_Extents>;
        BenchmarkExecutorT benchmark;
        boost::unit_test::test_case* s = BOOST_TEST_CASE(
            boost::bind((&BenchmarkExecutorT::operator()), benchmark, e_)
          );
        s->p_name.value = FFT::Title;
        ts_->add(s);
      }
      test_suite* result() { return ts_; }
    }; // Apply

    template<typename T_Extents>
    test_suite* createSuite(const T_Extents& e) {
      Apply<T_Extents> apply(e);
      boost::mpl::for_each<T_FFTs>( apply );
      return apply.result();
    }
  }; // BenchmarkFactory

  template<typename T_Context,
           typename T_FFT_Normalized,
           typename T_FFTs,
           typename T_Precisions
           >
  struct Run {
    const std::string title_;
    Run( ) : title_(T_Context::title()) { }
    Run( const char* title ) : title_(title) { }

    /// Functor for mpl::for_each<T_Precisions>
    struct Apply {
      test_suite* suite_;
      Apply(const std::string title) {
        suite_ = BOOST_TEST_SUITE( title );
      }
      template<typename T_Precision>
      void operator() (T_Precision) {
        using Factory = BenchmarkFactory<T_Context,
                                         T_FFT_Normalized,
                                         T_FFTs,
                                         T_Precision>;
        using App = Application<T_Context>;
        Factory factory;
        test_suite* sub_suite = BOOST_TEST_SUITE( ToString<T_Precision>::value() );
        Options& options = Options::getInstance();
        auto extents1D = options.getExtents1D();
        for(auto e : extents1D) {
          sub_suite->add( factory.createSuite(e) );
        }
        auto extents2D = options.getExtents2D();
        for(auto e : extents2D) {
          sub_suite->add( factory.createSuite(e) );
        }
        auto extents3D = options.getExtents3D();
        for(auto e : extents3D) {
          sub_suite->add( factory.createSuite(e) );
        }
        suite_->add(sub_suite);
      }
      test_suite* result() { return suite_; }
    }; // Apply

    /// create FFT benchmarks by FFT classes and FFT extents
    test_suite* operator()() {
      Apply apply(title_);
      boost::mpl::for_each<T_Precisions>( apply );
      return apply.result();
    }
  }; // Run
}

#endif
