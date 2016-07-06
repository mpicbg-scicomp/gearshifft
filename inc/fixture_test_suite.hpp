/**
 *
 */
#ifndef FIXTURE_TEST_SUITE_HPP_
#define FIXTURE_TEST_SUITE_HPP_

#include "application.hpp"
#include "fixture_benchmark.hpp"
#include "traits.hpp"
#include <ostream>
#include <array>
#include <boost/type_traits/function_traits.hpp>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/mpl/list.hpp>
#include <boost/mpl/for_each.hpp>

namespace gearshifft {

  /// list alias for user
  template<typename... Types>
  using List = boost::mpl::list<Types...>;
  using benchmark_suite = boost::unit_test::test_suite;

  template<size_t NDim>
  inline
  std::ostream& operator<<(std::ostream& os, const std::array<unsigned,NDim>& e) {
    os << e[0];
    for(size_t k=1; k<NDim; ++k)
      os << "x" << e[k];
  }

  template<typename T_Context,
           typename T_FFT_Normalized,
           typename T_FFTs,
           typename T_Precision>
  struct BenchmarkFactory {
    /// Functor for mpl::for_each<T_FFTs>
    template<typename T_Extents>
    struct Apply {
      const T_Extents e_;
      benchmark_suite* ts_;
      explicit Apply(const T_Extents& e) : e_(e) {
        std::stringstream ss;
        ss << e;
        ts_ = BOOST_TEST_SUITE( ss.str() );
      }
      template<typename FFT>
      void operator()(FFT) {
        static_assert( has_title<FFT>::value, "FFT Implementation has no static title method.");
        using FixtureBenchmarkT = FixtureBenchmark<T_Context,
                                                   FFT,
                                                   T_FFT_Normalized,
                                                   T_Precision,
                                                   T_Extents>;
        FixtureBenchmarkT benchmark;
        boost::unit_test::test_case* s = BOOST_TEST_CASE(
          boost::bind((&FixtureBenchmarkT::operator()),benchmark, e_));
        s->p_name.value = FFT::Title;
        ts_->add(s);
      }
      benchmark_suite* result() { return ts_; }
    }; // Apply

    template<typename T_Extents>
    benchmark_suite* createSuite(const T_Extents& e) {
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
      benchmark_suite* suite_;
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
        benchmark_suite* sub_suite = BOOST_TEST_SUITE( ToString<T_Precision>::value() );
        App& app = App::getInstance();
        auto extents1D = app.getExtents1D();
        for(auto e : extents1D) {
          sub_suite->add( factory.createSuite(e) );
        }
        auto extents2D = app.getExtents2D();
        for(auto e : extents2D) {
          sub_suite->add( factory.createSuite(e) );
        }
        auto extents3D = app.getExtents3D();
        for(auto e : extents3D) {
          sub_suite->add( factory.createSuite(e) );
        }
        suite_->add(sub_suite);
      }
      benchmark_suite* result() { return suite_; }
    }; // Apply

    /// create FFT benchmarks by FFT classes and FFT extents
    benchmark_suite* operator()() {
      Apply apply(title_);
      boost::mpl::for_each<T_Precisions>( apply );
      return apply.result();
    }
  }; // Run
}

#endif
