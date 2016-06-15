/**
 * Helper functions to control Boost automatic test suite.
 * Provides RUN_BENCHMARKS() function for FFT instances, 
 * which are [compile-time constant] Functors.
 */
#ifndef FIXTURE_TEST_SUITE_HPP_
#define FIXTURE_TEST_SUITE_HPP_

#include "application.hpp"
#include "fixture_benchmark.hpp"

#include <array>
#include <boost/type_traits/function_traits.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/tuple/enum.hpp>
#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/seq/cat.hpp>

//____________________________

// dimension test cases for benchmark (append or modify if needed)
#define EXTENTS ((32,32,32)) ((27,25,49)) ((1024)) ((16,25))

// if EXTENTS Tuple is not defined, then use the predefined ones
#ifndef EXTENTS
#include "fixture_test_suite_extents.hpp"
#endif
//____________________________

// compiler flag, set by cmake
#ifndef BENCH_PRECISION
 #define BENCH_PRECISION float
#endif


/* preprocessor helper functions for boost auto test framework */
// (ExampleName) -> ExampleNameType
#define MALGORITHMS_TYPE(s) BOOST_PP_CAT(s,Type)
// (<TUPLE>,i) -> TUPLE[i]
#define MGET(elem,i) BOOST_PP_TUPLE_ELEM(i, elem)
// (Name,<TUPLE>) -> TestNamexxx
#define MTEST_NAME(Name,elem) BOOST_PP_CAT(BOOST_PP_CAT(Test,Name),BOOST_PP_SEQ_CAT(BOOST_PP_TUPLE_TO_SEQ(elem)))

// (-,SuitName,<TUPLE>) -> generates auto template test body. Called in BOOST_PP_SEQ_FOR_EACH
#define MRUN_BENCHMARK_UNNORMALIZED_FFT(s, SuitName, elem) \
BOOST_AUTO_TEST_CASE_TEMPLATE(MTEST_NAME(SuitName,elem), T, MALGORITHMS_TYPE(SuitName)) \
{\
  std::array<unsigned, BOOST_PP_TUPLE_SIZE(elem)> extents = {BOOST_PP_TUPLE_ENUM(elem)}; \
  benchmark<T,true>(extents);\
}
#define MRUN_BENCHMARK_NORMALIZED_FFT(s, SuitName, elem) \
BOOST_AUTO_TEST_CASE_TEMPLATE(MTEST_NAME(SuitName,elem), T, MALGORITHMS_TYPE(SuitName)) \
{\
  std::array<unsigned, BOOST_PP_TUPLE_SIZE(elem)> extents = {BOOST_PP_TUPLE_ENUM(elem)};\
  benchmark<T,false>(extents);\
}
/**
 * (SuitName, Algorithms)
 * Algorithm is a [compile-time constant] Functor.
 *
 * Inverse FFT must be normalized to compare results.
 */
#define RUN_BENCHMARKS_UNNORMALIZED_FFT(SuitName, ContextType, ...)      \
namespace gearshifft { \
  typedef boost::mpl::list<__VA_ARGS__> MALGORITHMS_TYPE(SuitName);     \
  typedef Application<ContextType> ApplicationT;                 \
  typedef ApplicationFixture<ContextType> ApplicationFixtureT;                 \
  typedef FixtureBenchmark< BENCH_PRECISION, ApplicationT > FixtureBenchmarkT; \
  BOOST_GLOBAL_FIXTURE( ApplicationFixtureT );              \
  BOOST_FIXTURE_TEST_SUITE(SuitName, FixtureBenchmarkT)                 \
   BOOST_PP_SEQ_FOR_EACH(MRUN_BENCHMARK_UNNORMALIZED_FFT,SuitName,EXTENTS) \
  BOOST_AUTO_TEST_SUITE_END()                                           \
}

/**
 * (SuitName, Algorithms)
 * Algorithm is a [compile-time constant] Functor.
 *
 * Inverse FFT already is normalized.
 */
#define RUN_BENCHMARKS_NORMALIZED_FFT(SuitName, ContextType, ...)  \
namespace gearshifft { \
  typedef boost::mpl::list<__VA_ARGS__> MALGORITHMS_TYPE(SuitName);       \
  typedef Application<ContextType> ApplicationT;                 \
  typedef ApplicationFixture<ContextType> ApplicationFixtureT;                 \
  typedef FixtureBenchmark< BENCH_PRECISION, ApplicationT > FixtureBenchmarkT; \
  BOOST_GLOBAL_FIXTURE( ApplicationFixtureT );            \
  BOOST_FIXTURE_TEST_SUITE(SuitName, gearshifft::FixtureBenchmarkT)     \
   BOOST_PP_SEQ_FOR_EACH(MRUN_BENCHMARK_NORMALIZED_FFT,SuitName,EXTENTS) \
  BOOST_AUTO_TEST_SUITE_END() \
}


#endif
