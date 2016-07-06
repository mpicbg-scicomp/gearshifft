/*
 * Fixture class runs benchmark by a given Functor and test data from FFTFixtureData.
 *
 */
#ifndef FIXTURE_BENCHMARK_HPP_
#define FIXTURE_BENCHMARK_HPP_

#include "application.hpp"
#include "fixture_data.hpp"
#include "traits.hpp"
#include <stdlib.h>
#include <math.h>
#include <type_traits>
#include <boost/test/unit_test.hpp>

namespace gearshifft
{
  /*
fftkind, precision, dimkind, dim, nx, ny, nz, run, AllocBuffer (bytes), AllocPlan (bytes), {times}, time-unit
"Inplace Complex", float, 1, 3, 63,67,65, 0, ..., ..., ..., ms
   */

  /**
   * Benchmark body with 1 warmup and NR_RUNS repetitions of iFFT(FFT()) implementation.
   * Implementation is given with TFunctor.
   * Depending on TFunctor::InputIsReal it uses RealType or ComplexType test data.
   * Statistics will be printed after the runs.
   * FFT output will be normalized and compared to original input, which is kept stored in FFTFixtureData.
   * Is called from auto test case function in fixture_test_suite.
   */
  template<typename T_Context,
           typename T_Functor,
           typename T_FFT_Normalized,
           typename T_Precision,
           typename T_Extents
           >
  struct FixtureBenchmark {
    using ApplicationT = Application<T_Context>;
    using ResultT = typename ApplicationT::ResultT;
    static constexpr int NR_RUNS = ApplicationT::NR_RUNS;
    static constexpr double ERROR_BOUND = ApplicationT::ERROR_BOUND;
    static constexpr size_t NDim = std::tuple_size<T_Extents>::value;
    using VectorT = typename std::conditional<T_Functor::IsComplex,
                                              typename FixtureData<T_Precision,NDim>::ComplexVector,
                                              typename FixtureData<T_Precision,NDim>::RealVector>::type;
    static_assert(NDim<=3,"NDim<=3");

    void operator()(const T_Extents& extents) {
      auto data_set = FixtureData<T_Precision,NDim>::getInstancePtr(extents);
      VectorT data_buffer;
      data_set->copyTo(data_buffer);
      assert(data_buffer.data());

      auto fft = T_Functor();
      ResultT result;
      result.template init<T_Functor::IsComplex,
                           T_Functor::IsInplace > (extents,
                                                   ToString<T_Precision>::value() );
      try {
        // warmup
        fft(result, data_buffer, extents);
        for(int r=0; r<NR_RUNS; ++r)
        {
          result.setRun(r);
          data_set->copyTo(data_buffer);
          fft(result, data_buffer, extents);
        }
      } catch(const std::runtime_error& e) {
        BOOST_FAIL( e.what() );
      }
      auto deviation = data_set->template check_deviation
        <!T_FFT_Normalized::value> (data_buffer);
      const double error_bound = ERROR_BOUND;
      BOOST_CHECK_LE( deviation, error_bound );

      ApplicationT::getInstance().addRecord(result);
    }
  };

} // gearshifft

#endif /* FIXTURE_BENCHMARK_HPP_ */
