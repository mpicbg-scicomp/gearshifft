/*
 * Fixture class runs benchmark by a given Functor and test data from FFTFixtureData.
 * 
 */
#ifndef FIXTURE_BENCHMARK_HPP_
#define FIXTURE_BENCHMARK_HPP_

#include "result_benchmark.hpp"
#include "result_all.hpp"
#include "fixture_data.hpp"

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

  template<typename T_Precision>
  struct FixtureBenchmark
  {
    /// Number of benchmark runs after warmup
    static constexpr int NR_RUNS = 5;
    static constexpr int NR_TIMER = 8;
    using ResultAllT = ResultAll<NR_RUNS, NR_TIMER>;
    using ResultT = typename ResultAllT::ResultBenchmarkT;
    /// Boost tests will fail when deviation(iFFT(FFT(data)),data) returns a greater value
    const double ERROR_BOUND = 0.00001;

    /**
     * Benchmark body with 1 warmup and NR_RUNS repetitions of iFFT(FFT()) implementation.
     * Implementation is given with TFunctor.
     * Depending on TFunctor::InputIsReal it uses RealType or ComplexType test data.
     * Statistics will be printed after the runs.
     * FFT output will be normalized and compared to original input, which is kept stored in FFTFixtureData.
     * Is called from auto test case function in fixture_test_suite.
     */
    template<typename T_Functor, bool NormalizeResult, size_t T_NDim=3>
    void benchmark(const std::array<unsigned, T_NDim>& extents)
      {
        using VectorT = std::conditional_t<T_Functor::IsComplex,
                                           typename FixtureData<T_Precision,T_NDim>::ComplexVector,
                                           typename FixtureData<T_Precision,T_NDim>::RealVector>;
        static_assert(T_NDim<=3,"T_NDim<=3");
        /// for benchmark statistics
        static ResultAllT resultAll;
    
        auto data_set = FixtureData<T_Precision,T_NDim>::getInstancePtr(extents);
        VectorT* data_buffer_p = new VectorT;
        VectorT& data_buffer = *data_buffer_p; // @todo check: workaround, stack ctor fails
        data_set->copyTo(data_buffer);
        assert(data_buffer.data());

        auto fft = T_Functor();
        ResultT result;
        result.template init<T_Functor::IsComplex,
                             T_Functor::IsInplace,
                             T_NDim>(extents);
        // warmup
        fft(result, data_buffer, extents);

        for(int r=0; r<NR_RUNS; ++r)
        {
          result.setRun(r);
          data_set->copyTo(data_buffer);
          fft(result, data_buffer, extents);
        }
        auto deviation = data_set->template check_deviation<NormalizeResult>(data_buffer);
        BOOST_CHECK_LE( deviation, ERROR_BOUND );

        resultAll.add(result);

        delete data_buffer_p;
      }
  };

  typedef FixtureBenchmark< BENCH_PRECISION > FixtureBenchmarkT;

} // gearshifft

#endif /* FIXTURE_BENCHMARK_HPP_ */
