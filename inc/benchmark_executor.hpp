#ifndef BENCHMARK_EXECUTOR_HPP_
#define BENCHMARK_EXECUTOR_HPP_

#include "application.hpp"
#include "benchmark_data.hpp"
#include "traits.hpp"

#include <type_traits>

#include <boost/test/unit_test.hpp>

namespace gearshifft {
  /**
   * Benchmark body with 1 warmup and NR_RUNS repetitions of iFFT(FFT()) implementation.
   * Implementation is given with TFunctor.
   * Depending on TFunctor::InputIsReal it uses RealType or ComplexType test data.
   * FFT output will be [normalized and] compared to original input.
   */
  template<typename T_Context,
           typename T_Functor,
           typename T_FFT_Normalized,
           typename T_Precision,
           typename T_Extents
           >
  struct BenchmarkExecutor {
    using ApplicationT = Application<T_Context>;
    using ResultT = typename ApplicationT::ResultT;
    static constexpr int NR_RUNS = ApplicationT::NR_RUNS;
    static constexpr double ERROR_BOUND = ApplicationT::ERROR_BOUND;
    static constexpr size_t NDim = std::tuple_size<T_Extents>::value;
    using VectorT = typename std::conditional<T_Functor::IsComplex,
                                              typename BenchmarkData<T_Precision,NDim>::ComplexVector,
                                              typename BenchmarkData<T_Precision,NDim>::RealVector>::type;
    static_assert(NDim<=3,"NDim<=3");

    void operator()(const T_Extents& extents) {
      auto data_set = BenchmarkData<T_Precision,NDim>::getInstancePtr(extents);
      VectorT data_buffer;
      data_set->copyTo(data_buffer);
      assert(data_buffer.data());

      auto fft = T_Functor();
      ResultT result;
      result.template init<T_Functor::IsComplex, T_Functor::IsInplace >
                       (extents, ToString<T_Precision>::value() );
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
