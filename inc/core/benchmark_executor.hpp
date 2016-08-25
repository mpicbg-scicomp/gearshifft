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
      int r = 0;
      VectorT data_buffer;
      auto data_set = BenchmarkData<T_Precision,NDim>::getInstancePtr(extents);
      data_set->copyTo(data_buffer);
      assert(data_buffer.data());
      auto fft = T_Functor();
      ResultT result;
      result.template init<T_Functor::IsComplex, T_Functor::IsInplace >
                       (extents, ToString<T_Precision>::value() );
      try {
        // warmup
        fft(result, data_buffer, extents);
        for(r=0; r<NR_RUNS; ++r)
        {
          result.setRun(r);
          data_set->copyTo(data_buffer);
          fft(result, data_buffer, extents);

          auto deviation = data_set->template check_deviation
            <!T_FFT_Normalized::value> (data_buffer);
          if(deviation > ERROR_BOUND) {
            std::stringstream msg;
            msg << "Results mismatch: deviation=" << deviation << " [" << ERROR_BOUND << "]";
            result.setError(r, msg.str());
            throw std::runtime_error(msg.str());
          }
        }
      } catch(const std::runtime_error& e) {
        result.setError(r, e.what());
        ApplicationT::getInstance().addRecord(result);
        BOOST_FAIL( e.what() );
      }

      ApplicationT::getInstance().addRecord(result);
    }
  };

} // gearshifft

#endif /* FIXTURE_BENCHMARK_HPP_ */
