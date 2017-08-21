#ifndef BENCHMARK_EXECUTOR_HPP_
#define BENCHMARK_EXECUTOR_HPP_

#include "application.hpp"
#include "benchmark_data.hpp"
#include "types.hpp"

#include <boost/test/unit_test.hpp>

#include <type_traits>

#include <cmath>

namespace gearshifft {
  /**
   * Benchmark body with NR_RUNS repetitions of iFFT(FFT()) implementation.
   * Implementation is given with TFunctor.
   * Depending on TFunctor::InputIsReal it uses RealType or ComplexType test data.
   * FFT output will be [normalized and] compared to original input.
   */
  template<typename T_Context,
           typename T_FFT_Wrapper,
           typename T_FFT_Normalized,
           typename T_Precision,
           typename T_Extents
           >
  struct BenchmarkExecutor {
    using ApplicationT = Application<T_Context>;
    using ResultT = typename ApplicationT::ResultT;
    static constexpr int NR_RUNS = ApplicationT::NR_RUNS; // includes warmups
    static constexpr double ERROR_BOUND = ApplicationT::ERROR_BOUND;
    static constexpr size_t NDim = std::tuple_size<T_Extents>::value;
    using VectorT = typename std::conditional<T_FFT_Wrapper::IsComplex,
                                              typename BenchmarkData<T_Precision,NDim>::ComplexVector,
                                              typename BenchmarkData<T_Precision,NDim>::RealVector>::type;
    static_assert(NDim<=3,"NDim<=3");

    void operator()(const T_Extents& extents) {
      const auto& dataset = BenchmarkData<T_Precision,NDim>::data(extents);

      VectorT data_buffer;
      dataset.copyTo(data_buffer);
      assert(data_buffer.data());

      auto fft = T_FFT_Wrapper();
      ResultT result;
      result.template init<T_FFT_Wrapper::IsComplex,
                           T_FFT_Wrapper::IsInplace,
                           T_Precision >
                       (extents);

      int r;
      if(result.getID()==0) {
        // hidden initial warmup, which is not recorded (i.e. run=0 will be overwritten)
        for(r=0; r<2*NR_RUNS; ++r) {
          dataset.copyTo(data_buffer);
          fft(result, data_buffer, extents);
        }
      }

      try {
        const double error_bound = ERROR_BOUND<0.0 ? ErrorBound<T_Precision>()() : ERROR_BOUND;
        for(r=0; r<NR_RUNS; ++r)
        {
          result.setRun(r);
          dataset.copyTo(data_buffer);
          fft(result, data_buffer, extents);

          double deviation = 0.0; // sample standard deviation
          size_t mismatches = 0; // nr of mismatches
          // compute deviation and mismatches
          dataset.template check_deviation<!T_FFT_Normalized::value>
            (deviation, mismatches, data_buffer, error_bound);

          result.setValue(RecordType::Deviation, deviation);
          result.setValue(RecordType::Mismatches, static_cast<double>(mismatches));

          if(std::isnan(deviation) || deviation>error_bound) {
            std::stringstream msg;
            msg << "mismatches=" << mismatches
                << " deviation=" << deviation
                << " errorbound=" << error_bound;
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
