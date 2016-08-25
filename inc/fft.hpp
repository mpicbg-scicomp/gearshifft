#ifndef FFT_HPP_
#define FFT_HPP_

#include "timer_cpu.hpp"
#include "traits.hpp"
#include "types.hpp"

#include <assert.h>
#include <array>
#include <type_traits>
#include <ostream>

namespace gearshifft {

  struct FFT_Inplace_Real {
    static constexpr auto IsComplex = false;
    static constexpr auto Title = "Inplace_Real";
    static constexpr auto IsInplace = true;
  };
  struct  FFT_Outplace_Real {
    static constexpr auto IsComplex = false;
    static constexpr auto Title = "Outplace_Real";
    static constexpr auto IsInplace = false;
  };

  struct FFT_Inplace_Complex {
    static constexpr auto IsComplex = true;
    static constexpr auto Title = "Inplace_Complex";
    static constexpr auto IsInplace = true;
  };

  struct FFT_Outplace_Complex {
    static constexpr auto IsComplex = true;
    static constexpr auto Title = "Outplace_Complex";
    static constexpr auto IsInplace = false;
  };

/**
 * Functor being called from FixtureBenchmark::benchmark()
 */
  template<typename TFFT, // FFT_*_* [inplace.., real..]
           template <typename,typename,size_t> typename TPlan,
           typename TDeviceTimer
           >
  struct FFT : public TFFT {
    /**
     * Called by FixtureBenchmark
     * \tparam T_Result ResultBenchmark<NR_RUNS, NR_RECORDS>, also see class Application.
     * \tparam T_Vector
     * \tparam NDim Number of FFT dimensions
     */
    template<typename T_Result, typename T_Vector, size_t NDim>
    void operator()(T_Result& result,
                    T_Vector& vec,
                    const std::array<size_t,NDim>& extents
      ) {
      using PrecisionT = typename Precision<typename T_Vector::value_type,
                                            TFFT::IsComplex >::type;
      assert(vec.size());

      // prepare plan object
      // templates in: FFT type: in[,out][complex], PlanImpl, Precision, NDim
      auto plan = TPlan<TFFT, PrecisionT, NDim> (extents);
      result.setValue(RecordType::DevBufferSize, plan.getAllocSize());
      result.setValue(RecordType::DevPlanSize, plan.getPlanSize());
      result.setValue(RecordType::DevTransferSize, plan.getTransferSize());

      TimerCPU ttotal;
      TimerCPU talloc;
      TimerCPU tplaninit;
      TimerCPU tplandestroy;
      TDeviceTimer tdevupload;
      TDeviceTimer tdevdownload;
      TDeviceTimer tdevfft;
      TDeviceTimer tdevfftinverse;
      TimerCPU tdevtotal;
      /// --- Total CPU ---
      ttotal.startTimer();
      /// --- Malloc ---
      talloc.startTimer();
      plan.malloc();
      result.setValue(RecordType::Allocation, talloc.stopTimer());

      /// --- Create plan ---
      tplaninit.startTimer();
      plan.init_forward();
      result.setValue(RecordType::PlanInit, tplaninit.stopTimer());

      /// --- FFT+iFFT GPU ---
      tdevtotal.startTimer();

      tdevupload.startTimer();
      plan.upload(vec.data());
      result.setValue(RecordType::Upload, tdevupload.stopTimer());

      tdevfft.startTimer();
      plan.execute_forward();
      result.setValue(RecordType::FFT, tdevfft.stopTimer());

      plan.init_backward();

      tdevfftinverse.startTimer();
      plan.execute_backward();
      result.setValue(RecordType::FFTInverse, tdevfftinverse.stopTimer());

      tdevdownload.startTimer();
      plan.download(vec.data());
      result.setValue(RecordType::Download, tdevdownload.stopTimer());

      result.setValue(RecordType::Device, tdevtotal.stopTimer());

      /// --- Cleanup ---
      tplandestroy.startTimer();
      plan.destroy();
      result.setValue(RecordType::PlanDestroy, tplandestroy.stopTimer());
      result.setValue(RecordType::Total, ttotal.stopTimer());
    }
  };

}
#endif /* FFT_HPP_ */
