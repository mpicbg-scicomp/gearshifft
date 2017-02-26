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

  using FFT_Plan_Reusable = std::true_type;
  using FFT_Plan_Not_Reusable = std::false_type;

/**
 * Functor being called by BenchmarkExecutor
 */
  template<typename T_FFT, // FFT_*_* [inplace.., real..]
           typename T_ReusePlan, // can plan be reused ?
           template <typename,typename,size_t,typename... > typename T_Client, // user implementation FFT client
           typename T_DeviceTimer,
           typename... T_ClientArgs
           >
  struct FFT : public T_FFT {
    /**
     * Called by BenchmarkExecutor
     * \tparam T_Result ResultBenchmark<NR_RUNS, NR_RECORDS>, also see class Application.
     * \tparam T_Vector
     * \tparam NDim Number of FFT dimensions
     */
    template<typename T_Result, typename T_Vector, size_t NDim>
    void operator()(T_Result& result,
                    T_Vector& vec,
                    const std::array<size_t,NDim>& extents
      ) const {
      using PrecisionT = typename Precision<typename T_Vector::value_type,
                                            T_FFT::IsComplex >::type;
      assert(vec.size());

      // prepare plan object
      // templates in: FFT type: in[,out][complex], PlanImpl, Precision, NDim
      auto fft = T_Client<T_FFT, PrecisionT, NDim, T_ClientArgs...> (extents);
      result.setValue(RecordType::DevBufferSize, fft.get_allocation_size());
      result.setValue(RecordType::DevPlanSize, fft.get_plan_size());
      result.setValue(RecordType::DevTransferSize, fft.get_transfer_size());

      TimerCPU tcpu_total;
      TimerCPU tcpu;
      T_DeviceTimer tdev;
      /// --- Total CPU ---
      tcpu_total.startTimer();

      // allocate memory
      tcpu.startTimer();
      fft.allocate();
      result.setValue(RecordType::Allocation, tcpu.stopTimer());

      // init forward plan
      tcpu.startTimer();
      fft.init_forward();
      result.setValue(RecordType::PlanInitFwd, tcpu.stopTimer());

      if(T_ReusePlan::value == false) {
        // init inverse plan
        tcpu.startTimer();
        fft.init_inverse();
        result.setValue(RecordType::PlanInitInv, tcpu.stopTimer());
      }

      // upload data
      tdev.startTimer();
      fft.upload(vec.data());
      result.setValue(RecordType::Upload, tdev.stopTimer());

      // execute forward transform
      tdev.startTimer();
      fft.execute_forward();
      result.setValue(RecordType::FFT, tdev.stopTimer());

      if(T_ReusePlan::value == true) {
        // init inverse plan
        tcpu.startTimer();
        fft.init_inverse();
        result.setValue(RecordType::PlanInitInv, tcpu.stopTimer());
      }

      // execute inverse transform
      tdev.startTimer();
      fft.execute_inverse();
      result.setValue(RecordType::FFTInv, tdev.stopTimer());

      // download data
      tdev.startTimer();
      fft.download(vec.data());
      result.setValue(RecordType::Download, tdev.stopTimer());

      /// --- Cleanup ---
      tcpu.startTimer();
      fft.destroy();
      result.setValue(RecordType::PlanDestroy, tcpu.stopTimer());

      result.setValue(RecordType::Total, tcpu_total.stopTimer());
    }
  };

}
#endif /* FFT_HPP_ */
