#ifndef FFT_HPP_
#define FFT_HPP_

#include "timer_cpu.hpp"
#include "traits.hpp"
#include "types.hpp"

#include <array>
#include <assert.h>
#include <memory>
#include <numeric>
#include <ostream>
#include <type_traits>

#ifdef GEARSHIFFT_SCOREP_INSTRUMENTATION
#include "scorep/SCOREP_User.h"
#else
#define SCOREP_USER_REGION(...)
#endif

#ifdef GEARSHIFFT_FLUSH_CACHE
#define FLUSH(...) flush()
#else
#define FLUSH(...)
#endif

#ifndef GEARSHIFFT_FLUSH_LLC_SIZE_MEBIBYTES
#define GEARSHIFFT_FLUSH_LLC_SIZE_MEBIBYTES 32
#endif

#ifndef GEARSHIFFT_FLUSH_CL_SIZE_BYTES
#define GEARSHIFFT_FLUSH_CL_SIZE_BYTES 64
#endif

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
           template <typename,typename,size_t,typename... > class T_Client, // user implementation FFT client
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
      SCOREP_USER_REGION("fft_benchmark", SCOREP_USER_REGION_TYPE_FUNCTION)

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

      {
        FLUSH();
        SCOREP_USER_REGION("plan_forward", SCOREP_USER_REGION_TYPE_DYNAMIC)
        // init forward plan
        tcpu.startTimer();
        fft.init_forward();
        result.setValue(RecordType::PlanInitFwd, tcpu.stopTimer());
      }

      if(!T_ReusePlan::value) {
        FLUSH();
        SCOREP_USER_REGION("plan_backward_no_reuse", SCOREP_USER_REGION_TYPE_DYNAMIC)
        // init inverse plan
        tcpu.startTimer();
        fft.init_inverse();
        result.setValue(RecordType::PlanInitInv, tcpu.stopTimer());
      }

      // upload data
      tdev.startTimer();
      fft.upload(vec.data());
      result.setValue(RecordType::Upload, tdev.stopTimer());

      {
        FLUSH();
        SCOREP_USER_REGION("transform_forward", SCOREP_USER_REGION_TYPE_DYNAMIC)
        // execute forward transform
        tdev.startTimer();
        fft.execute_forward();
        result.setValue(RecordType::FFT, tdev.stopTimer());
      }

      if(T_ReusePlan::value) {
        FLUSH();
        SCOREP_USER_REGION("plan_backward_reuse", SCOREP_USER_REGION_TYPE_DYNAMIC)
        // init inverse plan
        tcpu.startTimer();
        fft.init_inverse();
        result.setValue(RecordType::PlanInitInv, tcpu.stopTimer());
      }

      {
        FLUSH();
        SCOREP_USER_REGION("transform_backward", SCOREP_USER_REGION_TYPE_DYNAMIC)
        // execute inverse transform
        tdev.startTimer();
        fft.execute_inverse();
        result.setValue(RecordType::FFTInv, tdev.stopTimer());
      }

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

    static constexpr std::size_t flushBufferSize = 4 * GEARSHIFFT_FLUSH_LLC_SIZE_MEBIBYTES * (1 << 20);
    static constexpr std::size_t flushStride     =     GEARSHIFFT_FLUSH_CL_SIZE_BYTES;
    static std::unique_ptr<volatile char[]> flushBuffer;  // no std::byte in C++14
    static volatile char flushSink;

    static void flush() {
      std::iota(flushBuffer.get(), flushBuffer.get() + flushBufferSize, 0x0);
      for (std::size_t i = 0; i < flushBufferSize; i += flushStride) {
        flushSink += flushBuffer[i];
      }
    }
  };

#ifdef GEARSHIFFT_FLUSH_CACHE
  template<typename T_FFT,
           typename T_ReusePlan,
           template <typename,typename,size_t,typename... > class T_Client,
           typename T_DeviceTimer,
           typename... T_ClientArgs>
  std::unique_ptr<volatile char[]>
  FFT<T_FFT, T_ReusePlan, T_Client, T_DeviceTimer, T_ClientArgs...>::flushBuffer =
      std::make_unique<volatile char[]>(flushBufferSize);

  template<typename T_FFT,
           typename T_ReusePlan,
           template <typename,typename,size_t,typename... > class T_Client,
           typename T_DeviceTimer,
           typename... T_ClientArgs>
  volatile char FFT<T_FFT, T_ReusePlan, T_Client, T_DeviceTimer, T_ClientArgs...>::flushSink = 'X';
#endif

}
#endif /* FFT_HPP_ */
