#ifndef FFT_ABSTRACT_HPP_
#define FFT_ABSTRACT_HPP_

#include "timer.hpp"
#include <assert.h>
#include <array>
#include <type_traits>
#include <ostream>

namespace gearshifft
{

  struct FFT_Inplace_Real
  {
    static constexpr auto IsComplex = false;
    static constexpr auto Title = "Inplace_Real";
    static constexpr auto IsInplace = true;
  };
  struct  FFT_Outplace_Real{
    static constexpr auto IsComplex = false;
    static constexpr auto Title = "Outplace_Real";
    static constexpr auto IsInplace = false;
  };

  struct FFT_Inplace_Complex
  {
    static constexpr auto IsComplex = true;
    static constexpr auto Title = "Inplace_Complex";
    static constexpr auto IsInplace = true;
  };

  struct FFT_Outplace_Complex
  {
    static constexpr auto IsComplex = true;
    static constexpr auto Title = "Outplace_Complex";
    static constexpr auto IsInplace = false;
  };

/**
 * Trait to get precision type of Real or Complex data type.
 */
  template<typename T, bool IsComplex>
  struct Precision;
  template<typename T>
  struct Precision<T, true> { using type = typename T::type; };
  template<typename T>
  struct Precision<T, false> { using type = T; };

  enum struct RecordType{
    Device = 0,
    Allocation,
    PlanInit,
    Upload,
    FFT,
    FFTInverse,
    Download,
    PlanDestroy,
    Total,
    DevBufferSize,
    DevPlanSize,
    _NrRecords
  };
  std::ostream& operator<< (std::ostream & os, RecordType r)
  {
    switch (r)
    {
    case RecordType::Device: return os << "Time_Device [ms]";
    case RecordType::Allocation: return os << "Time_Allocation [ms]";
    case RecordType::PlanInit: return os << "Time_PlanInit [ms]";
    case RecordType::Upload: return os << "Time_Upload [ms]";
    case RecordType::FFT: return os << "Time_FFT [ms]";
    case RecordType::FFTInverse: return os << "Time_iFFT [ms]";
    case RecordType::Download: return os << "Time_Download [ms]";
    case RecordType::PlanDestroy: return os << "Time_PlanDestroy [ms]";
    case RecordType::Total: return os << "Time_Total [ms]";
    case RecordType::DevBufferSize: return os << "Size_DeviceBuffer [bytes]";
    case RecordType::DevPlanSize: return os << "Size_DevicePlan [bytes]";
    };
    return os << static_cast<int>(r);
  }

/**
 * Functor being called from FixtureBenchmark::benchmark()
 */
  template<typename TFFT, // FFT_*_* [inplace.., real..]
           template <typename,typename,size_t> typename TPlan,
           typename TDeviceTimer
           >
  struct FFT : public TFFT
  {
    template<typename T_Result, typename T_Vector, size_t NDim>
    void operator()(T_Result& result,
                    T_Vector& vec,
                    const std::array<unsigned,NDim>& extents
      )
      {
        using PrecisionT = typename Precision<typename T_Vector::value_type,
                                              TFFT::IsComplex >::type;
        assert(vec.data());

        // prepare plan object
        // templates in: FFT type: in[,out][complex], PlanImpl, Precision, NDim
        auto plan = TPlan<TFFT, PrecisionT, NDim> (extents);
        result.setValue(RecordType::DevBufferSize, plan.getAllocSize());
        result.setValue(RecordType::DevPlanSize, plan.getPlanSize());

        TimerCPU ttotal;
        TimerCPU talloc;
        TimerCPU tplaninit;
        TimerCPU tplandestroy;
        TDeviceTimer tdevupload;
        TDeviceTimer tdevdownload;
        TDeviceTimer tdevfft;
        TDeviceTimer tdevfftinverse;
        TDeviceTimer tdevtotal;
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
#endif /* FFT_ABSTRACT_HPP_ */
