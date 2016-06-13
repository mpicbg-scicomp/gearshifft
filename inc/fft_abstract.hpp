#ifndef FFT_ABSTRACT_HPP_
#define FFT_ABSTRACT_HPP_

#include "timer.hpp"
#include <type_traits>

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

  enum struct Time {
    Device = 0,
    Allocation,
    Upload,
    FFT,
    FFTInverse,
    Download,
    Cleanup,
    Total,
    _Times
  };

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
        result.addSizeDeviceDataBuffer(plan.getAllocSize());
        result.addSizeDevicePlanBuffer(plan.getPlanSize());

        TimerCPU ttotal;
        TimerCPU talloc;
        TimerCPU tcleanup;
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
        result.addValue(Time::Allocation, talloc.stopTimer());

        /// --- Create plan ---
        plan.init_forward();

        /// --- FFT+iFFT GPU ---
        tdevtotal.startTimer();

        tdevupload.startTimer();
         plan.upload(vec.data());
        result.addValue(Time::Upload, tdevupload.stopTimer());

        tdevfft.startTimer();
         plan.execute_forward();
        result.addValue(Time::FFT, tdevfft.stopTimer());

        plan.init_backward();

        tdevfftinverse.startTimer();
         plan.execute_backward();
        result.addValue(Time::FFTInverse, tdevfftinverse.stopTimer());

        tdevdownload.startTimer();
         plan.download(vec.data());
        result.addValue(Time::Download, tdevdownload.stopTimer());

        result.addValue(Time::Device, tdevtotal.stopTimer());


        /// --- Cleanup ---
        tcleanup.startTimer();
         plan.destroy();
        result.addValue(Time::Cleanup, tcleanup.stopTimer());

        result.addValue(Time::Total, ttotal.stopTimer());
      }
  };

}
#endif /* FFT_ABSTRACT_HPP_ */
