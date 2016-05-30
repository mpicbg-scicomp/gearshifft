#ifndef FFT_ABSTRACT_HPP_
#define FFT_ABSTRACT_HPP_

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

/**
 * Functor being called from FixtureBenchmark::benchmark()
 */
  template<typename TFFT, // FFT_*_* [inplace.., real..]
           template <typename,typename,size_t> typename TPlan,
           typename TDeviceTimer
           >
  struct FFT : public TFFT
  {
    template<typename TVector, size_t NDim>
    void operator()(TVector& vec,
                    const std::array<unsigned,NDim>& cextents,
                    Results& results)
      {
        using TPrecision = typename Precision<typename TVector::value_type,
                                              TFFT::IsComplex >::type;
        assert(vec.data());
        Statistics& stats = results.stats;

        TimeStatistics<TDeviceTimer> timer_dev(&stats); // or OpenCL timer
        TimeStatistics<TimerCPU> timer_cpu(&stats);
        int i_gpu = timer_dev.append("Device Runtime");
        int i_cpu_alloc = timer_cpu.append("CPU Alloc");
        int i_gpu_upload = timer_dev.append("Device Upload");
        int i_gpu_fft = timer_dev.append("Device FFT");
        int i_gpu_ifft = timer_dev.append("Device iFFT");
        int i_gpu_download = timer_dev.append("Device Download");
        int i_cpu_cleanup = timer_cpu.append("CPU Cleanup");
        int i_cpu = timer_cpu.append("CPU Runtime");

        // prepare plan object
        // templates in: FFT type: in[,out][complex], PlanImpl, Precision, NDim
        auto plan = TPlan<TFFT, TPrecision, NDim> (cextents);
        results.alloc_mem_in_bytes = plan.getAllocSize();
        results.plan_mem_in_bytes  = plan.getPlanSize();

        /// --- Total CPU ---
        timer_cpu.start(i_cpu);
        /// --- Malloc ---
        timer_cpu.start(i_cpu_alloc);
        plan.malloc();
        timer_cpu.stop(i_cpu_alloc);

        /// --- Create plan ---
        plan.init_forward();

        /// --- FFT+iFFT GPU ---
        timer_dev.start(i_gpu);
        timer_dev.start(i_gpu_upload);
        plan.upload(vec.data());
        timer_dev.stop(i_gpu_upload);

        timer_dev.start(i_gpu_fft);
        plan.execute_forward();
        timer_dev.stop(i_gpu_fft);

        plan.init_backward();

        timer_dev.start(i_gpu_ifft);
        plan.execute_backward();
        timer_dev.stop(i_gpu_ifft);

        timer_dev.start(i_gpu_download);
        plan.download(vec.data());
        timer_dev.stop(i_gpu_download);
        timer_dev.stop(i_gpu);


        /// --- Cleanup ---
        timer_cpu.start(i_cpu_cleanup);
        plan.destroy();
        timer_cpu.stop(i_cpu_cleanup);

        timer_cpu.stop(i_cpu);
      }
  };

}
#endif /* FFT_ABSTRACT_HPP_ */
