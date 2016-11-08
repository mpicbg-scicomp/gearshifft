#include "core/benchmark.hpp"

// ----------------------------------------------------------------------------
template<typename... Types>
using List = gearshifft::List<Types...>;

#ifdef CUDA_ENABLED
#include "libraries/cufft/cufft.hpp"
using namespace gearshifft::CuFFT;
using FFTs              = List<Inplace_Real,
                               Inplace_Complex,
                               Outplace_Real,
                               Outplace_Complex>;
using Precisions        = List<float, double>;
using FFT_Is_Normalized = std::false_type;

#elif defined(OPENCL_ENABLED)
#include "libraries/clfft/clfft.hpp"
using namespace gearshifft::ClFFT;
using FFTs              = List<Inplace_Real,
                               Inplace_Complex,
                               Outplace_Real,
                               Outplace_Complex>;
using Precisions        = List<float, double>;
using FFT_Is_Normalized = std::true_type;

#elif defined(FFTW_ENABLED)
#include "libraries/fftw/fftw.hpp"
using namespace gearshifft::fftw;
struct FftwConfig {
#if defined(GEARSHIFFT_FFTW_USE_ESTIMATE)
  static constexpr unsigned PlanFlags = FFTW_ESTIMATE;
#elif defined(GEARSHIFFT_FFTW_USE_WISDOM)
  static constexpr unsigned PlanFlags = FFTW_WISDOM_ONLY;
  static constexpr auto WisdomFileSinglePrecision = "../config/fftwf_wisdom.txt";
  static constexpr auto WisdomFileDoublePrecision = "../config/fftw_wisdom.txt";
#else
  static constexpr unsigned PlanFlags = FFTW_MEASURE;
#endif
#if defined(GEARSHIFFT_FFTW_MEASURE) && defined(GEARSHIFFT_FFTW_TIMELIMIT)
  static constexpr double PlanTimeLimit = GEARSHIFFT_FFTW_TIMELIMIT;
#else
  static constexpr double PlanTimeLimit = FFTW_NO_TIMELIMIT;
#endif
};
using Context           = FftwContext<FftwConfig>;
using FFTs              = List<Inplace_Real<FftwConfig>,
                               Inplace_Complex<FftwConfig>,
                               Outplace_Real<FftwConfig>,
                               Outplace_Complex<FftwConfig> >;
using Precisions        = List<float, double>;
using FFT_Is_Normalized = std::false_type;
#endif

// ----------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
  try {
    gearshifft::Benchmark<Context> benchmark;

    benchmark.configure(argc, argv);
    benchmark.run<FFT_Is_Normalized, FFTs, Precisions>();

  }catch(const std::runtime_error& e){
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
