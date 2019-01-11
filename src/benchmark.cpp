#include "core/benchmark.hpp"
#include "core/types.hpp"

// ----------------------------------------------------------------------------
template<typename... Types>
using List = gearshifft::List<Types...>;

#ifdef CUFFT_ENABLED
#include "libraries/cufft/cufft.hpp"

using namespace gearshifft::CuFFT;
using Context           = CuFFTContext;
using FFTs              = List<Inplace_Real,
                               Inplace_Complex,
                               Outplace_Real,
                               Outplace_Complex>;

using Precisions        = gearshifft::DefaultPrecisions;
using FFT_Is_Normalized = std::false_type;

#elif defined(CLFFT_ENABLED)
#include "libraries/clfft/clfft.hpp"

using namespace gearshifft::ClFFT;
using Context           = ClFFTContext;
using FFTs              = List<Inplace_Real,
                               Inplace_Complex,
                               Outplace_Real,
                               Outplace_Complex>;
using Precisions        = gearshifft::DefaultPrecisionsWithoutHalfPrecision;
using FFT_Is_Normalized = std::true_type;

#elif defined(FFTW_ENABLED)
#include "libraries/fftw/fftw.hpp"

using namespace gearshifft::fftw;
using Context           = FftwContext;
using FFTs              = List<Inplace_Real,
                               Inplace_Complex,
                               Outplace_Real,
                               Outplace_Complex >;
using Precisions        = gearshifft::DefaultPrecisionsWithoutHalfPrecision;
using FFT_Is_Normalized = std::false_type;
#endif

// ----------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
  int ret = 0;
  try {
    gearshifft::Benchmark<Context> benchmark;

    benchmark.configure(argc, argv);
    ret = benchmark.run<FFT_Is_Normalized, FFTs, Precisions>();

  }catch(const std::runtime_error& e){
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return ret;
}
