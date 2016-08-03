#include "benchmark_suite.hpp"
#include "application.hpp"
#include "options.hpp"

#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>

/// List alias
template<typename... Types>
using List = boost::mpl::list<Types...>;

// ----------------------------------------------------------------------------

#ifdef CUDA_ENABLED
#include "cufft.hpp"
using namespace gearshifft::CuFFT;
using FFTs              = List<Inplace_Real,
                               Inplace_Complex,
                               Outplace_Real,
                               Outplace_Complex>;
using Precisions        = List<float, double>;
using FFT_Is_Normalized = std::false_type;

#elif defined(OPENCL_ENABLED)
#include "clfft.hpp"
using namespace gearshifft::ClFFT;
using FFTs              = List<Inplace_Real,
                               Inplace_Complex,
                               Outplace_Real,
                               Outplace_Complex>;
using Precisions        = List<float, double>;
using FFT_Is_Normalized = std::true_type;

#elif defined(FFTW_ENABLED)
#include "fftw.hpp"
using namespace gearshifft::fftw;
using FFTs              = List<Inplace_Real,
                               Inplace_Complex,
                               Outplace_Real,
                               Outplace_Complex>;
using Precisions        = List<float, double>;
using FFT_Is_Normalized = std::true_type;
#endif

// ----------------------------------------------------------------------------

/// functor for gearshifft benchmarks
gearshifft::Run<Context, FFT_Is_Normalized, FFTs, Precisions> instance;

/// global application handler as global fixture
using AppT = gearshifft::FixtureApplication<Context>;
BOOST_GLOBAL_FIXTURE(AppT);

/// called by Boost UTF
boost::unit_test::test_suite* init_unit_test_suite( int argc, char* argv[] )
{
  // process extra command line arguments for gearshifft
  //  if error or help is shown then do not run any benchmarks
  if( gearshifft::Options::getInstance().process(argc, argv) == 0 )
    return instance();
  else
    return nullptr;
}
