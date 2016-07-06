/// @todo
#include "fixture_test_suite.hpp"
#include "application.hpp"
#include <boost/test/included/unit_test.hpp>

#ifdef CUDA_ENABLED
#include "cufft.hpp"
using namespace gearshifft::CuFFT;
using FFTs       = gearshifft::List<Inplace_Real,
                                    Inplace_Complex,
                                    Outplace_Real,
                                    Outplace_Complex>;
using Precisions = gearshifft::List<float, double>;
using FFT_Is_Normalized = std::false_type;

#elif defined(OPENCL_ENABLED)
#include "clfft.hpp"
using namespace gearshifft::ClFFT;
using FFTs       = gearshifft::List<Inplace_Real,
                                    Inplace_Complex,
                                    Outplace_Real,
                                    Outplace_Complex>;
using Precisions = gearshifft::List<float, double>;
using FFT_Is_Normalized = std::true_type;
#endif

/**
 *
 */
gearshifft::Run<Context, FFT_Is_Normalized, FFTs, Precisions> instance;


using AppT = gearshifft::FixtureApplication<Context>;
BOOST_GLOBAL_FIXTURE(AppT);

using namespace boost::unit_test;
test_suite*
init_unit_test_suite( int argc, char* argv[] )
{
  return instance();
}
