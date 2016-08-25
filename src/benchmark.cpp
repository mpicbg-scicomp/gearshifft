#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

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
using FFT_Is_Normalized = std::false_type;
#endif

// ----------------------------------------------------------------------------

/// functor for gearshifft benchmarks, TODO: get rid of this global variable!
gearshifft::Run<Context, FFT_Is_Normalized, FFTs, Precisions> instance;

/// global application handler as global fixture
using AppT = gearshifft::FixtureApplication<Context>;
BOOST_GLOBAL_FIXTURE(AppT);

// /// called by Boost UTF
// boost::unit_test::test_suite* init_unit_test_suite( int argc, char* argv[] )
// {
//   // process extra command line arguments for gearshifft
//   //  if error or help is shown then do not run any benchmarks
//   if( gearshifft::Options::getInstance().process(argc, argv) == 0 )
//     return instance();
//   else
//     return nullptr;
// }

namespace utf = boost::unit_test;

bool init_function()
{

  //TODO: create instance here locally and update with options coming in through the function parameters
  
  utf::framework::master_test_suite().add( instance() );

  return true;
}

//____________________________________________________________________________//

int main( int argc, char* argv[] )
{
  
  std::vector<char*> vargv(argv, argv+argc);
  int parse_return_value = gearshifft::Options::getInstance().parse(vargv);
  
  if( parse_return_value == 0 ){

    //TODO: remove processed items here
    //      at best do: auto bound_init_function = std::bind(init_function, _1, opts);
    //      at best do: ::boost::unit_test::unit_test_main( &bound_init_function, vargc, vargv.data() );
    
    return ::boost::unit_test::unit_test_main( &init_function,
					       vargv.size(),
					       vargv.data() );
  }
  else if( parse_return_value  == 1){

    //TODO: can I call the utf help message directly?
    std::cout << "Detailed Benchmark ";
    return ::boost::unit_test::unit_test_main( &init_function,
					       vargv.size(),
					       vargv.data() );

      
  }
  else {
    return 1;
  }
}
