#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Fixtures

#include "fixture_benchmark.hpp"

#ifdef CUDA_ENABLED
//@todo measure init and reset time
#include "cufft.hpp"

// -- Execute Benchmarks --
RUN_BENCHMARKS_UNNORMALIZED_FFT(CuFFT,
                                gearshifft::CuFFT::Inplace_Real,
                                gearshifft::CuFFT::Outplace_Real,
                                gearshifft::CuFFT::Inplace_Complex,
                                gearshifft::CuFFT::Outplace_Complex)

#endif

#ifdef OPENCL_ENABLED
#include "clfft.hpp"
using namespace gearshifft::helper;
/**
 * Global setup for context (application lifetime).
 */
struct MyConfig {
  Statistics stats;
  std::stringstream ss;
  MyConfig()   {
    TimeStatistics<TimerCPU> timer(&stats);
    int i_create = timer.add("Init OpenCL+clFFT");
    timer.start(i_create);
    gearshifft::ClFFT::context.create();
    timer.stop(i_create);
  }
  ~MyConfig()  {
    TimeStatistics<TimerCPU> timer(&stats);
    int i_teardown = timer.add("Teardown OpenCL+clFFT");
    timer.start(i_teardown);
    gearshifft::ClFFT::context.destroy();
    timer.stop(i_teardown);
    // information on OpenCL device
    ss << gearshifft::ClFFT::getClDeviceInformations(
      gearshifft::ClFFT::context.device ).str()
       << std::endl;
    ss << stats;
    gearshifft::Output::write(ss.str());
  }
};
BOOST_GLOBAL_FIXTURE( MyConfig );
// -- Execute Benchmarks --
RUN_BENCHMARKS_NORMALIZED_FFT(ClFFT,
                              gearshifft::ClFFT::Inplace_Real,
                              gearshifft::ClFFT::Outplace_Real,
                              gearshifft::ClFFT::Inplace_Complex,
                              gearshifft::ClFFT::Outplace_Complex
                              )

#endif

