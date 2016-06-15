#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Fixtures

#include "application.hpp"
#include "fixture_benchmark.hpp"


#ifdef CUDA_ENABLED
//@todo measure init and reset time
#include "cufft.hpp"

// -- Execute Benchmarks --
RUN_BENCHMARKS_UNNORMALIZED_FFT(CuFFT,
                                gearshifft::CuFFT::Context,
                                gearshifft::CuFFT::Inplace_Real,
                                gearshifft::CuFFT::Outplace_Real,
                                gearshifft::CuFFT::Inplace_Complex,
                                gearshifft::CuFFT::Outplace_Complex)

#endif

#ifdef OPENCL_ENABLED
#include "clfft.hpp"

// -- Execute Benchmarks --
RUN_BENCHMARKS_NORMALIZED_FFT(ClFFT,
                              gearshifft::ClFFT::Context,
                              gearshifft::ClFFT::Inplace_Real,
                              gearshifft::ClFFT::Outplace_Real,
                              gearshifft::ClFFT::Inplace_Complex,
                              gearshifft::ClFFT::Outplace_Complex
                              )

#endif

