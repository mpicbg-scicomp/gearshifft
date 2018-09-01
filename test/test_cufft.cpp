#define BOOST_TEST_MODULE TestCuFFT

#include "libraries/cufft/cufft_helper.hpp"
#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace gearshifft::CuFFT;

BOOST_AUTO_TEST_CASE( Device )
{
  int nr=0;

  try {
    CHECK_CUDA( cudaGetDeviceCount(&nr) );
    for( auto i=0; i<nr; ++i ) {
      CHECK_CUDA( cudaSetDevice(i) );
      CHECK_CUDA( cudaDeviceReset() );
    }
  }catch(const std::runtime_error& e){
    BOOST_FAIL("CUDA Error: " << e.what());
  }
  BOOST_CHECK( true );
}

BOOST_AUTO_TEST_CASE( CuFFT_Single )
{
  cufftComplex* data = nullptr;
  cufftComplex* data_transform = nullptr;
  cufftHandle plan = 0;
  CHECK_CUDA( cudaSetDevice(0) );
  std::vector< std::array<size_t, 1> > vec_extents;
  std::array<size_t, 1> e0 = {{128}};
  std::array<size_t, 1> e1 = {{125}};
  std::array<size_t, 1> e2 = {{169}};
  std::array<size_t, 1> e3 = {{170}};
  vec_extents.push_back( e0 );
  vec_extents.push_back( e1 );
  vec_extents.push_back( e2 );
  vec_extents.push_back( e3 );

  for( auto extents : vec_extents ) {
    size_t data_size = extents[0]*sizeof(cufftComplex);
    size_t data_transform_size = extents[0]*sizeof(cufftComplex);

    try {
      CHECK_CUDA( cudaMalloc(&data, data_size));
      CHECK_CUDA( cudaMalloc(&data_transform, data_transform_size));

      CHECK_CUDA( cufftPlan1d(&plan, extents[0], CUFFT_C2C, 1));

      CHECK_CUDA( cufftExecC2C(plan, data, data_transform, CUFFT_FORWARD));
      CHECK_CUDA( cufftExecC2C(plan, data_transform, data, CUFFT_INVERSE));

      CHECK_CUDA( cudaFree(data) );
      CHECK_CUDA( cudaFree(data_transform) );
      CHECK_CUDA( cufftDestroy(plan) );
      plan = 0;
    }catch(std::runtime_error& e){
      // cleanup #1
      CHECK_CUDA( cudaFree(data) );
      CHECK_CUDA( cudaFree(data_transform) );
      if(plan) {
        CHECK_CUDA( cufftDestroy(plan) );
        plan=0;
      }
      BOOST_ERROR("Error for nx="<<extents[0]<<": "<<e.what());
    }
  }
  CHECK_CUDA( cudaDeviceReset() );
  BOOST_CHECK( true );

}

BOOST_AUTO_TEST_CASE( CuFFT_Double )
{
  cufftDoubleComplex* data = nullptr;
  cufftDoubleComplex* data_transform = nullptr;
  cufftHandle plan = 0;
  CHECK_CUDA( cudaSetDevice(0) );
  std::vector< std::array<size_t, 1> > vec_extents;
  std::array<size_t, 1> e0 = {{128}};
  std::array<size_t, 1> e1 = {{125}};
  std::array<size_t, 1> e2 = {{169}};
  std::array<size_t, 1> e3 = {{170}};
  vec_extents.push_back( e0 );
  vec_extents.push_back( e1 );
  vec_extents.push_back( e2 );
  vec_extents.push_back( e3 );

  for( auto extents : vec_extents ) {
    size_t data_size = extents[0]*sizeof(cufftDoubleComplex);
    size_t data_transform_size = extents[0]*sizeof(cufftDoubleComplex);

    try {
      CHECK_CUDA( cudaMalloc(&data, data_size));
      CHECK_CUDA( cudaMalloc(&data_transform, data_transform_size));

      CHECK_CUDA( cufftPlan1d(&plan, extents[0], CUFFT_Z2Z, 1));

      CHECK_CUDA( cufftExecZ2Z(plan, data, data_transform, CUFFT_FORWARD));
      CHECK_CUDA( cufftExecZ2Z(plan, data_transform, data, CUFFT_INVERSE));

      CHECK_CUDA( cudaFree(data) );
      CHECK_CUDA( cudaFree(data_transform) );
      CHECK_CUDA( cufftDestroy(plan) );
      plan = 0;
    }catch(std::runtime_error& e){
      // cleanup #1
      CHECK_CUDA( cudaFree(data) );
      CHECK_CUDA( cudaFree(data_transform) );
      if(plan) {
        CHECK_CUDA( cufftDestroy(plan) );
        plan=0;
      }
      BOOST_ERROR("Error for nx="<<extents[0]<<": "<<e.what());
    }
  }
  CHECK_CUDA( cudaDeviceReset() );
  BOOST_CHECK( true );
}
