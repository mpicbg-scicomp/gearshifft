// test cases for development purposes
#define BOOST_TEST_MODULE TestContext

#include <boost/test/included/unit_test.hpp>

#ifdef DEV_TESTS
#if defined(OPENCL_ENABLED)
#include "clfft_helper.hpp"
using namespace gearshifft::ClFFT;
struct Fixt {
    cl_device_id device = 0;
    cl_context ctx = 0;

  Fixt() {
    cl_platform_id platform;
    cl_int err = CL_SUCCESS;
    findClDevice(CL_DEVICE_TYPE_CPU, &platform, &device);
    ctx = clCreateContext( NULL, 1, &device, NULL, NULL, &err );
    CHECK_CL(err);
  }
  ~Fixt() {
    CHECK_CL(clReleaseContext( ctx ));
    CHECK_CL(clReleaseDevice(device));
  }
};
/**
 * if segfaults in the end, check if you are on a login node
 * instead of the compute node.
 */
BOOST_GLOBAL_FIXTURE(Fixt);
BOOST_AUTO_TEST_CASE( CL_Context_Global_Fixture )
{
  BOOST_CHECK( true );
}

#endif

#if defined(CUDA_ENABLED)
#include "cufft_helper.hpp"

BOOST_AUTO_TEST_CASE( CUDA_Large_MemAlloc_Test )
{
  float* data1 = nullptr;
  float* data2 = nullptr;
  size_t size = 1024*512*512*4;
  for(size_t k=0; k<4; ++k) {
    try {
      CHECK_CUDA(cudaMalloc(&data1, size));
      if(!data1) throw std::runtime_error("data1 failed.");
      CHECK_CUDA(cudaMalloc(&data2, size));
      if(!data2) throw std::runtime_error("data2 failed.");
      CHECK_CUDA(cudaFree(data1));
      CHECK_CUDA(cudaFree(data2));
    }catch(const std::runtime_error& e){
      std::cerr << "2x "<< size/1048576 << " MiB -- failed" << std::endl;
      BOOST_FAIL(e.what());
    }
    std::cout << "2x " << size/1048576 << " MiB -- ok" << std::endl;
    size<<=1;
  }
  BOOST_CHECK( true );
}

BOOST_AUTO_TEST_CASE( CUDA_KernelLeak_FFT_Test )
{
  cufftDoubleComplex* data;
  cufftDoubleComplex* data_transform;
  cufftHandle plan;
  CHECK_CUDA( cudaSetDevice(0) );
  std::vector< std::array<size_t, 1> > vec_extents;
  std::array<size_t, 1> e0 = {{262144}};   // works
  std::array<size_t, 1> e1 = {{67108863}}; // out of memory at kernel call (lmem)
  std::array<size_t, 1> e2 = {{262144}};   // still out of memory, even after reset
  vec_extents.push_back( e0 );
  vec_extents.push_back( e1 );
  vec_extents.push_back( e2 );

  for( auto extents : vec_extents ) {
    size_t data_size = extents[0]*sizeof(cufftDoubleComplex);
    size_t data_transform_size = extents[0]*sizeof(cufftDoubleComplex);
    size_t s = 0;

    try {
      CHECK_CUDA( cudaMalloc(&data, data_size));
      CHECK_CUDA( cudaMalloc(&data_transform, data_transform_size));

      size_t mem_free=0, mem_tot=0;
      CHECK_CUDA( cudaMemGetInfo(&mem_free, &mem_tot) );
      std::cerr << mem_free/1048576 << " MiB, ";

      CHECK_CUDA( cufftPlan1d(&plan, extents[0], CUFFT_Z2Z, 1)); // fails for 2^26-1
      CHECK_CUDA( cudaMemGetInfo(&mem_free, &mem_tot) );
      std::cerr << mem_free/1048576 << " MiB" << std::endl;

      CHECK_CUDA( cufftExecZ2Z(plan, data, data_transform, CUFFT_FORWARD));
      CHECK_CUDA( cufftExecZ2Z(plan, data_transform, data, CUFFT_INVERSE));

      CHECK_CUDA( cudaFree(data) );
      CHECK_CUDA( cudaFree(data_transform) );
      CHECK_CUDA( cufftDestroy(plan) );
      std::cerr << "Success: nx="<<extents[0]<<std::endl;
    }catch(const std::runtime_error& e){
      std::cerr << "Error for nx="<<extents[0]<<": "<<e.what() << std::endl;
      // cleanup #1
      CHECK_CUDA( cudaFree(data) );
      CHECK_CUDA( cudaFree(data_transform) );
      if(plan) {
        CHECK_CUDA( cufftDestroy(plan) );
        // cleanup #2
        //CHECK_CUDA( cudaDeviceReset() ); // does not help
        //CHECK_CUDA( cudaSetDevice(0) );

        // something different #1
        // < this later leads to CUFFT_SETUP_FAILED at cufftExecZ2Z()
        //CHECK_CUDA( cufftPlan2d(&plan, 4, 4, CUFFT_R2C));
        //CHECK_CUDA( cufftDestroy(plan) );
        // >
        // something different #2, also fails to recover cufft context
        //CHECK_CUDA( cufftCreate(&plan) );
        //CHECK_CUDA( cufftGetSize2d(plan, 4, 4, CUFFT_R2C, &s) );
        //CHECK_CUDA( cufftDestroy(plan) );
      }
    }
  }
  CHECK_CUDA( cudaDeviceReset() );
}

#endif // cuda

#endif // ifdef DEV_TESTS
