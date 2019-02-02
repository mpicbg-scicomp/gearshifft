#define BOOST_TEST_MODULE TestClFFT

#include "core/types.hpp"
#include "libraries/clfft/clfft_helper.hpp"

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

using test_types = boost::mpl::list<float, double>;

template< typename TPrecision=float >
struct FFTPrecision: std::integral_constant< clfftPrecision, CLFFT_SINGLE >{};
template<>
struct FFTPrecision<double>: std::integral_constant< clfftPrecision, CLFFT_DOUBLE >{};


using namespace gearshifft::ClFFT;


BOOST_AUTO_TEST_CASE( CL_Context_CPU )
{
  cl_device_id device = 0;
  cl_context ctx = 0;
  cl_platform_id platform=0;
  cl_int err = CL_SUCCESS;
  try{
    findClDevice(CL_DEVICE_TYPE_CPU, &platform, &device);
    ctx = clCreateContext( NULL, 1, &device, NULL, NULL, &err );
    CHECK_CL(err);
    BOOST_REQUIRE( err==CL_SUCCESS );
    CHECK_CL(clReleaseContext( ctx ));
    CHECK_CL(clReleaseDevice(device));
  }catch(std::runtime_error& e){
    BOOST_REQUIRE_MESSAGE(err == CL_SUCCESS, "failed with error: " << e.what());
  }
}

BOOST_AUTO_TEST_CASE( CL_Context_GPU )
{
  cl_device_id device = 0;
  cl_context ctx = 0;
  cl_platform_id platform=0;
  cl_int err = CL_SUCCESS;
  try{
    findClDevice(CL_DEVICE_TYPE_GPU, &platform, &device);
    ctx = clCreateContext( NULL, 1, &device, NULL, NULL, &err );
    CHECK_CL(err);
    BOOST_REQUIRE( err==CL_SUCCESS );
    CHECK_CL(clReleaseContext( ctx ));
    CHECK_CL(clReleaseDevice(device));
  }catch(std::runtime_error& e){
    BOOST_REQUIRE_MESSAGE(err == CL_SUCCESS, "failed with error: " << e.what());
  }
}

BOOST_AUTO_TEST_CASE( CLFFT_2D_C2C )
{
  cl_int err = CL_SUCCESS;
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
  cl_context ctx = 0;
  cl_command_queue queue = 0;
  cl_mem bufX;
  float *X;

  const size_t N0 = 1024, N1 = 8192;
  char platform_name[128];
  char device_name[128];

  /* FFT library realted declarations */
  clfftPlanHandle planHandle;
  clfftDim dim = CLFFT_2D;
  size_t clLengths[2] = {N0, N1};

  /* Setup OpenCL environment. */
  err |= clGetPlatformIDs( 1, &platform, NULL );

  size_t ret_param_size = 0;
  err |= clGetPlatformInfo(platform, CL_PLATFORM_NAME,
                          sizeof(platform_name), platform_name,
                          &ret_param_size);

  err |= clGetDeviceIDs( platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL );

  err |= clGetDeviceInfo(device, CL_DEVICE_NAME,
                        sizeof(device_name), device_name,
                        &ret_param_size);

  props[1] = reinterpret_cast<cl_context_properties>(platform);
  ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
  queue = clCreateCommandQueue( ctx, device, 0, &err );

  clfftSetupData fftSetup;
  err |= clfftInitSetupData(&fftSetup);
  err |= clfftSetup(&fftSetup);

  size_t buffer_size  = N0 * N1 * 2 * sizeof(float);
  X = new float[N0 * N1 * 2];

  size_t i, j;
  i = j = 0;
  for (i=0; i<N0; ++i) {
    for (j=0; j<N1; ++j) {
      size_t idx = 2*(j+i*N1);
      X[idx] = 0.5f;
      X[idx+1] = 0.5f;
    }
  }

  bufX = clCreateBuffer( ctx, CL_MEM_READ_WRITE, buffer_size, NULL, &err );
  err |= clEnqueueWriteBuffer( queue, bufX, CL_TRUE, 0, buffer_size, X, 0, NULL, NULL );

  err |= clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);
  err |= clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
  err |= clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
  err |= clfftSetResultLocation(planHandle, CLFFT_INPLACE);
  err |= clfftBakePlan(planHandle, 1, &queue, NULL, NULL);

  err |= clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &bufX, NULL, NULL);
  err |= clFinish(queue);

  err |= clEnqueueReadBuffer( queue, bufX, CL_TRUE, 0, buffer_size, X, 0, NULL, NULL );

  clReleaseMemObject( bufX );
  delete[] X;

  err |= clfftDestroyPlan( &planHandle );

  clfftTeardown( );

  clReleaseCommandQueue( queue );
  clReleaseContext( ctx );
  BOOST_CHECK( err == CL_SUCCESS );
}


BOOST_AUTO_TEST_CASE_TEMPLATE( CLFFT_1D_R2C_INPLACE, T, test_types )
{
  cl_int err = CL_SUCCESS;
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_context ctx = 0;
  cl_command_queue queue = 0;
  cl_mem bufX;
  T *X = NULL;

  size_t N = 2048;

  clfftPlanHandle planHandle;
  clfftDim dim = CLFFT_1D;

  size_t clLengths[1] = {N};
  size_t strides[3] = {1};
  size_t strides_complex[3] = {1};
  size_t dist;
  size_t dist_complex = 1;

  size_t bytes_r = N * sizeof(T);
  size_t bytes_c = (N/2+1) * 2 * sizeof(T);

  findClDevice(CL_DEVICE_TYPE_DEFAULT, &platform, &device);
  ctx = clCreateContext( NULL, 1, &device, NULL, NULL, &err );
  CHECK_CL( err );
  queue = clCreateCommandQueue( ctx, device, 0, &err );
  CHECK_CL( err );

  clfftSetupData fftSetup;
  CHECK_CL( clfftInitSetupData(&fftSetup) );
  CHECK_CL( clfftSetup(&fftSetup) );

  dist = 2 * (N/2 + 1);
  dist_complex = N/2+1;

  X = new T[dist];
  for(size_t k=0; k<dist; ++k)
    X[k] = 1.0;


  bufX = clCreateBuffer( ctx, CL_MEM_READ_WRITE, bytes_c, NULL, &err );
  CHECK_CL( err );
  CHECK_CL( clEnqueueWriteBuffer( queue, bufX, CL_TRUE, 0,
                                  bytes_r, X, 0, NULL, NULL ) );

  CHECK_CL( clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths) );

  CHECK_CL( clfftSetPlanPrecision(planHandle, FFTPrecision<T>::value) );
  CHECK_CL( clfftSetLayout(planHandle, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED) );
  CHECK_CL( clfftSetResultLocation(planHandle, CLFFT_INPLACE) );

  CHECK_CL(clfftSetPlanInStride(planHandle, dim, strides));
  CHECK_CL(clfftSetPlanOutStride(planHandle, dim, strides_complex));
  CHECK_CL(clfftSetPlanDistance(planHandle, dist, dist_complex));

  CHECK_CL( clfftBakePlan(planHandle, 1, &queue, NULL, NULL) );

  CHECK_CL( clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &bufX, NULL, NULL) );
  CHECK_CL( clFinish(queue) );

  CHECK_CL( clfftDestroyPlan( &planHandle ) );

  CHECK_CL( clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths) );
  CHECK_CL( clfftSetPlanPrecision(planHandle, FFTPrecision<T>::value) );
  CHECK_CL( clfftSetLayout(planHandle, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL) );
  CHECK_CL( clfftSetResultLocation(planHandle, CLFFT_INPLACE) );
  CHECK_CL(clfftSetPlanInStride(planHandle, dim, strides_complex));
  CHECK_CL(clfftSetPlanOutStride(planHandle, dim, strides));
  CHECK_CL(clfftSetPlanDistance(planHandle, dist_complex, dist));
  CHECK_CL( clfftBakePlan(planHandle, 1, &queue, NULL, NULL) );

  CHECK_CL( clfftEnqueueTransform(planHandle, CLFFT_BACKWARD, 1, &queue, 0, NULL, NULL, &bufX, NULL, NULL) );
  CHECK_CL( clFinish(queue) );

  CHECK_CL( clEnqueueReadBuffer( queue, bufX, CL_TRUE, 0, bytes_r, X, 0, NULL, NULL ) );

  CHECK_CL( clReleaseMemObject( bufX ) );

  //CHECK_CL( clfftDestroyPlan( &planHandle ) );
  CHECK_CL( clfftDestroyPlan( &planHandle ) );
  CHECK_CL( clfftTeardown() );
  CHECK_CL( clReleaseCommandQueue( queue ) );
  CHECK_CL( clReleaseContext( ctx ) );

  auto error_bound = gearshifft::ErrorBound<T>()();
  for(size_t k=0; k<dist; ++k)
    BOOST_TEST(X[k] == 1.0, boost::test_tools::tolerance(error_bound));

  delete[] X;
}
