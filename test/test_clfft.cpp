#define BOOST_TEST_MODULE TestClFFT

#include "libraries/clfft/clfft_helper.hpp"
#include <boost/test/unit_test.hpp>

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
