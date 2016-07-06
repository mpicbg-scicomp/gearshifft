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
#endif
