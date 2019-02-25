#define BOOST_TEST_MODULE TestrocFFTHelper
#include "libraries/rocfft/rocfft_helper.hpp"

#include <boost/test/included/unit_test.hpp> // Single-header usage variant

#include <iostream>
#include <vector>
#include <cstdint>


BOOST_AUTO_TEST_CASE( listDevices, * boost::unit_test::tolerance(0.0001f) )
{
    std::stringstream sstr("");
    gearshifft::RocFFT::listHipDevices(sstr);
    BOOST_CHECK_GT(sstr.str().size(),0);
}
