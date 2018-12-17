#define BOOST_TEST_MODULE TestrocFFTHlper
#include <iostream>
#include <vector>
#include <cstdint>

#include "libraries/rocfft/rocfft_helper.hpp"

#include <boost/test/unit_test.hpp>
#include <iostream>

BOOST_AUTO_TEST_CASE( listDevices, * boost::unit_test::tolerance(0.0001f) )
{
    auto sstr = gearshifft::Rocfft::listHipDevices();
    BOOST_CHECK_GT(sstr.str().size(),0);
}
