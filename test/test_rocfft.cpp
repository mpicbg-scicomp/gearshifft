#define BOOST_TEST_MODULE TestrocFFT
#include <iostream>
#include <vector>
#include <cstdint>
#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "rocfft.h"
#include <boost/test/unit_test.hpp>
#include <iostream>

BOOST_AUTO_TEST_CASE( FFT1DSingle, * boost::unit_test::tolerance(0.0001f) )
{
  // rocFFT gpu compute
        // ========================================

        rocfft_setup();

        std::size_t N = 32 << 10;
        std::size_t Nbytes = N * sizeof(float2);

        // Create HIP device buffer
        float2 *x;
        hipMalloc(&x, Nbytes);

        // Initialize data
        std::vector<float2> cx(N);
        for (std::size_t i = 0; i < N; i++)
        {
                cx[i].x = 1;
                cx[i].y = -1;
        }

        //  Copy data to device
        hipMemcpy(x, cx.data(), Nbytes, hipMemcpyHostToDevice);

        // Create fwd rocFFT fwd
        rocfft_plan fwd = NULL;
        std::size_t length = N;
        rocfft_plan_create(&fwd,
                           rocfft_placement_inplace,
                           rocfft_transform_type_complex_forward,
                           rocfft_precision_single,
                           1, &length, 1, NULL);

        // Create bwd rocFFT bwd
        rocfft_plan bwd = NULL;
        rocfft_plan_create(&bwd,
                           rocfft_placement_inplace,
                           rocfft_transform_type_complex_inverse,
                           rocfft_precision_single,
                           1, &length, 1, NULL);

        // Execute fwd
        rocfft_execute(fwd, (void**) &x, NULL, NULL);

        // Wait for execution to finish
        hipDeviceSynchronize();

        // Destroy fwd
        rocfft_plan_destroy(fwd);

        // Execute bwd
        rocfft_execute(bwd, (void**) &x, NULL, NULL);

        // Wait for execution to finish
        hipDeviceSynchronize();

        // Destroy bwd
        rocfft_plan_destroy(bwd);


        // Copy result back to host
        std::vector<float2> y(N);
        hipMemcpy(y.data(), x, Nbytes, hipMemcpyDeviceToHost);

        // Print results
        for (std::size_t i = 0; i < N; i++)
        {
          y[i].x *= 1.f/N;
          y[i].y *= 1.f/N;

          BOOST_TEST( cx[i].x == y[i].x );
          BOOST_TEST( cx[i].y == y[i].y );
        }

        // Free device buffer
        hipFree(x);

        rocfft_cleanup();


}

