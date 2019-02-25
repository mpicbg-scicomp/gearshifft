#define BOOST_TEST_MODULE TestrocFFT
#include "libraries/rocfft/rocfft_helper.hpp"

#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <rocfft.h>

#include <boost/test/included/unit_test.hpp> // Single-header usage variant

#include <cstdint>

#include <iostream>
#include <vector>
#include <complex>

BOOST_AUTO_TEST_CASE( FFT1DSmall, * boost::unit_test::tolerance(0.0001f) )
{
  // rocFFT gpu compute
        // ========================================

        CHECK_HIP(rocfft_setup());

        std::size_t N = 2 << 10; //should be small enough to not need a work_buffer
        std::size_t Nbytes = N * sizeof(float2);

        // Create HIP device buffer
        float2 *x;
        CHECK_HIP(hipMalloc(&x, Nbytes));
        // Initialize data
        std::vector<float2> cx(N);
        for (std::size_t i = 0; i < N; i++)
        {
                cx[i].x = 1;
                cx[i].y = -1;
        }

        //  Copy data to device
        CHECK_HIP(hipMemcpy(x, cx.data(), Nbytes, hipMemcpyHostToDevice));
        // Create fwd rocFFT fwd
        rocfft_plan fwd = nullptr;
        std::size_t length = N;
        CHECK_HIP(
          rocfft_plan_create(&fwd,
                             rocfft_placement_inplace,
                             rocfft_transform_type_complex_forward,
                             rocfft_precision_single,
                             1, &length, 1, nullptr));

        // Create bwd rocFFT bwd
        rocfft_plan bwd = nullptr;
        CHECK_HIP(
          rocfft_plan_create(&bwd,
                             rocfft_placement_inplace,
                             rocfft_transform_type_complex_inverse,
                             rocfft_precision_single,
                             1, &length, 1, nullptr));


        // Execute fwd
        CHECK_HIP(rocfft_execute(fwd, (void**) &x, nullptr, nullptr));

        // Wait for execution to finish
        CHECK_HIP(hipDeviceSynchronize());

        // Destroy fwd
        CHECK_HIP(rocfft_plan_destroy(fwd));

        // Execute bwd
        CHECK_HIP(rocfft_execute(bwd, (void**) &x, nullptr, nullptr));

        // Wait for execution to finish
        CHECK_HIP(hipDeviceSynchronize());

        // Destroy bwd
        CHECK_HIP(rocfft_plan_destroy(bwd));


        // Copy result back to host
        std::vector<float2> y(N);
        CHECK_HIP(hipMemcpy(y.data(), x, Nbytes, hipMemcpyDeviceToHost));

        // Print results
        for (std::size_t i = 0; i < N; i++)
        {
          y[i].x *= 1.f/N;
          y[i].y *= 1.f/N;

          BOOST_TEST( cx[i].x == y[i].x );
          BOOST_TEST( cx[i].y == y[i].y );
        }

        // Free device buffer
        CHECK_HIP(hipFree(x));

        CHECK_HIP(rocfft_cleanup());


}


BOOST_AUTO_TEST_CASE( FFT1D, * boost::unit_test::tolerance(0.0001f) )
{
        // rocFFT gpu compute
        // ========================================

        CHECK_HIP(rocfft_setup());

        std::size_t N = 128 << 10; //needs a work_buffer
        std::size_t Nbytes = N * sizeof(float2);

        // Create HIP device buffer
        float2 *x;
        CHECK_HIP(hipMalloc(&x, Nbytes));
        // Initialize data
        std::vector<float2> cx(N);
        for (std::size_t i = 0; i < N; i++)
        {
                cx[i].x = 1;
                cx[i].y = -1;
        }

        //  Copy data to device
        CHECK_HIP(hipMemcpy(x, cx.data(), Nbytes, hipMemcpyHostToDevice));
        // Create fwd rocFFT fwd
        rocfft_plan fwd = nullptr;
        std::size_t length = N;
        CHECK_HIP(
          rocfft_plan_create(&fwd,
                             rocfft_placement_inplace,
                             rocfft_transform_type_complex_forward,
                             rocfft_precision_single,
                             1, &length, 1, nullptr));

        // Create bwd rocFFT bwd
        rocfft_plan bwd = nullptr;
        CHECK_HIP(
          rocfft_plan_create(&bwd,
                             rocfft_placement_inplace,
                             rocfft_transform_type_complex_inverse,
                             rocfft_precision_single,
                             1, &length, 1, nullptr));

        // Setup work buffer
        void *workBuffer = nullptr;
        size_t workBufferSize_fwd = 0;
        CHECK_HIP(rocfft_plan_get_work_buffer_size(fwd, &workBufferSize_fwd));
        size_t workBufferSize_bwd = 0;
        CHECK_HIP(rocfft_plan_get_work_buffer_size(bwd, &workBufferSize_bwd));

        BOOST_REQUIRE_EQUAL(workBufferSize_bwd,workBufferSize_fwd);

        // Setup exec info to pass work buffer to the library
        rocfft_execution_info info = nullptr;
        CHECK_HIP(rocfft_execution_info_create(&info));

        if(workBufferSize_fwd > 0)
        {
                CHECK_HIP(hipMalloc(&workBuffer, workBufferSize_fwd));
                CHECK_HIP(rocfft_execution_info_set_work_buffer(info, workBuffer, workBufferSize_fwd));
        }

        // Execute fwd
        CHECK_HIP(rocfft_execute(fwd, (void**) &x, nullptr, info));

        // Wait for execution to finish
        CHECK_HIP(hipDeviceSynchronize());

        // Destroy fwd
        CHECK_HIP(rocfft_plan_destroy(fwd));

        // Execute bwd
        CHECK_HIP(rocfft_execute(bwd, (void**) &x, nullptr, info));

        // Wait for execution to finish
        CHECK_HIP(hipDeviceSynchronize());

        // Destroy bwd
        CHECK_HIP(rocfft_plan_destroy(bwd));

        if(workBuffer)
  	     	CHECK_HIP(hipFree(workBuffer));

        CHECK_HIP(rocfft_execution_info_destroy(info));

        // Copy result back to host
        std::vector<float2> y(N);
        CHECK_HIP(hipMemcpy(y.data(), x, Nbytes, hipMemcpyDeviceToHost));

        // Print results
        for (std::size_t i = 0; i < N; i++)
        {
                y[i].x *= 1.f/N;
                y[i].y *= 1.f/N;

                BOOST_TEST( cx[i].x == y[i].x );
                BOOST_TEST( cx[i].y == y[i].y );
        }

        // Free device buffer
        CHECK_HIP(hipFree(x));

        CHECK_HIP(rocfft_cleanup());


}

BOOST_AUTO_TEST_CASE( FFT1DSmall_stdcomplex, * boost::unit_test::tolerance(0.0001f) )
{
  // rocFFT gpu compute
        // ========================================

        CHECK_HIP(rocfft_setup());

        std::size_t N = 2 << 10; //should be small enough to not need a work_buffer
        std::size_t Nbytes = N * sizeof(std::complex<float>);

        // Create HIP device buffer
        std::complex<float> *x;
        CHECK_HIP(hipMalloc(&x, Nbytes));
        // Initialize data
        std::vector<std::complex<float>> cx(N);
        for (std::size_t i = 0; i < N; i++)
        {
                cx[i].real( 1 );
                cx[i].imag( -1);
        }

        //  Copy data to device
        CHECK_HIP(hipMemcpy(x, cx.data(), Nbytes, hipMemcpyHostToDevice));
        // Create fwd rocFFT fwd
        rocfft_plan fwd = nullptr;
        std::size_t length = N;
        CHECK_HIP(
          rocfft_plan_create(&fwd,
                             rocfft_placement_inplace,
                             rocfft_transform_type_complex_forward,
                             rocfft_precision_single,
                             1, &length, 1, nullptr));

        // Create bwd rocFFT bwd
        rocfft_plan bwd = nullptr;
        CHECK_HIP(
          rocfft_plan_create(&bwd,
                             rocfft_placement_inplace,
                             rocfft_transform_type_complex_inverse,
                             rocfft_precision_single,
                             1, &length, 1, nullptr));


        // Execute fwd
        CHECK_HIP(rocfft_execute(fwd, (void**) &x, nullptr, nullptr));

        // Wait for execution to finish
        CHECK_HIP(hipDeviceSynchronize());

        // Destroy fwd
        CHECK_HIP(rocfft_plan_destroy(fwd));

        // Execute bwd
        CHECK_HIP(rocfft_execute(bwd, (void**) &x, nullptr, nullptr));

        // Wait for execution to finish
        CHECK_HIP(hipDeviceSynchronize());

        // Destroy bwd
        CHECK_HIP(rocfft_plan_destroy(bwd));


        // Copy result back to host
        std::vector<std::complex<float>> y(N);
        CHECK_HIP(hipMemcpy(y.data(), x, Nbytes, hipMemcpyDeviceToHost));

        // Print results
        for (std::size_t i = 0; i < N; i++)
        {
          y[i].real ( y[i].real() * 1.f/N);
          y[i].imag ( y[i].imag() * 1.f/N);

          BOOST_TEST( cx[i].real() == y[i].real() );
          BOOST_TEST( cx[i].imag() == y[i].imag() );
        }

        // Free device buffer
        CHECK_HIP(hipFree(x));

        CHECK_HIP(rocfft_cleanup());


}

BOOST_AUTO_TEST_CASE( failing_float_265841_Inplace_Real, * boost::unit_test::tolerance(0.0001f) )
{
        // rocFFT gpu compute
        // ========================================

        CHECK_HIP(rocfft_setup());

        std::size_t N = 265841;
        std::size_t Nbytes = 2*(N/2 + 1) * sizeof(float);

        BOOST_REQUIRE_GT(Nbytes, N*sizeof(float));
        // Create HIP device buffer
        float *x;
        CHECK_HIP(hipMalloc(&x, Nbytes));
        // Initialize data
        std::vector<float> h_x(N,1.);

        //  Copy data to device
        CHECK_HIP(hipMemcpy(x, h_x.data(), Nbytes, hipMemcpyHostToDevice));
        // Create fwd rocFFT fwd
        rocfft_plan fwd = nullptr;
        std::size_t length = N;
        CHECK_HIP(
          rocfft_plan_create(&fwd,
                             rocfft_placement_inplace,
                             rocfft_transform_type_real_forward,
                             rocfft_precision_single,
                             1, &length, 1, nullptr));

        // Create bwd rocFFT bwd
        rocfft_plan bwd = nullptr;
        CHECK_HIP(
          rocfft_plan_create(&bwd,
                             rocfft_placement_inplace,
                             rocfft_transform_type_real_inverse,
                             rocfft_precision_single,
                             1, &length, 1, nullptr));

        // Setup work buffer
        void *workBuffer = nullptr;
        size_t workBufferSize_fwd = 0;
        CHECK_HIP(rocfft_plan_get_work_buffer_size(fwd, &workBufferSize_fwd));
        size_t workBufferSize_bwd = 0;
        CHECK_HIP(rocfft_plan_get_work_buffer_size(bwd, &workBufferSize_bwd));

        BOOST_REQUIRE_EQUAL(workBufferSize_bwd,workBufferSize_fwd);

        // Setup exec info to pass work buffer to the library
        rocfft_execution_info info = nullptr;
        CHECK_HIP(rocfft_execution_info_create(&info));

        if(workBufferSize_fwd > 0)
        {
                CHECK_HIP(hipMalloc(&workBuffer, workBufferSize_fwd));
                CHECK_HIP(rocfft_execution_info_set_work_buffer(info, workBuffer, workBufferSize_fwd));
        }

        // Execute fwd
        CHECK_HIP(rocfft_execute(fwd, (void**) &x, nullptr, info));

        // Wait for execution to finish
        CHECK_HIP(hipDeviceSynchronize());

        // Destroy fwd
        CHECK_HIP(rocfft_plan_destroy(fwd));

        // Execute bwd
        CHECK_HIP(rocfft_execute(bwd, (void**) &x, nullptr, info));

        // Wait for execution to finish
        CHECK_HIP(hipDeviceSynchronize());

        // Destroy bwd
        CHECK_HIP(rocfft_plan_destroy(bwd));

        if(workBuffer)
  	     	CHECK_HIP(hipFree(workBuffer));

        CHECK_HIP(rocfft_execution_info_destroy(info));

        // Copy result back to host
        std::vector<float> h_y(N);
        CHECK_HIP(hipMemcpy(h_y.data(), x, Nbytes, hipMemcpyDeviceToHost));

        // Print results
        for (std::size_t i = 0; i < N; i++)
        {
                h_y[i]*= 1.f/N;


                BOOST_TEST( h_x[i] == h_y[i] );

        }

        // Free device buffer
        CHECK_HIP(hipFree(x));

        CHECK_HIP(rocfft_cleanup());


}
