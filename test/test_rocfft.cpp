#define BOOST_TEST_MODULE TestrocFFT
#include <iostream>
#include <vector>
#include <complex>

#include <cstdint>
#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "rocfft.h"
#include <boost/test/unit_test.hpp>
#include <iostream>

BOOST_AUTO_TEST_CASE( FFT1DSmall, * boost::unit_test::tolerance(0.0001f) )
{
  // rocFFT gpu compute
        // ========================================

        rocfft_setup();

        std::size_t N = 2 << 10; //should be small enough to not need a work_buffer
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


BOOST_AUTO_TEST_CASE( FFT1D, * boost::unit_test::tolerance(0.0001f) )
{
        // rocFFT gpu compute
        // ========================================

        rocfft_setup();

        std::size_t N = 128 << 10; //needs a work_buffer
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

        // Setup work buffer
        void *workBuffer = nullptr;
        size_t workBufferSize_fwd = 0;
        rocfft_plan_get_work_buffer_size(fwd, &workBufferSize_fwd);
        size_t workBufferSize_bwd = 0;
        rocfft_plan_get_work_buffer_size(bwd, &workBufferSize_bwd);

        BOOST_REQUIRE_EQUAL(workBufferSize_bwd,workBufferSize_fwd);

        // Setup exec info to pass work buffer to the library
        rocfft_execution_info info = nullptr;
        rocfft_execution_info_create(&info);

        if(workBufferSize_fwd > 0)
        {
                hipMalloc(&workBuffer, workBufferSize_fwd);
                rocfft_execution_info_set_work_buffer(info, workBuffer, workBufferSize_fwd);
        }

        // Execute fwd
        rocfft_execute(fwd, (void**) &x, NULL, info);

        // Wait for execution to finish
        hipDeviceSynchronize();

        // Destroy fwd
        rocfft_plan_destroy(fwd);

        // Execute bwd
        rocfft_execute(bwd, (void**) &x, NULL, info);

        // Wait for execution to finish
        hipDeviceSynchronize();

        // Destroy bwd
        rocfft_plan_destroy(bwd);

        if(workBuffer)
  	     	hipFree(workBuffer);

        rocfft_execution_info_destroy(info);

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

BOOST_AUTO_TEST_CASE( FFT1DSmall_stdcomplex, * boost::unit_test::tolerance(0.0001f) )
{
  // rocFFT gpu compute
        // ========================================

        rocfft_setup();

        std::size_t N = 2 << 10; //should be small enough to not need a work_buffer
        std::size_t Nbytes = N * sizeof(std::complex<float>);

        // Create HIP device buffer
        std::complex<float> *x;
        hipMalloc(&x, Nbytes);
        // Initialize data
        std::vector<std::complex<float>> cx(N);
        for (std::size_t i = 0; i < N; i++)
        {
                cx[i].real( 1 );
                cx[i].imag( -1);
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
        std::vector<std::complex<float>> y(N);
        hipMemcpy(y.data(), x, Nbytes, hipMemcpyDeviceToHost);

        // Print results
        for (std::size_t i = 0; i < N; i++)
        {
          y[i].real ( y[i].real() * 1.f/N);
          y[i].imag ( y[i].imag() * 1.f/N);

          BOOST_TEST( cx[i].real() == y[i].real() );
          BOOST_TEST( cx[i].imag() == y[i].imag() );
        }

        // Free device buffer
        hipFree(x);

        rocfft_cleanup();


}

BOOST_AUTO_TEST_CASE( failing_float_265841_Inplace_Real, * boost::unit_test::tolerance(0.0001f) )
{
        // rocFFT gpu compute
        // ========================================

        rocfft_setup();

        std::size_t N = 265841;
        std::size_t Nbytes = 2*(N/2 + 1) * sizeof(float);

        BOOST_REQUIRE_GT(Nbytes, N*sizeof(float));
        // Create HIP device buffer
        float *x;
        hipMalloc(&x, Nbytes);
        // Initialize data
        std::vector<float> h_x(N,1.);

        //  Copy data to device
        hipMemcpy(x, h_x.data(), Nbytes, hipMemcpyHostToDevice);
        // Create fwd rocFFT fwd
        rocfft_plan fwd = NULL;
        std::size_t length = N;
        rocfft_plan_create(&fwd,
                           rocfft_placement_inplace,
                           rocfft_transform_type_real_forward,
                           rocfft_precision_single,
                           1, &length, 1, NULL);

        // Create bwd rocFFT bwd
        rocfft_plan bwd = NULL;
        rocfft_plan_create(&bwd,
                           rocfft_placement_inplace,
                           rocfft_transform_type_real_inverse,
                           rocfft_precision_single,
                           1, &length, 1, NULL);

        // Setup work buffer
        void *workBuffer = nullptr;
        size_t workBufferSize_fwd = 0;
        rocfft_plan_get_work_buffer_size(fwd, &workBufferSize_fwd);
        size_t workBufferSize_bwd = 0;
        rocfft_plan_get_work_buffer_size(bwd, &workBufferSize_bwd);

        BOOST_REQUIRE_EQUAL(workBufferSize_bwd,workBufferSize_fwd);

        // Setup exec info to pass work buffer to the library
        rocfft_execution_info info = nullptr;
        rocfft_execution_info_create(&info);

        if(workBufferSize_fwd > 0)
        {
                hipMalloc(&workBuffer, workBufferSize_fwd);
                rocfft_execution_info_set_work_buffer(info, workBuffer, workBufferSize_fwd);
        }

        // Execute fwd
        rocfft_execute(fwd, (void**) &x, NULL, info);

        // Wait for execution to finish
        hipDeviceSynchronize();

        // Destroy fwd
        rocfft_plan_destroy(fwd);

        // Execute bwd
        rocfft_execute(bwd, (void**) &x, NULL, info);

        // Wait for execution to finish
        hipDeviceSynchronize();

        // Destroy bwd
        rocfft_plan_destroy(bwd);

        if(workBuffer)
  	     	hipFree(workBuffer);

        rocfft_execution_info_destroy(info);

        // Copy result back to host
        std::vector<float> h_y(N);
        hipMemcpy(h_y.data(), x, Nbytes, hipMemcpyDeviceToHost);

        // Print results
        for (std::size_t i = 0; i < N; i++)
        {
                h_y[i]*= 1.f/N;


                BOOST_TEST( h_x[i] == h_y[i] );

        }

        // Free device buffer
        hipFree(x);

        rocfft_cleanup();


}
