#define BOOST_TEST_MODULE TestFFTW

#include <fftw3.h>
#include <boost/test/unit_test.hpp>
#include <iostream>

BOOST_AUTO_TEST_CASE( FFT1DSingle, * boost::unit_test::tolerance(0.0001f) )
{
  static const int N = 32;
  fftwf_complex in[N], out[N], in2[N];
  fftwf_plan fwd, bwd;
  int i;
  for (i = 0; i < N; i++) {
    in[i][0] = static_cast<float>(i%7)/6;
    in[i][1] = 0;
  }

  fwd = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  fftwf_execute(fwd);
  fftwf_destroy_plan(fwd);

  bwd = fftwf_plan_dft_1d(N, out, in2, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftwf_execute(bwd);

  for (i = 0; i < N; i++) {
    in2[i][0] *= 1.f/N;
    in2[i][1] *= 1.f/N;

    BOOST_TEST( in[i][0] == in2[i][0] );
    BOOST_TEST( in[i][1] == in2[i][1] );
  }

  fftwf_destroy_plan(bwd);
  fftwf_cleanup();
}


BOOST_AUTO_TEST_CASE( FFT1D, * boost::unit_test::tolerance(0.0001) )
{
  static const int N = 32;
  fftw_complex in[N], out[N], in2[N];
  fftw_plan fwd, bwd;
  int i;
  for (i = 0; i < N; i++) {
    in[i][0] = static_cast<double>(i%7)/6;
    in[i][1] = 0;
  }

  fwd = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(fwd);
  fftw_destroy_plan(fwd);

  bwd = fftw_plan_dft_1d(N, out, in2, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(bwd);

  for (i = 0; i < N; i++) {
    in2[i][0] *= 1./N;
    in2[i][1] *= 1./N;

    BOOST_TEST( in[i][0] == in2[i][0] );
    BOOST_TEST( in[i][1] == in2[i][1] );
  }

  fftw_destroy_plan(bwd);
  fftw_cleanup();
}

BOOST_AUTO_TEST_CASE( FFT1D2Threads, * boost::unit_test::tolerance(0.0001) )
{
  static const int N = 32;
  fftw_complex in[N], out[N], in2[N];
  fftw_plan fwd, bwd;
  int i;
  for (i = 0; i < N; i++) {
    in[i][0] = static_cast<double>(i%7)/6;
    in[i][1] = 0;
  }

  int res = fftw_init_threads();
  if(res==0)
    BOOST_ERROR("fftw thread initialization failed.");
  fftw_plan_with_nthreads(2);

  fwd = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  if(!fwd)
    BOOST_ERROR("forward plan could not be created.");
  fftw_execute(fwd);
  fftw_destroy_plan(fwd);

  bwd = fftw_plan_dft_1d(N, out, in2, FFTW_BACKWARD, FFTW_ESTIMATE);
  if(!bwd)
    BOOST_ERROR("backward plan could not be created.");
  fftw_execute(bwd);

  for (i = 0; i < N; i++) {
    in2[i][0] *= 1./N;
    in2[i][1] *= 1./N;

    BOOST_TEST( in[i][0] == in2[i][0] );
    BOOST_TEST( in[i][1] == in2[i][1] );
  }

  fftw_destroy_plan(bwd);
  fftw_cleanup_threads();
  fftw_cleanup();
}
