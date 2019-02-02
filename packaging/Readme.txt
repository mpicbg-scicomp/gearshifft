gearshifft is an FFT library benchmark suite for cuFFT (CUDA, Nvidia GPUs), clFFT (OpenCL) and FFTW.
The benchmarks consist of a round-trip FFT, where each step is measured.
Written in C++14 most of the code layers are removed at compile-time to aim for minimal overhead.
The results are stored to a csv file for later processing.

Results and processing scripts can be found on: https://github.com/mpicbg-scicomp/gearshifft_results

See help message (pass `--help|-h`) for the command line options.

Example: Benchmark of clFFT (also applies for cuFFT, FFTW, ...)

512x512-point FFT, single and double precision, all transforms, print result to console.
./gearshifft_clfft -e 512x512 -v
