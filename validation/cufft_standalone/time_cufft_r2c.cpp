// nvcc -o cufft test_cufft.cpp -std=c++11 -lcufft
#include <cuda_runtime.h>
#include <iostream>
#include <cufft.h>
#include <stdexcept>
#include <array>
#include <vector>
#include <chrono>
#include <cstring>
#include <ctime>
#include <sstream>
#include <fstream>

// CHECK_CUDA template function for cuda and cufft
//  and throws std::runtime_error
#define CHECK_CUDA(ans) check_cuda((ans), "", #ans, __FILE__, __LINE__)
inline
void throw_error(int code,
                 const char* error_string,
                 const char* msg,
                 const char* func,
                 const char* file,
                 int line) {
  throw std::runtime_error("CUDA error "
                           +std::string(msg)
                           +" "+std::string(error_string)
                           +" ["+std::to_string(code)+"]"
                           +" "+std::string(file)
                           +":"+std::to_string(line)
                           +" "+std::string(func)
    );
}
static const char* cufftResultToString(cufftResult error) {
  switch (error) {
  case CUFFT_SUCCESS:
    return "CUFFT_SUCCESS";

  case CUFFT_INVALID_PLAN:
    return "CUFFT_INVALID_PLAN";

  case CUFFT_ALLOC_FAILED:
    return "CUFFT_ALLOC_FAILED";

  case CUFFT_INVALID_TYPE:
    return "CUFFT_INVALID_TYPE";

  case CUFFT_INVALID_VALUE:
    return "CUFFT_INVALID_VALUE";

  case CUFFT_INTERNAL_ERROR:
    return "CUFFT_INTERNAL_ERROR";

  case CUFFT_EXEC_FAILED:
    return "CUFFT_EXEC_FAILED";

  case CUFFT_SETUP_FAILED:
    return "CUFFT_SETUP_FAILED";

  case CUFFT_INVALID_SIZE:
    return "CUFFT_INVALID_SIZE";

  case CUFFT_UNALIGNED_DATA:
    return "CUFFT_UNALIGNED_DATA";
  }
  return "<unknown>";
}

inline
void check_cuda(cudaError_t code, const char* msg, const char *func, const char *file, int line) {
  if (code != cudaSuccess) {
    throw_error(static_cast<int>(code),
                cudaGetErrorString(code), msg, func, file, line);
  }
}
inline
void check_cuda(cufftResult code, const char* msg,  const char *func, const char *file, int line) {
  if (code != CUFFT_SUCCESS) {
    throw_error(static_cast<int>(code),
                cufftResultToString(code), "cufft", func, file, line);
  }
}

std::stringstream getCUDADeviceInformations(int dev) {
  std::stringstream info;
  cudaDeviceProp prop;
  int runtimeVersion = 0;
  int cufftv = 0;
  size_t f=0, t=0;
  CHECK_CUDA( cufftGetVersion(&cufftv) );
  CHECK_CUDA( cudaRuntimeGetVersion(&runtimeVersion) );
  CHECK_CUDA( cudaGetDeviceProperties(&prop, dev) );
  CHECK_CUDA( cudaMemGetInfo(&f, &t) );
  info << '"' << prop.name << '"'
       << ", \"CC\", " << prop.major << '.' << prop.minor
       << ", \"Multiprocessors\", "<< prop.multiProcessorCount
       << ", \"Memory [MiB]\", "<< t/1048576
       << ", \"MemoryFree [MiB]\", " << f/1048576
       << ", \"MemClock [MHz]\", " << prop.memoryClockRate/1000
       << ", \"GPUClock [MHz]\", " << prop.clockRate/1000
       << ", \"CUDA Runtime\", " << runtimeVersion
       << ", \"cufft\", " << cufftv
    ;
  return info;
}


/**
 * Perform iterations of inplace real 1D round-trip FFT
 * and collect time-to-solution results.
 *
 * \param extent Extent of 1D FFT
 * \param id iteration id
 * \param buffer ... where results in csv format are stored to
 * \param runs number of repetitions of benchmark
 */
template<typename TReal, typename TComplex>
void run_time_to_solution(int extent, int ID, std::stringstream& buffer, int runs) {

  TReal* hdata = nullptr;
  TReal* hdata_buffer = nullptr;
  TReal* data;
  cudaDeviceProp prop;

  cufftHandle plan;
  std::vector< std::array<size_t, 1> > vec_extents;
  vec_extents.push_back( {static_cast<size_t>(extent)} );
//  vec_extents.push_back( {16777216} );

  typedef std::chrono::high_resolution_clock clock;

  for( auto extents : vec_extents ) {
    size_t data_size = 2*(extents[0]/2+1)*sizeof(TReal); // for inplace R2C

    hdata = new TReal[extents[0]];
    for(auto i=0; i<extents[0]; ++i)
      hdata[i] = 1.0f;
    hdata_buffer = new TReal[extents[0]];

    try {

      for(auto i=0; i<=runs; ++i) {
        clock::time_point starttotal;

        // copy data to buffer that can be written to as gearshifft does
        std::memcpy(hdata_buffer, hdata, extents[0]*sizeof(TReal));
        CHECK_CUDA( cudaDeviceSynchronize() );
        starttotal = clock::now();


        CHECK_CUDA( cudaMalloc(&data, data_size));

        CHECK_CUDA( cufftPlan1d(&plan, extents[0], CUFFT_R2C, 1));

        CHECK_CUDA( cudaMemcpy(data, hdata_buffer, extents[0]*sizeof(TReal), cudaMemcpyHostToDevice) );

        CHECK_CUDA( cufftExecR2C(plan,
                                 data,
                                 reinterpret_cast<TComplex*>(data))); // inplace

        CHECK_CUDA( cufftDestroy(plan) );
        CHECK_CUDA( cufftPlan1d(&plan, extents[0], CUFFT_C2R, 1));

        CHECK_CUDA( cufftExecC2R(plan,
                                 reinterpret_cast<TComplex*>(data),
                                 data)); // inplace

        CHECK_CUDA( cudaMemcpy(hdata_buffer, data, extents[0]*sizeof(TReal), cudaMemcpyDeviceToHost) );

        CHECK_CUDA( cudaFree(data) );
        CHECK_CUDA( cufftDestroy(plan) );

        auto diff = clock::now() - starttotal;
        auto duration = std::chrono::duration<double, std::milli> (diff).count();

        // validation
        for(auto i=0; i<extents[0]; ++i)
          if(fabs(hdata_buffer[i]/extents[0]-hdata[i])>0.00001)
            throw std::runtime_error("Mismatch.");

        if(i==0) {
          continue; //i=0 => warmup
        }

        CHECK_CUDA( cudaGetDeviceProperties(&prop, 0) );
        buffer << "\"cuFFT Standalone-TTS\",\"Inplace\",\"Real\",\"float\",1,\"powerof2\","<<extents[0]<<",0,0,"<<i-1<<","<<ID<<",\"Success\","
               << "0,"
               << "0,"
               << "0,"
               << "0,"
               << "0,"
               << "0,"
               << "0,"
               << "0,"
               << duration<<","
               << data_size
               << ",0,0"
               << "\n";
      }

      delete[] hdata;
      delete[] hdata_buffer;
      hdata = nullptr;

    }catch(...){
      std::cerr << "Error for nx="<<extents[0] << std::endl;
      // cleanup
      CHECK_CUDA( cudaFree(data) );
      delete[] hdata;
      delete[] hdata_buffer;
      if(plan) {
        CHECK_CUDA( cufftDestroy(plan) );
      }
    }
  }
}

/**
 * Perform iterations of inplace real 1D round-trip FFT
 * and collect timestamps of FFT related operations.
 *
 * \param extent Extent of 1D FFT
 * \param id iteration id
 * \param buffer ... where results in csv format are stored to
 * \param runs number of repetitions of benchmark
 */
template<typename TReal, typename TComplex>
void run_deep_timings(int extent, int ID, std::stringstream& buffer, int runs) {

  TReal* hdata = nullptr;
  TReal* hdata_buffer = nullptr;
  TReal* data;
  cudaDeviceProp prop;

  cufftHandle plan;
  std::vector< std::array<size_t, 1> > vec_extents;
  vec_extents.push_back( {static_cast<size_t>(extent)} );
//  vec_extents.push_back( {16777216} );

  typedef std::chrono::high_resolution_clock clock;

  for( auto extents : vec_extents ) {
    size_t data_size = 2*(extents[0]/2+1)*sizeof(TReal); // for inplace R2C

    hdata = new TReal[extents[0]];
    for(auto i=0; i<extents[0]; ++i)
      hdata[i] = 1.0f;
    hdata_buffer = new TReal[extents[0]];

    try {

      for(auto i=0; i<=runs; ++i) {
        clock::time_point starttotal;
        clock::time_point start;
        cudaEvent_t startdev, enddev;
        CHECK_CUDA(cudaEventCreate(&startdev));
        CHECK_CUDA(cudaEventCreate(&enddev));
        double duration_alloc = 0;
        double duration_initfwd = 0;
        double duration_initbwd = 0;
        float duration_upload = 0;
        float duration_fft = 0;
        float duration_ifft = 0;
        float duration_download = 0;
        double duration_destroy = 0;

        // copy data to buffer that can be written to as gearshifft does
        std::memcpy(hdata_buffer, hdata, extents[0]*sizeof(TReal));
        CHECK_CUDA( cudaDeviceSynchronize() );
        starttotal = clock::now();


        start = clock::now();
        CHECK_CUDA( cudaMalloc(&data, data_size));
        auto diff = clock::now() - start;
        duration_alloc = std::chrono::duration<double, std::milli> (diff).count();


        start = clock::now();
        CHECK_CUDA( cufftPlan1d(&plan, extents[0], CUFFT_R2C, 1));
        diff = clock::now() - start;
        duration_initfwd = std::chrono::duration<double, std::milli> (diff).count();


        CHECK_CUDA( cudaEventRecord(startdev) );
        CHECK_CUDA( cudaMemcpy(data, hdata_buffer, extents[0]*sizeof(TReal), cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaEventRecord(enddev) );
        CHECK_CUDA( cudaEventSynchronize(enddev));
        CHECK_CUDA( cudaEventElapsedTime(&duration_upload, startdev, enddev));


        CHECK_CUDA( cudaEventRecord(startdev) );
        CHECK_CUDA( cufftExecR2C(plan,
                                 data,
                                 reinterpret_cast<TComplex*>(data))); // inplace
        CHECK_CUDA( cudaEventRecord(enddev) );
        CHECK_CUDA( cudaEventSynchronize(enddev));
        CHECK_CUDA( cudaEventElapsedTime(&duration_fft, startdev, enddev));


        start = clock::now();
        CHECK_CUDA( cufftDestroy(plan) );
        CHECK_CUDA( cufftPlan1d(&plan, extents[0], CUFFT_C2R, 1));
        diff = clock::now() - start;
        duration_initbwd = std::chrono::duration<double, std::milli> (diff).count();


        CHECK_CUDA( cudaEventRecord(startdev) );
        CHECK_CUDA( cufftExecC2R(plan,
                                 reinterpret_cast<TComplex*>(data),
                                 data)); // inplace
        CHECK_CUDA( cudaEventRecord(enddev) );
        CHECK_CUDA( cudaEventSynchronize(enddev));
        CHECK_CUDA( cudaEventElapsedTime(&duration_ifft, startdev, enddev));

        CHECK_CUDA( cudaEventRecord(startdev) );
        CHECK_CUDA( cudaMemcpy(hdata_buffer, data, extents[0]*sizeof(TReal), cudaMemcpyDeviceToHost) );
        CHECK_CUDA( cudaEventRecord(enddev) );
        CHECK_CUDA( cudaEventSynchronize(enddev));
        CHECK_CUDA( cudaEventElapsedTime(&duration_download, startdev, enddev));


        start = clock::now();
        CHECK_CUDA( cudaFree(data) );
        CHECK_CUDA( cufftDestroy(plan) );
        diff = clock::now() - start;
        duration_destroy = std::chrono::duration<double, std::milli> (diff).count();

        diff = clock::now() - starttotal;
        auto duration = std::chrono::duration<double, std::milli> (diff).count();

        CHECK_CUDA( cudaEventDestroy(startdev) );
        CHECK_CUDA( cudaEventDestroy(enddev) );

        // validation
        for(auto i=0; i<extents[0]; ++i)
          if(fabs(hdata_buffer[i]/extents[0]-hdata[i])>0.00001)
            throw std::runtime_error("Mismatch.");

        if(i==0) {
          continue; //i=0 => warmup
        }

        CHECK_CUDA( cudaGetDeviceProperties(&prop, 0) );
        buffer << "\"cuFFT Standalone\",\"Inplace\",\"Real\",\"float\",1,\"powerof2\","<<extents[0]<<",0,0,"<<i-1<<","<<ID<<",\"Success\","
               << duration_alloc<<","
               << duration_initfwd<<","
               << duration_initbwd<<","
               << duration_upload<<","
               << duration_fft<<","
               << duration_ifft<<","
               << duration_download<<","
               << duration_destroy<<","
               << duration<<","
               << data_size
               << ",0,0"
               << "\n";
      }

      delete[] hdata;
      delete[] hdata_buffer;
      hdata = nullptr;

    }catch(...){
      std::cerr << "Error for nx="<<extents[0] << std::endl;
      // cleanup
      CHECK_CUDA( cudaFree(data) );
      delete[] hdata;
      delete[] hdata_buffer;
      if(plan) {
        CHECK_CUDA( cufftDestroy(plan) );
      }
    }
  }
}


//
int main(int argc, char** argv) {
  int extent = atoi(argv[1]);
  int iterations = atoi(argv[2]);
  const char* fname = argv[3];
  int runs = atoi(argv[4]);
  int timings = atoi(argv[5]);
  std::stringstream buffer;

  if(extent<=0)
    throw std::runtime_error("extent must be a positive integer.");
  if(iterations<=0)
    throw std::runtime_error("iterations must be a positive integer.");
  if(runs<=0)
    throw std::runtime_error("runs must be a positive integer.");

  CHECK_CUDA( cudaSetDevice(0) );
// ; "Tesla K80", "CC", 3.7, "Multiprocessors", 13, "Memory [MiB]", 11439, "MemoryFree [MiB]", 11376, "MemClock [MHz]", 2505, "GPUClock [MHz]", 823, "CUDA Runtime", 8000, "cufft", 8000,"NumberRuns",10,"NumberWarmups",1,"ErrorBound",1e-05,"CurrentTime",1489878090,"CurrentTimeLocal","Sun Mar 19 00:01:30 2017"
  std::time_t now = std::time(nullptr);
  std::stringstream meta_information;
  meta_information << getCUDADeviceInformations(0).str()
                   << ",\"NumberRuns\"," << runs
                   << ",\"NumberWarmups\"," << 1
                   << ",\"ErrorBound\"," << 0.00001
                   << ",\"CurrentTime\"," << now
                   << ",\"CurrentTimeLocal\",\"" << strtok(ctime(&now), "\n") << "\""
    ;
  buffer << meta_information.str() << "\n"
         << "; \"Time_ContextCreate [ms]\", NA\n"
         << "; \"Time_ContextDestroy [ms]\", NA\n"
         << "\"library\",\"inplace\",\"complex\",\"precision\",\"dim\",\"kind\",\"nx\",\"ny\",\"nz\",\"run\",\"id\",\"success\",\"Time_Allocation [ms]\",\"Time_PlanInitFwd [ms]\",\"Time_PlanInitInv [ms]\",\"Time_Upload [ms]\",\"Time_FFT [ms]\",\"Time_iFFT [ms]\",\"Time_Download [ms]\",\"Time_PlanDestroy [ms]\",\"Time_Total [ms]\",\"Size_DeviceBuffer [bytes]\",\"Size_DevicePlan [bytes]\",\"Size_DeviceTransfer [bytes]\"\n";

  if(timings) {
    for(int i=0; i<iterations; ++i) {
      run_deep_timings<float,cufftComplex>(extent, i, buffer, runs);
    }
  } else {
    for(int i=0; i<iterations; ++i) {
      run_time_to_solution<float,cufftComplex>(extent, i, buffer, runs);
    }
  }

  std::ofstream fs;
  fs.open(fname);
  fs << buffer.str();
  fs.close();
  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
