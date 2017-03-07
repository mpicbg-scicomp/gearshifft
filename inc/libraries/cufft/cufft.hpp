#ifndef CUFFT_HPP_
#define CUFFT_HPP_

#include "core/application.hpp"
#include "core/timer_cuda.hpp"
#include "core/fft.hpp"
#include "core/traits.hpp"

#include "cufft_helper.hpp"

#include <array>
#include <cufft.h>
#include <vector_types.h>
#include <regex>

namespace gearshifft {
namespace CuFFT {
  namespace traits{
    template<typename T_Precision=float>
    struct Types
    {
      using ComplexType = cufftComplex;
      using RealType = cufftReal;
      static constexpr cufftType FFTForward = CUFFT_R2C;
      static constexpr cufftType FFTComplex = CUFFT_C2C;
      static constexpr cufftType FFTInverse = CUFFT_C2R;

      struct FFTExecuteForward{
        void operator()(cufftHandle plan, RealType* in, ComplexType* out){
          CHECK_CUDA(cufftExecR2C(plan, in, out));
        }
        void operator()(cufftHandle plan, ComplexType* in, ComplexType* out){
          CHECK_CUDA(cufftExecC2C(plan, in, out, CUFFT_FORWARD));
        }
      };
      struct FFTExecuteInverse{
        void operator()(cufftHandle plan, ComplexType* in, RealType* out){
          CHECK_CUDA(cufftExecC2R(plan, in, out));
        }
        void operator()(cufftHandle plan, ComplexType* in, ComplexType* out){
          CHECK_CUDA(cufftExecC2C(plan, in, out, CUFFT_INVERSE));
        }
      };
    };

    template<>
    struct Types<double>
    {
      using ComplexType = cufftDoubleComplex;
      using RealType = cufftDoubleReal;
      static constexpr cufftType FFTForward = CUFFT_D2Z;
      static constexpr cufftType FFTComplex = CUFFT_Z2Z;
      static constexpr cufftType FFTInverse = CUFFT_Z2D;

      struct FFTExecuteForward{
        void operator()(cufftHandle plan, RealType* in, ComplexType* out){
          CHECK_CUDA(cufftExecD2Z(plan, in, out));
        }
        void operator()(cufftHandle plan, ComplexType* in, ComplexType* out){
          CHECK_CUDA(cufftExecZ2Z(plan, in, out, CUFFT_FORWARD));
        }
      };
      struct FFTExecuteInverse{
        void operator()(cufftHandle plan, ComplexType* in, RealType* out){
          CHECK_CUDA(cufftExecZ2D(plan, in, out));
        }
        void operator()(cufftHandle plan, ComplexType* in, ComplexType* out){
          CHECK_CUDA(cufftExecZ2Z(plan, in, out, CUFFT_INVERSE));
        }
      };
    };

  }  // namespace traits

  /**
   * CUDA implicit context init and reset wrapper. Time is benchmarked.
   */
  struct Context {

    int device = 0;

    static const std::string title() {
      return "CuFFT";
    }

    static std::string get_device_list() {
      return listCudaDevices().str();
    }

    std::string get_used_device_properties() {
      auto ss = getCUDADeviceInformations(device);
      return ss.str();
    }

    void create() {
      const std::string options_devtype = Options::getInstance().getDevice();
      device = atoi(options_devtype.c_str());
      int nrdev=0;
      CHECK_CUDA(cudaGetDeviceCount(&nrdev));
      assert(nrdev>0);
      if(device<0 || device>=nrdev)
        device = 0;
      CHECK_CUDA(cudaSetDevice(device));
    }

    void destroy() {
      CHECK_CUDA(cudaDeviceReset());
    }
  };


  /**
   * Estimates memory reserved by cufft plan depending on FFT transform type
   * (CUFFT_R2C, ...) and depending on number of dimensions {1,2,3}.
   */
  template<cufftType FFTType, size_t NDim>
  size_t estimateAllocSize(cufftHandle& plan, const std::array<size_t,NDim>& e)
  {
    size_t s=0;
    CHECK_CUDA( cufftCreate(&plan) );
    if(NDim==1){
      CHECK_CUDA( cufftGetSize1d(plan, e[0], FFTType, 1, &s) );
    }
    if(NDim==2){
      CHECK_CUDA( cufftGetSize2d(plan, e[0], e[1], FFTType, &s) );
    }
    if(NDim==3){
      CHECK_CUDA( cufftGetSize3d(plan, e[0], e[1], e[2], FFTType, &s) );
    }
    return s;
  }

  template<cufftType FFTType, size_t NDim>
  size_t estimateAllocSize64(cufftHandle& plan, const std::array<size_t,NDim>& e) {
    size_t worksize=0;
    std::array<long long int, 3> lengths = {{0}};
    std::copy(e.begin(), e.end(), lengths.begin());
    CHECK_CUDA( cufftCreate(&plan) );
    CHECK_CUDA( cufftGetSizeMany64(plan, NDim, lengths.data(), NULL, 0, 0, NULL, 0, 0,
                                    FFTType,
                                    1, //batches
                                    &worksize));
    return worksize;
  }

  /**
   * Plan Creator depending on FFT transform type (CUFFT_R2C, ...).
   */
  template<cufftType FFTType>
  void makePlan(cufftHandle& plan, const std::array<size_t,1>& e){
    CHECK_CUDA(cufftPlan1d(&plan, e[0], FFTType, 1));
  }
  template<cufftType FFTType>
  void makePlan(cufftHandle& plan, const std::array<size_t,2>& e){
    CHECK_CUDA(cufftPlan2d(&plan, e[0], e[1], FFTType));
  }
  template<cufftType FFTType>
  void makePlan(cufftHandle& plan, const std::array<size_t,3>& e){
    CHECK_CUDA(cufftPlan3d(&plan, e[0], e[1], e[2], FFTType));
  }

  template<cufftType FFTType, size_t NDim>
  void makePlan64(cufftHandle& plan, const std::array<size_t,NDim>& e) {
    size_t worksize=0;
    std::array<long long int, 3> lengths = {{0}};
    std::copy(e.begin(), e.end(), lengths.begin());
    CHECK_CUDA( cufftCreate(&plan) );
    CHECK_CUDA( cufftMakePlanMany64(plan,
                                     NDim,
                                     lengths.data(),
                                     NULL, 0, 0, NULL, 0, 0, //ignored
                                     FFTType,
                                     1, //batches
                                     &worksize));
  }


  /**
   * CuFFT plan and execution class.
   *
   * This class handles:
   * - {1D, 2D, 3D} x {R2C, C2R, C2C} x {inplace, outplace} x {float, double}.
   */
  template<typename TFFT, // see fft.hpp (FFT_Inplace_Real, ...)
           typename TPrecision, // double, float
           size_t   NDim // 1..3
           >
  struct CuFFTImpl {
    using Extent = std::array<size_t,NDim>;
    using Types  = typename traits::Types<TPrecision>;
    using ComplexType = typename Types::ComplexType;
    using RealType = typename Types::RealType;
    using FFTExecuteForward  = typename Types::FFTExecuteForward;
    using FFTExecuteInverse = typename Types::FFTExecuteInverse;

    static constexpr
     bool IsInplace = TFFT::IsInplace;
    static constexpr
     bool IsComplex = TFFT::IsComplex;
    static constexpr
     bool IsInplaceReal = IsInplace && IsComplex==false;
    static constexpr
     cufftType FFTForward  = IsComplex ? Types::FFTComplex : Types::FFTForward;
    static constexpr
     cufftType FFTInverse = IsComplex ? Types::FFTComplex : Types::FFTInverse;

    using value_type  = typename std::conditional<IsComplex,ComplexType,RealType>::type;

    /// extents of the FFT input data
    Extent extents_   = {{0}};
    /// extents of the FFT complex data (=FFT(input))
    Extent extents_complex_ = {{0}};
    /// product of corresponding extents
    size_t n_         = 0;
    /// product of corresponding extents
    size_t n_complex_ = 0;

    cufftHandle  plan_              = 0;
    value_type*  data_              = nullptr;
    ComplexType* data_complex_      = nullptr;
    /// size in bytes of FFT input data
    size_t       data_size_         = 0;
    /// size in bytes of FFT(input) for out-of-place transforms
    size_t       data_complex_size_ = 0;
    /// if data sizes exceed 32-bit limit, the 64-bit cufft API is used
    bool         use64bit_          = false;

    CuFFTImpl(const Extent& cextents) {
      extents_ = interpret_as::column_major(cextents);
      extents_complex_ = extents_;
      n_ = std::accumulate(extents_.begin(),
                           extents_.end(),
                           1,
                           std::multiplies<size_t>());

      if(IsComplex==false){
        extents_complex_.back() = (extents_.back()/2 + 1);
      }

      n_complex_ = std::accumulate(extents_complex_.begin(),
                                   extents_complex_.end(),
                                   1,
                                   std::multiplies<size_t>());

      data_size_ = (IsInplaceReal? 2*n_complex_ : n_) * sizeof(value_type);
      if(IsInplace==false)
        data_complex_size_ = n_complex_ * sizeof(ComplexType);

      // There are some additional restrictions when using 64-bit API of cufft
      // see http://docs.nvidia.com/cuda/cufft/#unique_1972991535
      if(data_size_ >= (1ull<<32) || data_complex_size_ >= (1ull<<32))
        use64bit_ = true;
    }

    ~CuFFTImpl() {
      destroy();
    }

    /**
     * Returns allocated memory on device for FFT
     */
    size_t get_allocation_size() {
      return data_size_ + data_complex_size_;
    }

    /**
     * Returns size in bytes of one data transfer.
     *
     * Upload and download have the same size due to round-trip FFT.
     * \return Size in bytes of FFT data to be transferred (to device or to host memory buffer).
     */
    size_t get_transfer_size() {
      // when inplace-real then alloc'd data is bigger than data to be transferred
      return IsInplaceReal ? n_*sizeof(RealType) : data_size_;
    }

    /**
     * Returns estimated allocated memory on device for FFT plan
     */
    size_t get_plan_size() {
      size_t size1 = 0;
      size_t size2 = 0;
      if(use64bit_)
        size1 = estimateAllocSize64<FFTForward>(plan_,extents_);
      else{
        size1 = estimateAllocSize<FFTForward>(plan_,extents_);
      }
      CHECK_CUDA(cufftDestroy(plan_));
      plan_=0;
      if(use64bit_)
        size2 = estimateAllocSize64<FFTInverse>(plan_,extents_);
      else{
        size2 = estimateAllocSize<FFTInverse>(plan_,extents_);
      }
      CHECK_CUDA(cufftDestroy(plan_));
      plan_=0;

      // check available GPU memory
      size_t mem_free=0, mem_tot=0;
      CHECK_CUDA( cudaMemGetInfo(&mem_free, &mem_tot) );
      size_t wanted = std::max(size1,size2) + data_size_ + data_complex_size_;
      if(mem_free<wanted) {
        std::stringstream ss;
        ss << mem_free << "<" << wanted << " (bytes)";
        throw std::runtime_error("Not enough GPU memory available. "+ss.str());
      }
      return std::max(size1,size2);
    }

    // --- next methods are benchmarked ---

    /**
     * Allocate buffers on CUDA device
     */
    void allocate() {
      CHECK_CUDA(cudaMalloc(&data_, data_size_));

      if(IsInplace) {
        data_complex_ = reinterpret_cast<ComplexType*>(data_);
      }else{
        CHECK_CUDA(cudaMalloc(&data_complex_, data_complex_size_));
      }
    }

    // create FFT plan handle
    void init_forward() {
      if(use64bit_)
        makePlan64<FFTForward>(plan_, extents_);
      else
        makePlan<FFTForward>(plan_, extents_);
    }

    // recreates plan if needed
    void init_inverse() {
      if(IsComplex==false){
        CHECK_CUDA(cufftDestroy(plan_));
        plan_=0;
        if(use64bit_)
          makePlan64<FFTInverse>(plan_, extents_);
        else
          makePlan<FFTInverse>(plan_, extents_);
      }
    }

    void execute_forward() {
      FFTExecuteForward()(plan_, data_, data_complex_);
    }

    void execute_inverse() {
      FFTExecuteInverse()(plan_, data_complex_, data_);
    }

    template<typename THostData>
    void upload(THostData* input) {
      if(IsInplaceReal && NDim>1) {
        size_t w      = extents_[NDim-1] * sizeof(THostData);
        size_t h      = n_ * sizeof(THostData) / w;
        size_t pitch  = (extents_[NDim-1]/2+1) * sizeof(ComplexType);
        CHECK_CUDA(cudaMemcpy2D(data_, pitch, input, w, w, h, cudaMemcpyHostToDevice));
      }else{
        CHECK_CUDA(cudaMemcpy(data_, input, get_transfer_size(), cudaMemcpyHostToDevice));
      }
    }

    template<typename THostData>
    void download(THostData* output) {
      if(IsInplaceReal && NDim>1) {
        size_t w      = extents_[NDim-1] * sizeof(THostData);
        size_t h      = n_ * sizeof(THostData) / w;
        size_t pitch  = (extents_[NDim-1]/2+1) * sizeof(ComplexType);
        CHECK_CUDA(cudaMemcpy2D(output, w, data_, pitch, w, h, cudaMemcpyDeviceToHost));
      }else{
        CHECK_CUDA(cudaMemcpy(output, data_, get_transfer_size(), cudaMemcpyDeviceToHost));
      }
    }

    void destroy() {
      CHECK_CUDA( cudaFree(data_) );
      data_=nullptr;
      if(IsInplace==false && data_complex_) {
        CHECK_CUDA( cudaFree(data_complex_) );
        data_complex_ = nullptr;
      }
      if(plan_) {
        CHECK_CUDA( cufftDestroy(plan_) );
        plan_=0;
      }
    }
  };

    using Inplace_Real = gearshifft::FFT<FFT_Inplace_Real,
                                         FFT_Plan_Reusable,
                                         CuFFTImpl,
                                         TimerGPU >;
    using Outplace_Real = gearshifft::FFT<FFT_Outplace_Real,
                                          FFT_Plan_Reusable,
                                          CuFFTImpl,
                                          TimerGPU >;
    using Inplace_Complex = gearshifft::FFT<FFT_Inplace_Complex,
                                            FFT_Plan_Reusable,
                                            CuFFTImpl,
                                            TimerGPU >;
    using Outplace_Complex = gearshifft::FFT<FFT_Outplace_Complex,
                                             FFT_Plan_Reusable,
                                             CuFFTImpl,
                                             TimerGPU >;

} // namespace CuFFT
} // namespace gearshifft


#endif /* CUFFT_HPP_ */
