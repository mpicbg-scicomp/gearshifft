#ifndef CUFFT_HPP_
#define CUFFT_HPP_

#include "application.hpp"
#include "timer.hpp"
#include "fft.hpp"
#include "benchmark_suite.hpp"
#include "cufft_helper.hpp"
#include "traits.hpp"

#include <array>
#include <cufft.h>
#include <vector_types.h>

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
      static constexpr cufftType FFTBackward = CUFFT_C2R;

      struct FFTExecuteForward{
        void operator()(cufftHandle plan, RealType* in, ComplexType* out){
          CHECK_CUDA(cufftExecR2C(plan, in, out));
        }
        void operator()(cufftHandle plan, ComplexType* in, ComplexType* out){
          CHECK_CUDA(cufftExecC2C(plan, in, out, CUFFT_FORWARD));
        }
      };
      struct FFTExecuteBackward{
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
      static constexpr cufftType FFTBackward = CUFFT_Z2D;

      struct FFTExecuteForward{
        void operator()(cufftHandle plan, RealType* in, ComplexType* out){
          CHECK_CUDA(cufftExecD2Z(plan, in, out));
        }
        void operator()(cufftHandle plan, ComplexType* in, ComplexType* out){
          CHECK_CUDA(cufftExecZ2Z(plan, in, out, CUFFT_FORWARD));
        }
      };
      struct FFTExecuteBackward{
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

    static const std::string title() {
      return "CuFFT";
    }

    std::string getDeviceInfos() {
      auto ss = getCUDADeviceInformations(0);
      return ss.str();
    }

    void create() {
      CHECK_CUDA(cudaSetDevice(0));
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
  void makePlan(cufftHandle& plan, const std::array<size_t,3>& e){
    CHECK_CUDA(cufftPlan3d(&plan, e[0], e[1], e[2], FFTType));
  }
  template<cufftType FFTType>
  void makePlan(cufftHandle& plan, const std::array<size_t,1>& e){
    CHECK_CUDA(cufftPlan1d(&plan, e[0], FFTType, 1));
  }
  template<cufftType FFTType>
  void makePlan(cufftHandle& plan, const std::array<size_t,2>& e){
    CHECK_CUDA(cufftPlan2d(&plan, e[0], e[1], FFTType));
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
  template<typename TFFT, // see fft_abstract.hpp (FFT_Inplace_Real, ...)
           typename TPrecision, // double, float
           size_t   NDim // 1..3
           >
  struct CuFFTImpl {
    using Extent = std::array<size_t,NDim>;
    using Types  = typename traits::Types<TPrecision>;
    using ComplexType = typename Types::ComplexType;
    using RealType = typename Types::RealType;
    using FFTExecuteForward  = typename Types::FFTExecuteForward;
    using FFTExecuteBackward = typename Types::FFTExecuteBackward;

    static constexpr
     bool IsInplace = TFFT::IsInplace;
    static constexpr
     bool IsComplex = TFFT::IsComplex;
    static constexpr
     bool Padding = IsInplace && IsComplex==false && NDim>1;
    static constexpr
     cufftType FFTForward  = IsComplex ? Types::FFTComplex : Types::FFTForward;
    static constexpr
     cufftType FFTBackward = IsComplex ? Types::FFTComplex : Types::FFTBackward;

    using RealOrComplexType  = typename std::conditional<IsComplex,ComplexType,RealType>::type;

    size_t n_;        // =[1]*..*[dim]
    size_t n_padded_; // =[1]*..*[dim-1]*([dim]/2+1)
    Extent extents_;

    cufftHandle        plan_           = 0;
    RealOrComplexType* data_           = nullptr;
    ComplexType*       data_transform_ = nullptr; // intermediate buffer
    size_t             data_size_;
    size_t             data_transform_size_;
    bool               use64bit_       = false;

    CuFFTImpl(const Extent& cextents) {
      extents_ = interpret_as::column_major(cextents);
      n_ = std::accumulate(extents_.begin(),
                           extents_.end(),
                           static_cast<size_t>(1),
                           std::multiplies<size_t>());
      if(Padding)
        n_padded_ = n_ / extents_[NDim-1] * (extents_[NDim-1]/2 + 1);

      data_size_ = ( Padding ? 2*n_padded_ : n_ ) * sizeof(RealOrComplexType);
      data_transform_size_ = IsInplace ? 0 : n_ * sizeof(ComplexType);
      // There are some additional restrictions when using 64-bit API of cufft
      // see http://docs.nvidia.com/cuda/cufft/#unique_1972991535
      if(data_size_ >= (1ull<<32) || data_transform_size_ >= (1ull<<32))
        use64bit_ = true;
    }

    ~CuFFTImpl() {
      destroy();
    }

    /**
     * Returns allocated memory on device for FFT
     */
    size_t getAllocSize() {
      return data_size_ + data_transform_size_;
    }

    /**
     * Returns estimated allocated memory on device for FFT plan
     */
    size_t getPlanSize() {
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
        size2 = estimateAllocSize64<FFTBackward>(plan_,extents_);
      else{
        size2 = estimateAllocSize<FFTBackward>(plan_,extents_);
      }
      CHECK_CUDA(cufftDestroy(plan_));
      plan_=0;

      // check available GPU memory
      size_t mem_free=0, mem_tot=0;
      CHECK_CUDA( cudaMemGetInfo(&mem_free, &mem_tot) );
      size_t wanted = std::max(size1,size2) + data_size_ + data_transform_size_;
      if(mem_free<wanted) {
        std::stringstream ss;
        ss << mem_free << "<" << wanted << " (bytes)";
        throw std::runtime_error("Not enough GPU memory available. "+ss.str());
      }
      return std::max(size1,size2);
    }

    // --- next methods are benchmarked ---

    void malloc() {
      CHECK_CUDA(cudaMalloc(&data_, data_size_));
      if(!data_) throw std::runtime_error("Could not allocate GPU input buffer.");

      if(IsInplace) {
        data_transform_ = reinterpret_cast<ComplexType*>(data_);
      }else{
        CHECK_CUDA(cudaMalloc(&data_transform_, data_transform_size_));
        if(!data_transform_) throw std::runtime_error("Could not allocate GPU buffer for outplace transform.");
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
    void init_backward() {
      if(IsComplex==false){
        CHECK_CUDA(cufftDestroy(plan_));
        plan_=0;
        if(use64bit_)
          makePlan64<FFTBackward>(plan_, extents_);
        else
          makePlan<FFTBackward>(plan_, extents_);
      }
    }

    void execute_forward() {
      FFTExecuteForward()(plan_, data_, data_transform_);
    }

    void execute_backward() {
      FFTExecuteBackward()(plan_, data_transform_, data_);
    }

    template<typename THostData>
    void upload(THostData* input) {
      if(Padding) { // real + inplace + ndim>1
        size_t w      = extents_[NDim-1] * sizeof(THostData);
        size_t h      = n_ * sizeof(THostData) / w;
        size_t pitch  = (extents_[NDim-1]/2+1) * sizeof(ComplexType);
        CHECK_CUDA(cudaMemcpy2D(data_, pitch, input, w, w, h, cudaMemcpyHostToDevice));
      }else{
        CHECK_CUDA(cudaMemcpy(data_, input, data_size_, cudaMemcpyHostToDevice));
      }
    }

    template<typename THostData>
    void download(THostData* output) {
      if(Padding) { // real + inplace + ndim>1
        size_t w      = extents_[NDim-1] * sizeof(THostData);
        size_t h      = n_ * sizeof(THostData) / w;
        size_t pitch  = (extents_[NDim-1]/2+1) * sizeof(ComplexType);
        CHECK_CUDA(cudaMemcpy2D(output, w, data_, pitch, w, h, cudaMemcpyDeviceToHost));
      }else{
        CHECK_CUDA(cudaMemcpy(output, data_, data_size_, cudaMemcpyDeviceToHost));
      }
    }

    void destroy() {
      CHECK_CUDA( cudaFree(data_) );
      data_=nullptr;
      if(IsInplace==false && data_transform_) {
        CHECK_CUDA( cudaFree(data_transform_) );
        data_transform_ = nullptr;
      }
      if(plan_) {
        CHECK_CUDA( cufftDestroy(plan_) );
        plan_=0;
      }
    }
  };

  typedef gearshifft::FFT<gearshifft::FFT_Inplace_Real, CuFFTImpl, TimerGPU> Inplace_Real;
  typedef gearshifft::FFT<gearshifft::FFT_Outplace_Real, CuFFTImpl, TimerGPU> Outplace_Real;
  typedef gearshifft::FFT<gearshifft::FFT_Inplace_Complex, CuFFTImpl, TimerGPU> Inplace_Complex;
  typedef gearshifft::FFT<gearshifft::FFT_Outplace_Complex, CuFFTImpl, TimerGPU> Outplace_Complex;

} // namespace CuFFT
} // namespace gearshifft


#endif /* CUFFT_HPP_ */
