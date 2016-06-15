#ifndef CUFFT_HPP_
#define CUFFT_HPP_

#include "application.hpp"
#include "timer.hpp"
#include "fft_abstract.hpp"
#include "fixture_test_suite.hpp"
#include "cufft_helper.hpp"

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
          CHECK_CUFFT(cufftExecR2C(plan, in, out));
        }
        void operator()(cufftHandle plan, ComplexType* in, ComplexType* out){
          CHECK_CUFFT(cufftExecC2C(plan, in, out, CUFFT_FORWARD));
        }
      };
      struct FFTExecuteBackward{
        void operator()(cufftHandle plan, ComplexType* in, RealType* out){
          CHECK_CUFFT(cufftExecC2R(plan, in, out));
        }
        void operator()(cufftHandle plan, ComplexType* in, ComplexType* out){
          CHECK_CUFFT(cufftExecC2C(plan, in, out, CUFFT_INVERSE));
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
          CHECK_CUFFT(cufftExecD2Z(plan, in, out));
        }
        void operator()(cufftHandle plan, ComplexType* in, ComplexType* out){
          CHECK_CUFFT(cufftExecZ2Z(plan, in, out, CUFFT_FORWARD));
        }
      };
      struct FFTExecuteBackward{
        void operator()(cufftHandle plan, ComplexType* in, RealType* out){
          CHECK_CUFFT(cufftExecZ2D(plan, in, out));
        }
        void operator()(cufftHandle plan, ComplexType* in, ComplexType* out){
          CHECK_CUFFT(cufftExecZ2Z(plan, in, out, CUFFT_INVERSE));
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
  size_t estimateAllocSize(const std::array<unsigned,NDim>& e, cufftHandle& plan)
  {
    size_t s=0;
    if(NDim==1){
      CHECK_CUFFT( cufftGetSize1d(plan, e[0], FFTType, 1, &s) );
    }
    if(NDim==2){
      CHECK_CUFFT( cufftGetSize2d(plan, e[0], e[1], FFTType, &s) );
    }
    if(NDim==3){
      CHECK_CUFFT( cufftGetSize3d(plan, e[0], e[1], e[2], FFTType, &s) );
    }
    return s;
  }
  /**
   * Plan Creator depending on FFT transform type (CUFFT_R2C, ...).
   */
  template<cufftType FFTType>
  void makePlan(cufftHandle& plan, const std::array<unsigned,3>& e){
    CHECK_CUFFT(cufftPlan3d(&plan, e[0], e[1], e[2], FFTType));
  }
  template<cufftType FFTType>
  void makePlan(cufftHandle& plan, const std::array<unsigned,1>& e){
    CHECK_CUFFT(cufftPlan1d(&plan, e[0], FFTType, 1));
  }
  template<cufftType FFTType>
  void makePlan(cufftHandle& plan, const std::array<unsigned,2>& e){
    CHECK_CUFFT(cufftPlan2d(&plan, e[0], e[1], FFTType));
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
  struct CuFFTImpl
  {
    using Extent = std::array<unsigned,NDim>;
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

    CuFFTImpl(const Extent& cextents)
      : extents_(cextents)
    {
      n_ = std::accumulate(extents_.begin(),
                           extents_.end(),
                           1,
                           std::multiplies<unsigned>());
      if(Padding)
        n_padded_ = n_ / extents_[NDim-1] * (extents_[NDim-1]/2 + 1);

      data_size_ = ( Padding ? 2*n_padded_ : n_ ) * sizeof(RealOrComplexType);
      data_transform_size_ = IsInplace ? 0 : n_ * sizeof(ComplexType);
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
      init_forward();
      size1 = estimateAllocSize<FFTForward>(extents_,plan_);
      init_backward();
      size2 = estimateAllocSize<FFTBackward>(extents_,plan_);
      CHECK_CUFFT( cufftDestroy(plan_) );
      return std::max(size1,size2);
    }

    // --- next methods are benchmarked ---

    void malloc() {
      if(IsInplace){
        CHECK_CUDA(cudaMalloc(&data_, data_size_));
        data_transform_ = reinterpret_cast<ComplexType*>(data_);
      }else{
        CHECK_CUDA(cudaMalloc(&data_, data_size_));
        CHECK_CUDA(cudaMalloc(&data_transform_, data_transform_size_));
      }
    }

    // create FFT plan handle
    void init_forward() {
      makePlan<FFTForward>(plan_, extents_);
    }

    // recreates plan if needed
    void init_backward() {
      if(IsComplex==false){
        CHECK_CUFFT(cufftDestroy(plan_));
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
      if(Padding) // real + inplace + ndim>1
      {
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
      if(Padding) // real + inplace + ndim>1
      {
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
      if(IsInplace==false)
        CHECK_CUDA( cudaFree(data_transform_) );
      CHECK_CUFFT( cufftDestroy(plan_) );
    }
  };

  typedef gearshifft::FFT<gearshifft::FFT_Inplace_Real, CuFFTImpl, TimerGPU> Inplace_Real;
  typedef gearshifft::FFT<gearshifft::FFT_Outplace_Real, CuFFTImpl, TimerGPU> Outplace_Real;
  typedef gearshifft::FFT<gearshifft::FFT_Inplace_Complex, CuFFTImpl, TimerGPU> Inplace_Complex;
  typedef gearshifft::FFT<gearshifft::FFT_Outplace_Complex, CuFFTImpl, TimerGPU> Outplace_Complex;

} // namespace CuFFT
} // namespace gearshifft


#endif /* CUFFT_HPP_ */
