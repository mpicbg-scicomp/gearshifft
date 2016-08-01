#ifndef FFTW_HPP_
#define FFTW_HPP_

#include "application.hpp"
#include "timer.hpp"
#include "fft.hpp"
#include "benchmark_suite.hpp"
//#include "fftw_helper.hpp"

#include <array>
#include <fftw.h>
#include <vector_types.h>

namespace gearshifft {
namespace Fftw {
  namespace traits{
    template<typename T_Precision=float>
    struct Types
    {
      using ComplexType = fftwComplex;
      using RealType = fftwReal;
      static constexpr fftwType FFTForward = FFTW_R2C;
      static constexpr fftwType FFTComplex = FFTW_C2C;
      static constexpr fftwType FFTBackward = FFTW_C2R;

      struct FFTExecuteForward{
        void operator()(fftwHandle plan, RealType* in, ComplexType* out){
          CHECK_FFTW(fftwExecR2C(plan, in, out));
        }
        void operator()(fftwHandle plan, ComplexType* in, ComplexType* out){
          CHECK_FFTW(fftwExecC2C(plan, in, out, FFTW_FORWARD));
        }
      };
      struct FFTExecuteBackward{
        void operator()(fftwHandle plan, ComplexType* in, RealType* out){
          CHECK_FFTW(fftwExecC2R(plan, in, out));
        }
        void operator()(fftwHandle plan, ComplexType* in, ComplexType* out){
          CHECK_FFTW(fftwExecC2C(plan, in, out, FFTW_INVERSE));
        }
      };
    };

    template<>
    struct Types<double>
    {
      using ComplexType = fftwDoubleComplex;
      using RealType = fftwDoubleReal;
      static constexpr fftwType FFTForward = FFTW_D2Z;
      static constexpr fftwType FFTComplex = FFTW_Z2Z;
      static constexpr fftwType FFTBackward = FFTW_Z2D;

      struct FFTExecuteForward{
        void operator()(fftwHandle plan, RealType* in, ComplexType* out){
          CHECK_FFTW(fftwExecD2Z(plan, in, out));
        }
        void operator()(fftwHandle plan, ComplexType* in, ComplexType* out){
          CHECK_FFTW(fftwExecZ2Z(plan, in, out, FFTW_FORWARD));
        }
      };
      struct FFTExecuteBackward{
        void operator()(fftwHandle plan, ComplexType* in, RealType* out){
          CHECK_FFTW(fftwExecZ2D(plan, in, out));
        }
        void operator()(fftwHandle plan, ComplexType* in, ComplexType* out){
          CHECK_FFTW(fftwExecZ2Z(plan, in, out, FFTW_INVERSE));
        }
      };
    };

  }  // namespace traits

  /**
   * CUDA implicit context init and reset wrapper. Time is benchmarked.
   */
  struct Context {

    static const std::string title() {
      return "Fftw";
    }

    std::string getDeviceInfos() {
      //obtain number of processors
      return ;
    }

    void create() {
      //set number of threads
    }

    void destroy() {
      //not needed
    }
  };


  /**
   * Estimates memory reserved by fftw plan depending on FFT transform type
   * (FFTW_R2C, ...) and depending on number of dimensions {1,2,3}.
   */
  template<fftwType FFTType, size_t NDim>
  size_t estimateAllocSize(const std::array<unsigned,NDim>& e, fftwHandle& plan)
  {
    size_t s=0;
    if(NDim==1){
      CHECK_FFTW( fftwGetSize1d(plan, e[0], FFTType, 1, &s) );
    }
    if(NDim==2){
      CHECK_FFTW( fftwGetSize2d(plan, e[0], e[1], FFTType, &s) );
    }
    if(NDim==3){
      CHECK_FFTW( fftwGetSize3d(plan, e[0], e[1], e[2], FFTType, &s) );
    }
    return s;
  }
  /**
   * Plan Creator depending on FFT transform type (FFTW_R2C, ...).
   */
  template<fftwType FFTType>
  void makePlan(fftwHandle& plan, const std::array<unsigned,1>& e){
    CHECK_FFTW(fftwPlan1d(&plan, e[0], FFTType, 1));
  }
  template<fftwType FFTType>
  void makePlan(fftwHandle& plan, const std::array<unsigned,2>& e){
    CHECK_FFTW(fftwPlan2d(&plan, e[0], e[1], FFTType));
  }
  template<fftwType FFTType>
  void makePlan(fftwHandle& plan, const std::array<unsigned,3>& e){
    CHECK_FFTW(fftwPlan3d(&plan, e[0], e[1], e[2], FFTType));
  }
  

  /**
   * Fftw plan and execution class.
   *
   * This class handles:
   * - {1D, 2D, 3D} x {R2C, C2R, C2C} x {inplace, outplace} x {float, double}.
   */
  template<typename TFFT, // see fft_abstract.hpp (FFT_Inplace_Real, ...)
           typename TPrecision, // double, float
           size_t   NDim // 1..3
           >
  struct FftwImpl
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
     fftwType FFTForward  = IsComplex ? Types::FFTComplex : Types::FFTForward;
    static constexpr
     fftwType FFTBackward = IsComplex ? Types::FFTComplex : Types::FFTBackward;

    using RealOrComplexType  = typename std::conditional<IsComplex,ComplexType,RealType>::type;

    size_t n_;        // =[1]*..*[dim]
    size_t n_padded_; // =[1]*..*[dim-1]*([dim]/2+1)
    Extent extents_;

    fftwHandle        plan_           = 0;
    RealOrComplexType* data_           = nullptr;
    ComplexType*       data_transform_ = nullptr; // intermediate buffer
    size_t             data_size_;
    size_t             data_transform_size_;

    FftwImpl(const Extent& cextents)
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
      CHECK_FFTW( fftwDestroy(plan_) );
      return std::max(size1,size2);
    }

    // --- next methods are benchmarked ---

    void malloc() {
      //TODO: fftw_malloc
      
      // if(IsInplace){
      //   CHECK_CUDA(cudaMalloc(&data_, data_size_));
      //   data_transform_ = reinterpret_cast<ComplexType*>(data_);
      // }else{
      //   CHECK_CUDA(cudaMalloc(&data_, data_size_));
      //   CHECK_CUDA(cudaMalloc(&data_transform_, data_transform_size_));
      // }
    }

    // create FFT plan handle
    void init_forward() {
      makePlan<FFTForward>(plan_, extents_);
    }

    // recreates plan if needed
    void init_backward() {
      if(IsComplex==false){
        CHECK_FFTW(fftwDestroy(plan_));
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

      //TODO: unnecessary?
      
      // if(Padding) // real + inplace + ndim>1
      // {
      //   size_t w      = extents_[NDim-1] * sizeof(THostData);
      //   size_t h      = n_ * sizeof(THostData) / w;
      //   size_t pitch  = (extents_[NDim-1]/2+1) * sizeof(ComplexType);
      //   CHECK_CUDA(cudaMemcpy2D(data_, pitch, input, w, w, h, cudaMemcpyHostToDevice));
      // }else{
      //   CHECK_CUDA(cudaMemcpy(data_, input, data_size_, cudaMemcpyHostToDevice));
      // }

      
    }

    template<typename THostData>
    void download(THostData* output) {

      //TODO: unnecessary?
      
      // if(Padding) // real + inplace + ndim>1
      // {
      //   size_t w      = extents_[NDim-1] * sizeof(THostData);
      //   size_t h      = n_ * sizeof(THostData) / w;
      //   size_t pitch  = (extents_[NDim-1]/2+1) * sizeof(ComplexType);
      //   CHECK_CUDA(cudaMemcpy2D(output, w, data_, pitch, w, h, cudaMemcpyDeviceToHost));
      // }else{
      //   CHECK_CUDA(cudaMemcpy(output, data_, data_size_, cudaMemcpyDeviceToHost));
      // }
    }

    void destroy() {

      //TODO: destroy plan
      
      // CHECK_CUDA( cudaFree(data_) );
      // if(IsInplace==false)
      //   CHECK_CUDA( cudaFree(data_transform_) );
      // CHECK_FFTW( fftwDestroy(plan_) );
    }
  };

  typedef gearshifft::FFT<gearshifft::FFT_Inplace_Real, FftwImpl, TimerGPU> Inplace_Real;
  typedef gearshifft::FFT<gearshifft::FFT_Outplace_Real, FftwImpl, TimerGPU> Outplace_Real;
  typedef gearshifft::FFT<gearshifft::FFT_Inplace_Complex, FftwImpl, TimerGPU> Inplace_Complex;
  typedef gearshifft::FFT<gearshifft::FFT_Outplace_Complex, FftwImpl, TimerGPU> Outplace_Complex;

} // namespace Fftw
} // namespace gearshifft


#endif /* FFTW_HPP_ */
