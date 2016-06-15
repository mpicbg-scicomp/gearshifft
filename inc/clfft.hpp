#ifndef CLFFT_HPP_
#define CLFFT_HPP_

#include "application.hpp"
#include "timer.hpp"
#include "fft_abstract.hpp"
#include "fixture_test_suite.hpp"
#include "clfft_helper.hpp"
#include <clFFT.h>
#include <stdexcept>

namespace gearshifft
{
namespace ClFFT
{
  namespace traits
  {
    template< typename T_Precision >
    struct Types;

    template<>
    struct Types<float>
    {
        using ComplexType = cl_float2;
        using RealType = cl_float;
    };

    template<>
    struct Types<double>
    {
        using ComplexType = cl_double2;
        using RealType = cl_double;
    };

    template< typename TPrecision=float >
    struct FFTPrecision: std::integral_constant< clfftPrecision, CLFFT_SINGLE >{};
    template<>
    struct FFTPrecision<double>: std::integral_constant< clfftPrecision, CLFFT_DOUBLE >{};

    template< bool IsComplex=true >
    struct FFTLayout {
      static constexpr clfftLayout value = CLFFT_COMPLEX_INTERLEAVED;
      static constexpr clfftLayout value_transformed = CLFFT_COMPLEX_INTERLEAVED;
    };
    template<>
    struct FFTLayout<false> {
      static constexpr clfftLayout value = CLFFT_REAL;
      static constexpr clfftLayout value_transformed = CLFFT_HERMITIAN_INTERLEAVED;
    };

    template< bool T_isInplace=true >
    struct FFTInplace: std::integral_constant< clfftResultLocation, CLFFT_INPLACE >{};
    template<>
    struct FFTInplace<false>: std::integral_constant< clfftResultLocation, CLFFT_OUTOFPLACE >{};
  } // traits

  struct Context {
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_device_id device_used = 0;
    cl_context ctx = 0;

    static const std::string title() {
      return "ClFFT";
    }

    std::string getDeviceInfos() {
      assert(device_used);
      auto ss = getClDeviceInformations(device_used);
      return ss.str();
    }

    void create() {
      cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
      cl_int err = CL_SUCCESS;
      findClDevice(CL_DEVICE_TYPE_GPU, &platform, &device);
      device_used = device;
      props[1] = (cl_context_properties)platform;
      ctx = clCreateContext( props, 1, &device, nullptr, nullptr, &err );
      CHECK_CL(err);
      clfftSetupData fftSetup;
      CHECK_CL(clfftInitSetupData(&fftSetup));
      CHECK_CL(clfftSetup(&fftSetup));
    }

    void destroy() {
      if(ctx) {
        CHECK_CL( clfftTeardown( ) );
        CHECK_CL(clReleaseContext( ctx ));
        CHECK_CL(clReleaseDevice(device));
        device = 0;
        ctx = 0;
      }
    }
  };

  /**
   * Plan Creator depending on FFT transform type.
   */
  template<clfftDim FFTDim, size_t Ndim>
  constexpr void makePlan(cl_context ctx, clfftPlanHandle& plan, const std::array<unsigned,Ndim>& e){
    size_t clLengths[3] = {e[0], Ndim==2?e[1]:1, Ndim==3?e[2]:1};
    CHECK_CL(clfftCreateDefaultPlan(&plan, ctx, FFTDim, clLengths));
  }

  /**
   * ClFFT plan and execution class.
   *
   * This class handles:
   * - {1D, 2D, 3D} x {R2C, C2R, C2C} x {inplace, outplace} x {float, double}.
   */
  template<typename TFFT, // see fft_abstract.hpp (FFT_Inplace_Real, ...)
           typename TPrecision, // double, float
           size_t   NDim // 1..3
           >
  struct ClFFTImpl
  {
    using ComplexType = typename traits::Types<TPrecision>::ComplexType;
    using RealType = typename traits::Types<TPrecision>::RealType;
    using Extent = std::array<unsigned,NDim>;
    static constexpr auto IsInplace = TFFT::IsInplace;
    static constexpr auto IsComplex = TFFT::IsComplex;
    static constexpr auto Padding = IsInplace && IsComplex==false;
    static constexpr clfftDim FFTDim = NDim==1 ? CLFFT_1D : NDim==2 ? CLFFT_2D : CLFFT_3D;
    using RealOrComplexType = typename std::conditional<IsComplex,ComplexType,RealType>::type;

    size_t n_;
    size_t n_padded_;
    Extent extents_;

    Context context_;
    cl_command_queue queue_ = 0;
    clfftPlanHandle plan_   = 0;
    cl_mem data_            = 0;
    cl_mem data_transform_  = 0; // intermediate buffer
    size_t data_size_       = 0;
    size_t data_transform_size_ = 0;
    size_t w;
    size_t h;
    size_t pitch;
    size_t region[3];
    size_t offset[3] = {0, 0, 0};
    size_t strides[3] = {1};
    size_t transform_strides[3] = {1};
    size_t dist;
    size_t transform_dist;

    ClFFTImpl(const Extent& cextents)
      : extents_(cextents)
    {
      cl_int err;
      context_ = Application<Context>::getContext();
      if(context_.ctx==0)
        throw std::runtime_error("Context has not been created.");
      queue_ = clCreateCommandQueue( context_.ctx, context_.device, 0, &err );
      CHECK_CL(err);

      n_ = std::accumulate(extents_.begin(), extents_.end(), 1, std::multiplies<unsigned>());
      if(Padding){
        n_padded_ = n_ / extents_[0] * (extents_[0]/2 + 1);
        w      = extents_[0] * sizeof(RealType);
        h      = n_ / extents_[0];
        pitch  = (extents_[0]/2+1) * sizeof(ComplexType);
        region[0] = w; // in bytes
        region[1] = h; // in counts (OpenCL1.1 is wrong here saying in bytes)
        region[2] = 1; // in counts (same)
        strides[1] = 2*(extents_[0]/2+1);
        strides[2] = 2 * n_padded_ / extents_[NDim-1];
        transform_strides[1] = extents_[0]/2+1;
        transform_strides[2] = n_padded_ / extents_[NDim-1];
        dist = 2 * n_padded_;
        transform_dist = n_padded_;
      }

      data_size_ = ( Padding ? 2*n_padded_*sizeof(RealType) : n_ * sizeof(RealOrComplexType) );
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
      CHECK_CL(clfftGetTmpBufSize( plan_, &size1 ));
      init_backward();
      CHECK_CL(clfftGetTmpBufSize( plan_, &size2 ));
      CHECK_CL(clfftDestroyPlan( &plan_ ));
      return std::max(size1,size2);
    }

    // --- next methods are benchmarked ---

    void malloc() {
      cl_int err;
      data_ = clCreateBuffer( context_.ctx,
                              CL_MEM_READ_WRITE,
                              data_size_,
                              nullptr, // host pointer @todo
                              &err );
      //std::cout << " data " << data_size_ << std::endl;
      if(IsInplace==false){
        data_transform_ = clCreateBuffer( context_.ctx,
                                          CL_MEM_READ_WRITE,
                                          data_transform_size_,
                                          nullptr, // host pointer
                                          &err );
        //std::cout << " transform " <<data_transform_size_ << std::endl;
      }
    }

    // create FFT plan handle
    void init_forward() {
      makePlan<FFTDim>(context_.ctx, plan_, extents_);
      CHECK_CL(clfftSetPlanPrecision(plan_, traits::FFTPrecision<TPrecision>::value));
      CHECK_CL(clfftSetLayout(plan_,
                                traits::FFTLayout<IsComplex>::value,
                                traits::FFTLayout<IsComplex>::value_transformed));
      CHECK_CL(clfftSetResultLocation(plan_, traits::FFTInplace<IsInplace>::value));
      if(Padding){
        CHECK_CL(clfftSetPlanInStride(plan_, FFTDim, strides));
        CHECK_CL(clfftSetPlanOutStride(plan_, FFTDim, transform_strides));
        CHECK_CL(clfftSetPlanDistance(plan_, dist, transform_dist));
      }
      CHECK_CL(clfftBakePlan(plan_,
                               1, // number of queues
                               &queue_,
                               nullptr, // callback
                               nullptr)); // user data
    }

    // recreates plan if needed
    void init_backward() {
      if(IsComplex==false){
        CHECK_CL(clfftSetLayout(plan_,
                                traits::FFTLayout<IsComplex>::value_transformed,
                                traits::FFTLayout<IsComplex>::value));
        if(Padding){
          CHECK_CL(clfftSetPlanOutStride(plan_, FFTDim, strides));
          CHECK_CL(clfftSetPlanInStride(plan_, FFTDim, transform_strides));
          CHECK_CL(clfftSetPlanDistance(plan_, transform_dist, dist));
        }

        CHECK_CL(clfftBakePlan(plan_,
                                 1, // number of queues
                                 &queue_,
                                 0, // callback
                                 0)); // user data
      }
    }

    void execute_forward() {
      CHECK_CL(clfftEnqueueTransform(plan_,
                                       CLFFT_FORWARD,
                                       1, // numQueuesAndEvents
                                       &queue_,
                                       0, // numWaitEvents
                                       0, // waitEvents
                                       0, // outEvents
                                       &data_,  // input
                                       IsInplace ? &data_ : &data_transform_, // output
                                       0)); // tmpBuffer
      CHECK_CL(clFinish(queue_));
    }
    void execute_backward() {
      CHECK_CL(clfftEnqueueTransform(plan_,
                                       CLFFT_BACKWARD,
                                       1, // numQueuesAndEvents
                                       &queue_,
                                       0, // numWaitEvents
                                       nullptr, // waitEvents
                                       nullptr, // outEvents
                                       IsInplace ? &data_ : &data_transform_, // input
                                       IsInplace ? &data_ : &data_, // output
                                       nullptr)); // tmpBuffer
      CHECK_CL(clFinish(queue_));
    }
    template<typename THostData>
    void upload(THostData* input) {
      if(Padding && NDim>1)
      {
        //printf("pitch=%zu w=%zu h=%zu\n", pitch, w, h);
        CHECK_CL(clEnqueueWriteBufferRect( queue_,
                                             data_,
                                             CL_TRUE, // blocking_write
                                             offset, // buffer origin
                                             offset, // host origin
                                             region,
                                             pitch, // buffer row pitch
                                             0, // buffer slice pitch
                                             0, // host row pitch
                                             0, // host slice pitch
                                             input,
                                             0, // num_events_in_wait_list
                                             nullptr, // event_wait_list
                                             nullptr )); // event
      }else{
        CHECK_CL(clEnqueueWriteBuffer( queue_,
                                         data_,
                                         CL_TRUE, // blocking_write
                                         0, // offset
                                         Padding ? n_*sizeof(RealType) : data_size_,
                                         input,
                                         0, // num_events_in_wait_list
                                         nullptr, // event_wait_list
                                         nullptr )); // event
      }
    }
    template<typename THostData>
    void download(THostData* output) {
      if(Padding && NDim>1)
      {
        CHECK_CL(clEnqueueReadBufferRect( queue_,
                                            data_,
                                            CL_TRUE, // blocking_write
                                            offset, // buffer origin
                                            offset, // host origin
                                            region,
                                            pitch, // buffer row pitch
                                            0, // buffer slice pitch
                                            0, // host row pitch
                                            0, // host slice pitch
                                            output,
                                            0, // num_events_in_wait_list
                                            nullptr, // event_wait_list
                                            nullptr )); // event
      }else{
        CHECK_CL(clEnqueueReadBuffer( queue_,
                                        data_,
                                        CL_TRUE, // blocking_write
                                        0, // offset
                                        Padding ? n_*sizeof(RealType) : data_size_,
                                        output,
                                        0, // num_events_in_wait_list
                                        nullptr, // event_wait_list
                                        nullptr )); // event
      }
    }

    void destroy() {
      CHECK_CL( clFinish(queue_) );
      CHECK_CL( clReleaseMemObject( data_ ) );
      if(IsInplace==false)
        CHECK_CL( clReleaseMemObject( data_transform_ ) );

      CHECK_CL(clfftDestroyPlan( &plan_ ));
      CHECK_CL( clReleaseCommandQueue( queue_ ) );
      data_ = 0;
      data_transform_ = 0;
      plan_ = 0;
      queue_ = 0;
    }
  };

  typedef gearshifft::FFT<gearshifft::FFT_Inplace_Real, ClFFTImpl, TimerCPU> Inplace_Real;
  typedef gearshifft::FFT<gearshifft::FFT_Outplace_Real, ClFFTImpl, TimerCPU> Outplace_Real;
  typedef gearshifft::FFT<gearshifft::FFT_Inplace_Complex, ClFFTImpl, TimerCPU> Inplace_Complex;
  typedef gearshifft::FFT<gearshifft::FFT_Outplace_Complex, ClFFTImpl, TimerCPU> Outplace_Complex;

} // namespace ClFFT
} // gearshifft



#endif /* CLFFT_HPP_ */
