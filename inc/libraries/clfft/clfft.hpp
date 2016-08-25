#ifndef CLFFT_HPP_
#define CLFFT_HPP_

#include "core/application.hpp"
#include "core/timer_opencl.hpp"
#include "core/fft.hpp"
#include "core/traits.hpp"

#include "clfft_helper.hpp"

#include <clFFT.h>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <boost/algorithm/string/predicate.hpp> // iequals
#include <boost/algorithm/string.hpp> // split, is_any_of

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
      std::shared_ptr<cl_event> event;

      static const std::string title() {
        return "ClFFT";
      }

      static const std::string getListDevices() {
        auto ss = listClDevices();
        return ss.str();
      }

      std::string getDeviceInfos() {
        assert(device_used);
        auto ss = getClDeviceInformations(device_used);
        return ss.str();
      }

      void create() {
        cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
        cl_int err = CL_SUCCESS;
        cl_device_type devtype = CL_DEVICE_TYPE_GPU;
        event = std::make_shared<cl_event>();

        const std::string options_devtype = Options::getInstance().getDevice();
        std::regex e("^([0-9]+):([0-9]+)$");
        if(std::regex_search(options_devtype, e)) {
          std::vector<std::string> token;
          boost::split(token, options_devtype, boost::is_any_of(":"));
          unsigned long id_platform = std::stoul(token[0].c_str());
          unsigned long id_device = std::stoul(token[1].c_str());
          getPlatformAndDeviceByID(&platform, &device, id_platform, id_device);
        }else{
          if(boost::iequals(options_devtype, "cpu")) // case insensitive compare
            devtype = CL_DEVICE_TYPE_CPU;
          else if(boost::iequals(options_devtype, "acc"))
            devtype = CL_DEVICE_TYPE_ACCELERATOR;
          else if(boost::iequals(options_devtype, "gpu"))
            devtype = CL_DEVICE_TYPE_GPU;
          else
            throw std::runtime_error("Unsupported device type");
          findClDevice(devtype, &platform, &device);
        }

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
          // @todo crashes at ClFFT/double/2187x625x729/Outplace_Complex with INVALID_EVENT on K20Xm
          //CHECK_CL(clReleaseEvent(*event));
          CHECK_CL(clReleaseContext( ctx ));
          CHECK_CL(clReleaseDevice(device));
          CHECK_CL( clfftTeardown( ) );
          device = 0;
          ctx = 0;
          event.reset();
        }
      }
    };

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
      using Extent = std::array<size_t,NDim>;
      static constexpr auto IsInplace = TFFT::IsInplace;
      static constexpr auto IsComplex = TFFT::IsComplex;
      static constexpr auto IsInplaceReal = IsInplace && IsComplex==false;
      static constexpr clfftDim FFTDim = NDim==1 ? CLFFT_1D : NDim==2 ? CLFFT_2D : CLFFT_3D;
      using RealOrComplexType = typename std::conditional<IsComplex,ComplexType,RealType>::type;

      Context context_;
      cl_command_queue queue_ = 0;
      clfftPlanHandle plan_   = 0;
      /// input buffer
      cl_mem data_            = 0;
      /// intermediate output buffer (transformed input values)
      cl_mem data_complex_    = 0;
      /// data size of input buffer
      size_t data_size_       = 0;
      /// data size of intermediate output buffer
      size_t data_complex_size_ = 0;

      /// total length (product of extents)
      size_t n_;
      /// FFT extents
      std::array<size_t, 3> extents_ = {{0}};
      /// strides setting in input data, for clfft kernel
      size_t strides_[3] = {1};
      /// strides setting in output data, for clfft kernel
      size_t strides_complex_[3] = {1};
      /// total length of input values, for clfft kernel
      size_t dist_;
      /// total length of output values, for clfft kernel
      size_t dist_complex_;
      // for padding in memory transfers
      size_t pitch_;
      size_t region_[3] = {0};
      size_t offset_[3] = {0};


      ClFFTImpl(const Extent& cextents) {
        cl_int err;
        context_ = Application<Context>::getContext();
        if(context_.ctx==0)
          throw std::runtime_error("Context has not been created.");
//        queue_ = clCreateCommandQueue( context_.ctx, context_.device, 0, &err );
        queue_ = clCreateCommandQueue( context_.ctx, context_.device, CL_QUEUE_PROFILING_ENABLE, &err );
        CHECK_CL(err);

        n_ = std::accumulate(cextents.begin(), cextents.end(), 1, std::multiplies<size_t>());

        auto cl_extents = interpret_as::row_major(cextents);
        std::copy(cl_extents.begin(), cl_extents.end(), extents_.begin());

        size_t n_half = n_ / extents_[0] * (extents_[0]/2 + 1);

        // strides_ for clfft kernel, if less than 3D, strides_ will be ignored
        if(IsInplaceReal) {
          strides_[1] = 2 * (extents_[0]/2+1);
          strides_[2] = 2 * n_half / extents_[NDim-1];
          dist_       = 2 * n_half;
        }else{
          strides_[1] = extents_[0];
          strides_[2] = n_ / extents_[NDim-1];
          dist_       = n_;
        }

        if(IsComplex==false){
          strides_complex_[1] = extents_[0]/2+1;
          strides_complex_[2] = n_half / extents_[NDim-1];
          dist_complex_       = n_half;
        }else{ // Complex, inplace, outplace
          strides_complex_[1] = extents_[0];
          strides_complex_[2] = n_ / extents_[NDim-1];
          dist_complex_       = n_;
        }
        // if memory transfer must pad reals
        if(IsInplaceReal && NDim>1) {
          size_t width  = extents_[0] * sizeof(RealType);
          size_t height = n_ / extents_[0];
          pitch_ = (extents_[0]/2+1) * sizeof(ComplexType);
          region_[0] = width; // in bytes
          region_[1] = height; // in counts (OpenCL1.1 is wrong here speaking of bytes)
          region_[2] = 1; // in counts (same)
        }

        data_size_ = dist_ * sizeof(RealOrComplexType);
        data_complex_size_ = ( IsInplace ? 0 : IsComplex ? n_ : n_half ) * sizeof(ComplexType);
      }

      ~ClFFTImpl() {
        destroy();
      }

      /**
       * Returns data to be transfered to and from device for FFT
       */
      size_t getTransferSize() {
        return IsInplaceReal ? n_*sizeof(RealType) : data_size_;
      }

      /**
       * Returns allocated memory on device for FFT
       */
      size_t getAllocSize() {
        return data_size_ + data_complex_size_;
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

      /**
       * Allocate buffers on OpenCL device
       */
      void malloc() {
        cl_int err;
        data_ = clCreateBuffer( context_.ctx,
                                CL_MEM_READ_WRITE,
                                data_size_,
                                nullptr, // host pointer @todo
                                &err );
        if(IsInplace==false){
          data_complex_ = clCreateBuffer( context_.ctx,
                                          CL_MEM_READ_WRITE,
                                          data_complex_size_,
                                          nullptr, // host pointer
                                          &err );
        }
      }

      /**
       * Create clfft forward plan with layout, precision, strides and distances
       */
      void init_forward() {
        CHECK_CL(clfftCreateDefaultPlan(&plan_, context_.ctx, FFTDim, extents_.data()));
        CHECK_CL(clfftSetPlanPrecision(plan_, traits::FFTPrecision<TPrecision>::value));
        CHECK_CL(clfftSetLayout(plan_,
                                traits::FFTLayout<IsComplex>::value,
                                traits::FFTLayout<IsComplex>::value_transformed));
        CHECK_CL(clfftSetResultLocation(plan_, traits::FFTInplace<IsInplace>::value));
        CHECK_CL(clfftSetPlanInStride(plan_, FFTDim, strides_));
        CHECK_CL(clfftSetPlanOutStride(plan_, FFTDim, strides_complex_));
        CHECK_CL(clfftSetPlanDistance(plan_, dist_, dist_complex_));
        CHECK_CL(clfftBakePlan(plan_,
                               1, // number of queues
                               &queue_,
                               nullptr, // callback
                               nullptr)); // user data
      }


      /**
       * If real-transform: create clfft backward plan with layout, precision, strides and distances
       */
      void init_backward() {
        if(IsComplex==false){
          CHECK_CL(clfftDestroyPlan( &plan_ ));
          CHECK_CL(clfftCreateDefaultPlan(&plan_, context_.ctx, FFTDim, extents_.data()));
          CHECK_CL(clfftSetPlanPrecision(plan_, traits::FFTPrecision<TPrecision>::value));
          CHECK_CL(clfftSetLayout(plan_,
                                  traits::FFTLayout<IsComplex>::value_transformed,
                                  traits::FFTLayout<IsComplex>::value));
          CHECK_CL(clfftSetPlanOutStride(plan_, FFTDim, strides_));
          CHECK_CL(clfftSetPlanInStride(plan_, FFTDim, strides_complex_));
          CHECK_CL(clfftSetPlanDistance(plan_, dist_complex_, dist_));
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
                                       context_.event.get(), // outEvents
                                       &data_,  // input
                                       IsInplace ? &data_ : &data_complex_, // output
                                       0)); // tmpBuffer
      }

      void execute_backward() {
        CHECK_CL(clfftEnqueueTransform(plan_,
                                       CLFFT_BACKWARD,
                                       1, // numQueuesAndEvents
                                       &queue_,
                                       0, // numWaitEvents
                                       nullptr, // waitEvents
                                       context_.event.get(), // outEvents
                                       IsInplace ? &data_ : &data_complex_, // input
                                       &data_, // output
                                       nullptr)); // tmpBuffer
      }

      template<typename THostData>
      void upload(THostData* input) {
        if(IsInplaceReal && NDim>1) {
          CHECK_CL(clEnqueueWriteBufferRect( queue_,
                                             data_,
                                             CL_FALSE, // blocking_write
                                             offset_, // buffer origin
                                             offset_, // host origin
                                             region_,
                                             pitch_, // buffer row pitch
                                             0, // buffer slice pitch
                                             0, // host row pitch
                                             0, // host slice pitch
                                             input,
                                             0, // num_events_in_wait_list
                                             nullptr, // event_wait_list
                                             context_.event.get() )); // event
        }else{
          CHECK_CL(clEnqueueWriteBuffer( queue_,
                                         data_,
                                         CL_FALSE, // blocking_write
                                         0, // offset
                                         getTransferSize(),
                                         input,
                                         0, // num_events_in_wait_list
                                         nullptr, // event_wait_list
                                         context_.event.get() )); // event
        }
      }

      template<typename THostData>
      void download(THostData* output) {
        if(IsInplaceReal && NDim>1) {
          CHECK_CL(clEnqueueReadBufferRect( queue_,
                                            data_,
                                            CL_FALSE, // blocking_write
                                            offset_, // buffer origin
                                            offset_, // host origin
                                            region_,
                                            pitch_, // buffer row pitch
                                            0, // buffer slice pitch
                                            0, // host row pitch
                                            0, // host slice pitch
                                            output,
                                            0, // num_events_in_wait_list
                                            nullptr, // event_wait_list
                                            context_.event.get() )); // event
        }else{
          CHECK_CL(clEnqueueReadBuffer( queue_,
                                        data_,
                                        CL_FALSE, // blocking_write
                                        0, // offset
                                        getTransferSize(),
                                        output,
                                        0, // num_events_in_wait_list
                                        nullptr, // event_wait_list
                                        context_.event.get() )); // event
        }
      }

      void destroy() {
        if(queue_) {
          CHECK_CL( clFinish(queue_) );
          if( data_ ) {
            CHECK_CL( clReleaseMemObject( data_ ) );
            data_ = 0;
            if(IsInplace==false && data_complex_){
              CHECK_CL( clReleaseMemObject( data_complex_ ) );
              data_complex_ = 0;
            }
          }
          if(plan_) {
            CHECK_CL(clfftDestroyPlan( &plan_ ));
            plan_ = 0;
          }
          CHECK_CL( clReleaseCommandQueue( queue_ ) );
          queue_ = 0;
        }
      }
    };

    typedef gearshifft::FFT<gearshifft::FFT_Inplace_Real, ClFFTImpl, TimerGPU<Context> > Inplace_Real;
    typedef gearshifft::FFT<gearshifft::FFT_Outplace_Real, ClFFTImpl, TimerGPU<Context> > Outplace_Real;
    typedef gearshifft::FFT<gearshifft::FFT_Inplace_Complex, ClFFTImpl, TimerGPU<Context> > Inplace_Complex;
    typedef gearshifft::FFT<gearshifft::FFT_Outplace_Complex, ClFFTImpl, TimerGPU<Context> > Outplace_Complex;
  } // namespace ClFFT
} // gearshifft



#endif /* CLFFT_HPP_ */
