#ifndef ROCFFT_HPP_
#define ROCFFT_HPP_

#include "core/application.hpp"
#include "core/timer_hip.hpp"
#include "core/fft.hpp"
#include "core/types.hpp"
#include "core/traits.hpp"
#include "core/context.hpp"

#include "rocfft_helper.hpp"

#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <rocfft.h>

#include <array>
#include <regex>

namespace gearshifft {
namespace RocFFT {


  namespace traits{

    template<typename T_Precision=float>
    struct Types
    {
        using ComplexType = std::complex<float>;
        using RealType = float;

        static constexpr rocfft_transform_type_e FFTRealForward = rocfft_transform_type_real_forward;
        static constexpr rocfft_transform_type_e FFTRealInverse = rocfft_transform_type_real_inverse;

        static constexpr rocfft_transform_type_e FFTComplexForward = rocfft_transform_type_complex_forward;
        static constexpr rocfft_transform_type_e FFTComplexInverse = rocfft_transform_type_complex_inverse;


    };

    template<>
    struct Types<double>
    {
        using ComplexType = std::complex<double>;
        using RealType = double;

        static constexpr rocfft_transform_type_e FFTRealForward = rocfft_transform_type_real_forward;
        static constexpr rocfft_transform_type_e FFTRealInverse = rocfft_transform_type_real_inverse;

        static constexpr rocfft_transform_type_e FFTComplexForward = rocfft_transform_type_complex_forward;
        static constexpr rocfft_transform_type_e FFTComplexInverse = rocfft_transform_type_complex_inverse;


    };

  }  // namespace traits

  /**
   * HIP context create() and destroy(). Time is benchmarked.
   */
    struct RocFFTContext : public ContextDefault<> {

        int device = 0;

        static const std::string title() {
            return "RocFFT";
        }

        static std::string get_device_list() {
            std::stringstream msg;
            listHipDevices(msg);
            return msg.str();
        }

        std::string get_used_device_properties() {
            std::stringstream ss;
            getHIPDeviceInformations(device,ss);
            return ss.str();
        }

        void create() {
            const std::string options_devtype = options().getDevice();
            device = atoi(options_devtype.c_str());
            int nrdev=0;
            CHECK_HIP(hipGetDeviceCount(&nrdev));
            assert(nrdev>0);
            if(device<0 || device>=nrdev)
                device = 0;
            CHECK_HIP(hipSetDevice(device));
        }

        void destroy() {
            CHECK_HIP(hipDeviceReset());
        }
    };


  /**
   * Estimates memory required/reserved by rocfft plan to perform the transformation

   */
  size_t estimateAllocSize(rocfft_plan& plan)
  {
      size_t s=0;
      if(plan != 0){
          CHECK_HIP( rocfft_plan_get_work_buffer_size(plan, &s) );}

      return s;
  }

  


  /**
   * RocFFT plan and execution class.
   *
   * This class handles:
   * - {1D, 2D, 3D} x {R2C, C2R, C2C} x {inplace, outplace} x {float, double}.
   */
  template<typename TFFT, // see fft.hpp (FFT_Inplace_Real, ...)
           typename TPrecision, // double, float
           size_t   NDim // 1..3
           >
  struct RocFFTImpl {
    using Extent = std::array<size_t,NDim>;
    using Types  = typename traits::Types<TPrecision>;
    using ComplexType = typename Types::ComplexType;
    using RealType = typename Types::RealType;

    static constexpr
     bool IsInplace = TFFT::IsInplace;
    static constexpr
     bool IsComplex = TFFT::IsComplex;
    // static constexpr
    //  bool IsHalf = std::is_same<TPrecision, float16>::value;
      static constexpr
     bool IsSingle = std::is_same<TPrecision, float>::value;
    static constexpr
     bool IsInplaceReal = IsInplace && IsComplex==false;
    static constexpr
     rocfft_transform_type FFTForward  = IsComplex ? Types::FFTComplexForward : Types::FFTRealForward;
    static constexpr
     rocfft_transform_type FFTInverse = IsComplex ? Types::FFTComplexInverse : Types::FFTRealInverse;
    static constexpr
    rocfft_precision FFTPrecision = IsSingle ? rocfft_precision_single : rocfft_precision_double;
      static constexpr
    rocfft_result_placement FFTPlacement = IsInplace ? rocfft_placement_inplace : rocfft_placement_notinplace;



    using value_type  = typename std::conditional<IsComplex,ComplexType,RealType>::type;

    /// extents of the FFT input data
    Extent extents_   = {{0}};
    /// extents of the FFT complex data (=FFT(input))
    Extent extents_complex_ = {{0}};
    /// product of corresponding extents
    size_t n_         = 0;
    /// product of corresponding extents
    size_t n_complex_ = 0;

    rocfft_plan  plan_              = 0;
    rocfft_execution_info plan_info_= nullptr;
      void * work_buffer_           = nullptr;
      value_type*  data_              = nullptr;
    ComplexType* data_complex_      = nullptr;
    /// size in bytes of FFT input data
    size_t       data_size_         = 0;
    /// size in bytes of FFT(input) for out-of-place transforms
    size_t       data_complex_size_ = 0;

    size_t       work_buffer_size_allocated_         = 0;

    RocFFTImpl(const Extent& cextents) {
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

      CHECK_HIP(rocfft_setup());
    }

    ~RocFFTImpl() {
      destroy();
      CHECK_HIP(rocfft_cleanup());
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
      size_t size1 = 0; // size forward trafo
      size_t size2 = 0; // size inverse trafo


      size1 = work_buffer_size_allocated_ > 0 ? work_buffer_size_allocated_ : estimateAllocSize(plan_);
      size2 = size1;

      // check available GPU memory
      size_t mem_free=0, mem_tot=0;
      CHECK_HIP( hipMemGetInfo(&mem_free, &mem_tot) );
      size_t wanted = std::max(size1,size2) + data_size_ + data_complex_size_;
      if(mem_free<wanted) {
        std::stringstream ss;
        ss << mem_free << "<" << wanted << " (bytes)";
        throw std::runtime_error("Not enough GPU memory available. "+ss.str());
      }
      size_t total_mem = 95*getMemorySize()/100; // keep some memory available, otherwise an out-of-memory killer becomes more likely
      if(total_mem < 2*data_size_) { // includes host input buffers
        std::stringstream ss;
        ss << total_mem << "<" << 2*data_size_ << " (bytes)";
        throw std::runtime_error("Host data exceeds physical memory. "+ss.str());
      }
      return std::max(size1,size2);
    }
/**
     * Free work buffer for rocfft
     */
      void free_work_buffer(){
          if(work_buffer_){
              CHECK_HIP( hipFree(work_buffer_) );
              work_buffer_ = nullptr;
          }
          work_buffer_size_allocated_ = 0;

      }

/**
     * Allocate work buffer for rocfft on HIP device (free it first if requred, do nothing if work buffer is already large enough)
     */
      void allocate_work_buffer(std::size_t _bytes){
          if(_bytes > work_buffer_size_allocated_){
              free_work_buffer();
              CHECK_HIP(hipMalloc(&work_buffer_, _bytes));
              work_buffer_size_allocated_ = _bytes;
          }
      }

    // --- next methods are benchmarked ---

    /**
     * Allocate buffers on HIP device
     */
    void allocate() {
      CHECK_HIP(hipMalloc(&data_, data_size_));

      if(IsInplace) {
        data_complex_ = reinterpret_cast<ComplexType*>(data_);
      }else{
        CHECK_HIP(hipMalloc(&data_complex_, data_complex_size_));
      }
    }

      // create FFT plan handle
      void init_forward() {

          CHECK_HIP(rocfft_plan_create(&plan_,
                             FFTPlacement,
                             FFTForward,
                             FFTPrecision,
                                       extents_.size(), extents_.data(), 1, nullptr));


          size_t workBufferSize = 0;
          CHECK_HIP(rocfft_plan_get_work_buffer_size(plan_, &workBufferSize));

          if(!plan_info_)
              CHECK_HIP(rocfft_execution_info_create(&plan_info_));

          if(work_buffer_size_allocated_ < workBufferSize)
          {
              allocate_work_buffer(workBufferSize);
              CHECK_HIP(rocfft_execution_info_set_work_buffer(plan_info_, work_buffer_, workBufferSize));
          }


      }

      // recreates plan if needed
      void init_inverse() {

          CHECK_HIP(rocfft_plan_create(&plan_,
                             FFTPlacement,
                             FFTInverse,
                             FFTPrecision,
                                       extents_.size(), extents_.data(), 1, nullptr));
          size_t workBufferSize = 0;
          CHECK_HIP(rocfft_plan_get_work_buffer_size(plan_, &workBufferSize));

          if(!plan_info_)
              CHECK_HIP(rocfft_execution_info_create(&plan_info_));

          if(work_buffer_size_allocated_ < workBufferSize)
          {
              allocate_work_buffer(workBufferSize);
              CHECK_HIP(rocfft_execution_info_set_work_buffer(plan_info_, work_buffer_, workBufferSize));
          }


      }

    void execute_forward() {

        CHECK_HIP(rocfft_execute(plan_, (void**) &data_, (void**) &data_complex_, plan_info_));

    }

    void execute_inverse() {

        CHECK_HIP(rocfft_execute(plan_, (void**) &data_complex_, (void**) &data_, plan_info_));

    }

    template<typename THostData>
    void upload(THostData* input) {

      if(IsInplaceReal && NDim>1) {
        size_t w      = extents_[NDim-1] * sizeof(THostData);
        size_t h      = n_ * sizeof(THostData) / w;
        size_t pitch  = (extents_[NDim-1]/2+1) * sizeof(ComplexType);
        CHECK_HIP(hipMemcpy2D(data_, pitch, input, w, w, h, hipMemcpyHostToDevice));
      }else{
        CHECK_HIP(hipMemcpy(data_, input, get_transfer_size(), hipMemcpyHostToDevice));
      }
    }

    template<typename THostData>
    void download(THostData* output) {

      if(IsInplaceReal && NDim>1) {
        size_t w      = extents_[NDim-1] * sizeof(THostData);
        size_t h      = n_ * sizeof(THostData) / w;
        size_t pitch  = (extents_[NDim-1]/2+1) * sizeof(ComplexType);
        CHECK_HIP(hipMemcpy2D(output, w, data_, pitch, w, h, hipMemcpyDeviceToHost));
      }else{
        CHECK_HIP(hipMemcpy(output, data_, get_transfer_size(), hipMemcpyDeviceToHost));
      }
    }

    void destroy() {
      CHECK_HIP( hipFree(data_) );
      data_=nullptr;
      if(IsInplace==false && data_complex_) {
        CHECK_HIP( hipFree(data_complex_) );
        data_complex_ = nullptr;
      }
      if(plan_) {
        CHECK_HIP( rocfft_plan_destroy(plan_) );
        plan_=0;
      }
      if(work_buffer_){
          CHECK_HIP( hipFree(work_buffer_) );
          work_buffer_ = nullptr;
      }
      if(plan_info_){
          CHECK_HIP(rocfft_execution_info_destroy(plan_info_));
          plan_info_ = nullptr;
      }
    }
  };

    using Inplace_Real = gearshifft::FFT<FFT_Inplace_Real,
                                         FFT_Plan_Reusable,
                                         RocFFTImpl,
                                         TimerGPU >;
    using Outplace_Real = gearshifft::FFT<FFT_Outplace_Real,
                                          FFT_Plan_Reusable,
                                          RocFFTImpl,
                                          TimerGPU >;
    using Inplace_Complex = gearshifft::FFT<FFT_Inplace_Complex,
                                            FFT_Plan_Reusable,
                                            RocFFTImpl,
                                            TimerGPU >;
    using Outplace_Complex = gearshifft::FFT<FFT_Outplace_Complex,
                                             FFT_Plan_Reusable,
                                             RocFFTImpl,
                                             TimerGPU >;

} // namespace RocFFT
} // namespace gearshifft


#endif /* ROCFFT_HPP_ */
