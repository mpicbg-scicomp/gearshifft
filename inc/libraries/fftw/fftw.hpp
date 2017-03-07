#ifndef FFTW_HPP_
#define FFTW_HPP_

#include "core/types.hpp"
#include "core/application.hpp"
#include "core/timer.hpp"
#include "core/fft.hpp"
#include "core/benchmark_suite.hpp"
#include "get_memory_size.h"

#include <string.h>
#include <vector>
#include <array>
#include <thread>
#include <sstream>
#include <type_traits>
#include <fftw3.h>

namespace gearshifft {
namespace fftw {
    
  namespace traits{

    template <typename T>
    struct memory_api {};

    template <>
    struct memory_api<float> {

      static void* malloc(size_t nbytes) { return fftwf_malloc(nbytes); }

      static void free(void* p) { fftwf_free(p); }
    };

    template <>
    struct memory_api<double> {

      static void* malloc(size_t nbytes) { return fftw_malloc(nbytes); }

      static void free(void* p) { fftw_free(p); }
    };


    //http://www.fftw.org/fftw3_doc/Usage-of-Multi_002dthreaded-FFTW.html
    template <typename T>
    struct thread_api {};


    template <>
    struct thread_api<double> {

      static int init_threads() {
        int res = fftw_init_threads();
        return res;
      }

      static void plan_with_threads(int nthreads = -1,
                                    double timelimit=FFTW_NO_TIMELIMIT){

        int av_procs = std::thread::hardware_concurrency();

        if(nthreads<1 || nthreads>av_procs)
          nthreads = av_procs;

        fftw_plan_with_nthreads(nthreads);
        fftw_set_timelimit(timelimit);
      }

      static void cleanup_threads(){
        fftw_cleanup_threads();
      }

    };

    template <>
    struct thread_api<float> {


      static int init_threads() {
        int res = fftwf_init_threads();
        return res;
      }

      static void plan_with_threads(int nthreads = -1,
                                    double timelimit=FFTW_NO_TIMELIMIT){

        int av_procs = std::thread::hardware_concurrency();

        if(nthreads<1 || nthreads>av_procs)
          nthreads = av_procs;

        fftwf_plan_with_nthreads(nthreads);
        fftwf_set_timelimit(timelimit);
      }

      static void cleanup_threads(){
        fftwf_cleanup_threads();
      }

    };

    enum class fftw_direction {
      inverse = FFTW_BACKWARD,
      forward = FFTW_FORWARD
    };


    template<typename T_Precision=float>
    struct plan
    {
      using ComplexType = fftwf_complex;
      using RealType = float;
      using PlanType = fftwf_plan;

      static void execute(const PlanType _plan){
        fftwf_execute(_plan);
      }

      static void destroy(PlanType _plan){

        if(_plan)
          fftwf_destroy_plan(_plan);
        _plan = nullptr;

      }

      template <size_t NDims>
      static PlanType create(const std::array<std::size_t,NDims>& _shape,
                             RealType* _in,
                             ComplexType* _out,
                             fftw_direction _dir = fftw_direction::forward,
                             unsigned plan_flags = FFTW_MEASURE){

        std::array<int,NDims> converted;
        for(int i = 0;i < NDims;++i)
          converted[i] = _shape[i];

        PlanType value = fftwf_plan_dft_r2c(NDims,
                                            converted.data(),
                                            _in,
                                            _out,
                                            plan_flags );
        return value;
      }

      template <size_t NDims>
      static PlanType create(const std::array<std::size_t,NDims>& _shape,
                             ComplexType* _in,
                             RealType* _out,
                             fftw_direction _dir = fftw_direction::forward,
                             unsigned plan_flags = FFTW_MEASURE){

        std::array<int,NDims> converted;
        for(int i = 0;i < NDims;++i)
          converted[i] = _shape[i];

        PlanType value = fftwf_plan_dft_c2r(NDims,
                                            converted.data(),
                                            _in,
                                            _out,
                                            plan_flags );
        return value;
      }

      template <size_t NDims>
      static PlanType create(const std::array<std::size_t,NDims>& _shape,
                             ComplexType* _in,
                             ComplexType* _out,
                             fftw_direction _dir = fftw_direction::forward,
                             unsigned plan_flags = FFTW_MEASURE){

        std::array<int,NDims> converted;
        for(int i = 0;i < NDims;++i)
          converted[i] = _shape[i];


        PlanType value = fftwf_plan_dft(NDims,
                                        converted.data(),
                                        _in,
                                        _out,
                                        static_cast<int>(_dir),
                                        plan_flags );
        return value;
      }

    };

    template<>
    struct plan<double>
    {
      using ComplexType = fftw_complex;
      using RealType = double;
      using PlanType = fftw_plan;

      static void execute(const PlanType _plan){
        fftw_execute(_plan);
      }

      static void destroy(PlanType _plan){

        if(_plan)
          fftw_destroy_plan(_plan);
        _plan = nullptr;

      }

      // //ND
      template <size_t NDims>
      static PlanType create(const std::array<std::size_t,NDims>& _shape,
                             RealType* _in,
                             ComplexType* _out,
                             fftw_direction _dir = fftw_direction::forward,
                             unsigned plan_flags = FFTW_MEASURE){

        std::array<int,NDims> converted;
        for(int i = 0;i < NDims;++i)
          converted[i] = _shape[i];


        PlanType value = fftw_plan_dft_r2c(NDims,
                                           converted.data(),
                                           _in,
                                           _out,
                                           plan_flags );
        return value;
      }

      template <size_t NDims>
      static PlanType create(const std::array<std::size_t,NDims>& _shape,
                             ComplexType* _in,
                             RealType* _out,
                             fftw_direction _dir = fftw_direction::forward,
                             unsigned plan_flags = FFTW_MEASURE){

        std::array<int,NDims> converted;
        for(int i = 0;i < NDims;++i)
          converted[i] = _shape[i];


        PlanType value = fftw_plan_dft_c2r(NDims,
                                           converted.data(),
                                           _in,
                                           _out,
                                           plan_flags );
        return value;
      }

      template <size_t NDims>
      static PlanType create(const std::array<std::size_t,NDims>& _shape,
                             ComplexType* _in,
                             ComplexType* _out,
                             fftw_direction _dir = fftw_direction::forward,
                             unsigned plan_flags = FFTW_MEASURE){

        std::array<int,NDims> converted;
        for(int i = 0;i < NDims;++i)
          converted[i] = _shape[i];


        PlanType value = fftw_plan_dft(NDims,
                                       converted.data(),
                                       _in,
                                       _out,
                                       static_cast<int>(_dir),
                                       plan_flags );
        return value;
      }

    };
  }  // namespace traits


  template<typename T_Precision>
  struct FftwWisdomLoader {
    std::string source;
    FftwWisdomLoader(const char* filename) {
      std::cout << "Loading wisdom: " << filename << std::endl;
      std::ifstream ifs;
      std::stringstream ss;
      std::string line;
      ifs.open(filename, std::ifstream::in);

      if(ifs.is_open()) {
        while ( getline (ifs,line) )
        {
          ss << line;
        }
        ifs.close();
        source = ss.str();
      }
    }

    int import() {
      if(std::is_same<T_Precision,float>::value)
        return fftwf_import_wisdom_from_string(source.c_str());
      else
        return fftw_import_wisdom_from_string(source.c_str());
    }

  }; // FftwWisdomLoader


  template<bool IfTrue = false>
  struct ImportWisdomIf {
    template <typename T_Precision, typename T_FftwConfig>
    void import() { }
  };

  template<>
  struct ImportWisdomIf<true> {

    template <typename T_Precision, typename T_FftwConfig>
    void import() {
      static
        FftwWisdomLoader<T_Precision> wisdom(
          std::is_same<T_Precision,float>::value
          ? T_FftwConfig::WisdomFileSinglePrecision
          : T_FftwConfig::WisdomFileDoublePrecision );
      if( !wisdom.import() )
        throw std::runtime_error("Wisdom file could not be loaded.");
    }
  };


  /**
   * FFTW implicit context init and reset wrapper. Time is benchmarked.
   */
  template<typename T_FftwConfig>
  struct FftwContext {

    static const std::string title() {
      return "Fftw";
    }

    static std::string get_device_list() {
      int av_procs = std::thread::hardware_concurrency();
      std::ostringstream msg;
      msg << av_procs << " CPU Threads supported.\n";
      return msg.str();
    }

    std::string get_used_device_properties() {
      static constexpr unsigned PlanFlags = T_FftwConfig::PlanFlags;
      static constexpr double PlanTimeLimit = T_FftwConfig::PlanTimeLimit;
      // Returns the number of supported concurrent threads
      size_t maxndevs = std::thread::hardware_concurrency();
      size_t ndevs = Options::getInstance().getNumberDevices();
      if( ndevs==0 || ndevs>maxndevs )
        ndevs = maxndevs;

      std::ostringstream msg;
      msg << "\"SupportedThreads\"," << maxndevs
          << ",\"UsedThreads\"," << ndevs
          << ",\"TotalMemory\"," << getMemorySize()
          << ",\"PlanFlags\",\"" <<
        (PlanFlags==FFTW_MEASURE ? "FFTW_MEASURE" : PlanFlags==FFTW_ESTIMATE ? "FFTW_ESTIMATE":"FFTW_WISDOM_ONLY");
      if(PlanTimeLimit > 0.0)
        msg << "\",\"FftwTimeLimit [s]\"," << PlanTimeLimit;
      else
        msg << "\",\"FftwTimeLimit [s]\"," << "\"None\"";
      return msg.str();
    }

    void create() {
    }

    void destroy() {
    }

  };


  /**
   * Fftw plan and execution class.
   *
   * This class handles:
   * - {1D, 2D, 3D} x {R2C, C2R, C2C} x {inplace, outplace} x {float, double}.
   */
  template<typename TFFT, // see fft_abstract.hpp (FFT_Inplace_Real, ...)
           typename TPrecision, // double, float
           size_t   NDim, // 1..3
           typename TFftwConfig
           >
  struct FftwImpl
  {

    //////////////////////////////////////////////////////////////////////////////////////
    // COMPILE TIME FIELDS

    using Extent = std::array<std::size_t,(int)NDim>;
    using Api  = typename traits::plan<TPrecision>;
    using ComplexType = typename traits::plan<TPrecision>::ComplexType;
    using RealType = typename traits::plan<TPrecision>::RealType;
    using PlanType = typename traits::plan<TPrecision>::PlanType;

    static_assert(NDim > 0 && NDim < 4, "[fftw.hpp]\treceived NDim not in [1,3], currently unsupported" );

    static constexpr
    bool IsInplace = TFFT::IsInplace;
    static constexpr
    bool IsComplex = TFFT::IsComplex;
    static constexpr
    bool IsInplaceReal = IsInplace && IsComplex==false;

    using value_type  = typename std::conditional<IsComplex,ComplexType,RealType>::type;

    //////////////////////////////////////////////////////////////////////////////////////

    /// extents of the FFT input data
    Extent extents_   = {{0}};
    /// extents of the FFT complex data (=FFT(input))
    Extent extents_complex_ = {{0}};
    /// product of corresponding extents
    size_t n_         = 0;
    /// product of corresponding extents
    size_t n_complex_ = 0;

    PlanType      fwd_plan_          = nullptr;
    PlanType      bwd_plan_          = nullptr;
    value_type*   data_              = nullptr;
    ComplexType*  data_complex_      = nullptr;
    /// size in bytes of FFT input data
    size_t        data_size_         = 0;
    /// size in bytes of FFT(input) for out-of-place transforms
    size_t        data_complex_size_ = 0;

    FftwImpl(const Extent& cextents) {
        extents_ = interpret_as::column_major(cextents);
        extents_complex_ = extents_;

        n_ = std::accumulate(extents_.begin(),
                             extents_.end(),
                             1,
                             std::multiplies<std::size_t>());

        if(IsComplex==false){
          extents_complex_.back() = (extents_.back()/2 + 1);
        }

        n_complex_ = std::accumulate(extents_complex_.begin(),
                                     extents_complex_.end(),
                                     1,
                                     std::multiplies<size_t>());

        data_size_ = (IsInplace ? 2*n_complex_ : n_) * sizeof(value_type);
        if(IsInplace==false)
          data_complex_size_ = n_complex_ * sizeof(ComplexType);

        //size_t total_mem = getMemorySize();
        size_t total_mem = 95*getMemorySize()/100; // keep some memory available, otherwise an out-of-memory killer becomes more likely
        if(total_mem < 3*data_size_+data_complex_size_) { // includes host input buffers
          std::stringstream ss;
          ss << total_mem << "<" << 3*data_size_+data_complex_size_ << " (bytes)";
          throw std::runtime_error("FFT data exceeds physical memory. "+ss.str());
        }

        traits::thread_api<TPrecision>::init_threads();
        traits::thread_api<TPrecision>::plan_with_threads(Options::getInstance().getNumberDevices());

        ImportWisdomIf<TFftwConfig::PlanFlags == FFTW_WISDOM_ONLY>()
          .template import<TPrecision, TFftwConfig>();
      }

    ~FftwImpl(){

      destroy();

      traits::thread_api<TPrecision>::cleanup_threads();
    }
      
    /**
     * Returns allocated memory for FFT
     */
    size_t get_allocation_size() {
      return data_size_ + data_complex_size_;
    }

    // create FFT plan handle
    void init_forward() {

      //Note: these calls clear the content of data_ et al
      fwd_plan_ = traits::plan<TPrecision>::create(extents_,
                                                   data_,
                                                   data_complex_,
                                                   traits::fftw_direction::forward,
                                                   TFftwConfig::PlanFlags);
      if(!fwd_plan_)
        if(TFftwConfig::PlanFlags == FFTW_WISDOM_ONLY)
          throw std::runtime_error("fftw forward plan could not be created as wisdom is not available for this problem.");
        else
          throw std::runtime_error("fftw forward plan could not be created.");
    }

    //
    void init_inverse() {
      bwd_plan_ = traits::plan<TPrecision>::create(extents_,
                                                   data_complex_,
                                                   data_,
                                                   traits::fftw_direction::inverse,
                                                   TFftwConfig::PlanFlags);//leave this here for now
      if(!bwd_plan_)
        if(TFftwConfig::PlanFlags == FFTW_WISDOM_ONLY)
          throw std::runtime_error("fftw inverse plan could not be created as wisdom is not available for this problem.");
        else
          throw std::runtime_error("fftw inverse plan could not be created.");

    }

    /**
     * Returns estimated allocated memory on device for FFT plan, accessing the plan size in fftw is impossible
     following @tdd11235813 advice:
     get_plan_size() should return 0, Sizes of transform buffers are already returned by get_allocation_size().
    */
    size_t get_plan_size() {
      
      return 0;
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

    //////////////////////////////////////////////////////////////////////////////////////
    // --- next methods are benchmarked ---

    void allocate() {
      data_ = static_cast<value_type*>(traits::memory_api<TPrecision>::malloc(data_size_));
      if(IsInplace){
        data_complex_ = reinterpret_cast<ComplexType*>(data_);
      }
      else{
        data_complex_ = static_cast<ComplexType*>(traits::memory_api<TPrecision>::malloc(data_complex_size_));
      }
    }


    void execute_forward() {

      traits::plan<TPrecision>::execute(fwd_plan_);
    }

    void execute_inverse() {

      traits::plan<TPrecision>::execute(bwd_plan_);
    }

    template<typename THostData>
    void upload(THostData* input) {

      if(!IsInplaceReal){
        memcpy(data_,input,data_size_);
      }
      else{
        const std::size_t max_z = (NDim >= 3 ? extents_[NDim-3] : 1);
        const std::size_t max_y = (NDim >= 2 ? extents_[NDim-2] : 1);
        const std::size_t max_x = extents_[NDim-1];
        const std::size_t allocated_x = extents_complex_[NDim-1];

        std::size_t input_index = 0;
        std::size_t data_index = 0;

        for(std::size_t z = 0;z < max_z;++z){
          for(std::size_t y = 0;y < max_y;++y){
            input_index = z*(max_y*max_x) + y*max_x;
            data_index = z*(max_y*allocated_x) + y*allocated_x;

            memcpy(data_+data_index,
                   input + input_index,
                   max_x*sizeof(value_type));

          }
        }
      }
    }

    template<typename THostData>
    void download(THostData* output) {

      if(!IsInplaceReal){
        memcpy(output,data_,data_size_);
      }
      else{
        const std::size_t max_z = (NDim >= 3 ? extents_[NDim-3] : 1);
        const std::size_t max_y = (NDim >= 2 ? extents_[NDim-2] : 1);
        const std::size_t max_x = extents_[NDim-1];
        const std::size_t allocated_x = extents_complex_[NDim-1];

        std::size_t output_index = 0;
        std::size_t data_index = 0;

        for(std::size_t z = 0;z < max_z;++z){
          for(std::size_t y = 0;y < max_y;++y){
            output_index = z*(max_y*max_x) + y*max_x;
            data_index = z*(max_y*allocated_x) + y*allocated_x;
            memcpy(output+output_index,
                   data_+data_index,
                   max_x*sizeof(value_type));
          }
        }
      }

    }

    void destroy() {

      if(data_)
        traits::memory_api<TPrecision>::free(data_);
      data_ = nullptr;

      if(data_complex_ && !IsInplace)
        traits::memory_api<TPrecision>::free(data_complex_);
      data_complex_ = nullptr;

      if(fwd_plan_)
        traits::plan<TPrecision>::destroy(fwd_plan_);
      fwd_plan_ = nullptr;

      if(bwd_plan_)
        traits::plan<TPrecision>::destroy(bwd_plan_);
      bwd_plan_ = nullptr;

    }
  };

  template<typename T_FftwConfig>
  using Inplace_Real = gearshifft::FFT<FFT_Inplace_Real,
                                       FFT_Plan_Not_Reusable,
                                       FftwImpl,
                                       TimerCPU,
                                       T_FftwConfig>;

  template<typename T_FftwConfig>
  using Outplace_Real = gearshifft::FFT<FFT_Outplace_Real,
                                        FFT_Plan_Not_Reusable,
                                        FftwImpl,
                                        TimerCPU,
                                        T_FftwConfig>;

  template<typename T_FftwConfig>
  using Inplace_Complex = gearshifft::FFT<FFT_Inplace_Complex,
                                          FFT_Plan_Not_Reusable,
                                          FftwImpl,
                                          TimerCPU,
                                          T_FftwConfig>;

  template<typename T_FftwConfig>
  using Outplace_Complex = gearshifft::FFT<FFT_Outplace_Complex,
                                           FFT_Plan_Not_Reusable,
                                           FftwImpl,
                                           TimerCPU,
                                           T_FftwConfig>;
} // namespace fftw
} // namespace gearshifft


#endif /* FFTW_HPP_ */
