#ifndef FFTW_HPP_
#define FFTW_HPP_


#include "core/types.hpp"
#include "core/application.hpp"
#include "core/timer.hpp"
#include "core/fft.hpp"
#include "core/benchmark_suite.hpp"

#include <string.h>
#include <vector>
#include <array>
#include <thread>
#include <sstream>
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

	static void plan_with_threads(int nthreads = -1){

	  int av_procs = std::thread::hardware_concurrency();
	  
	  if(nthreads<1 || nthreads>av_procs)
	    nthreads = av_procs;

	  fftw_plan_with_nthreads(nthreads);
	    
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

	static void plan_with_threads(int nthreads = -1){

	  int av_procs = std::thread::hardware_concurrency();
	  
	  if(nthreads<1 || nthreads>av_procs)
	    nthreads = av_procs;

	  fftwf_plan_with_nthreads(nthreads);
	    
	}

	static void cleanup_threads(){
	  fftwf_cleanup_threads();
	}
	
      };

    
      enum fftw_direction {
	backward = FFTW_BACKWARD,
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
			     fftw_direction _dir = forward){

	  std::array<int,NDims> converted;
	  for(int i = 0;i < NDims;++i)
	    converted[i] = _shape[i];
	  
	  PlanType value = fftwf_plan_dft_r2c(NDims,
				    converted.data(),
				    _in,
				    _out,
				    FFTW_MEASURE );
	  return value;
	}

	template <size_t NDims>
	static PlanType create(const std::array<std::size_t,NDims>& _shape,
			     ComplexType* _in,
			     RealType* _out,
			     fftw_direction _dir = forward){

	  std::array<int,NDims> converted;
	  for(int i = 0;i < NDims;++i)
	    converted[i] = _shape[i];

	  PlanType value = fftwf_plan_dft_c2r(NDims,
				    converted.data(),
				    _in,
				    _out, FFTW_MEASURE );
	  return value;
	}
	
	template <size_t NDims>
	static PlanType create(const std::array<std::size_t,NDims>& _shape,
			     ComplexType* _in,
			     ComplexType* _out,
			     fftw_direction _dir = forward){

	  std::array<int,NDims> converted;
	  for(int i = 0;i < NDims;++i)
	    converted[i] = _shape[i];
	  

	  PlanType value = fftwf_plan_dft(NDims,
				converted.data(),
				_in,
				_out,
				_dir, FFTW_MEASURE );
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
			     fftw_direction _dir = forward){

	  std::array<int,NDims> converted;
	  for(int i = 0;i < NDims;++i)
	    converted[i] = _shape[i];
	  

	  PlanType value = fftw_plan_dft_r2c(NDims,
				    converted.data(),
				    _in,
				    _out,
				    FFTW_MEASURE );
	  return value;
	}

	template <size_t NDims>
	static PlanType create(const std::array<std::size_t,NDims>& _shape,
			     ComplexType* _in,
			     RealType* _out,
			     fftw_direction _dir = forward){

	  std::array<int,NDims> converted;
	  for(int i = 0;i < NDims;++i)
	    converted[i] = _shape[i];
	  

	  PlanType value = fftw_plan_dft_c2r(NDims,
				    converted.data(),
				    _in,
				    _out, FFTW_MEASURE );
	  return value;
	}
	
	template <size_t NDims>
	static PlanType create(const std::array<std::size_t,NDims>& _shape,
			     ComplexType* _in,
			     ComplexType* _out,
			     fftw_direction _dir = forward){

	  std::array<int,NDims> converted;
	  for(int i = 0;i < NDims;++i)
	    converted[i] = _shape[i];
	  

	  PlanType value = fftw_plan_dft(NDims,
				converted.data(),
				_in,
				_out,
				_dir, FFTW_MEASURE );
	  return value;
	}
	
      };
    }  // namespace traits

    /**
     * FFTW implicit context init and reset wrapper. Time is benchmarked.
     */
    struct Context {

      static const std::string title() {
	return "Fftw";
      }

      static std::string getListDevices() {
        return "CPU only";
      }

      std::string getDeviceInfos() {
	//obtain number of processors
      
	std::ostringstream msg;
	msg << std::thread::hardware_concurrency() << "xCPU";
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
	     size_t   NDim // 1..3
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
      
      // using FFTExecuteForward  = typename traits::plan::FFTExecuteForward;
      // using FFTExecuteBackward = typename traits::plan::FFTExecuteBackward;

      static constexpr
      bool IsInplace = TFFT::IsInplace;
      static constexpr
      bool IsComplex = TFFT::IsComplex;
      static constexpr
      bool Padding = IsInplace && IsComplex==false;

      using value_type  = typename std::conditional<IsComplex,ComplexType,RealType>::type;
    
      // static constexpr
      //  fftwType FFTForward  = IsComplex ? traits::plan::FFTComplex : traits::plan::FFTForward;
      // static constexpr
      //  fftwType FFTBackward = IsComplex ? traits::plan::FFTComplex : traits::plan::FFTBackward;

      //////////////////////////////////////////////////////////////////////////////////////
      // RUNTIME TIME FIELDS
    
      size_t n_;        // =[1]*..*[dim]
      size_t n_allocated_; // =[1]*..*[dim-1]*([dim]/2+1)
      Extent extents_;
      Extent allocated_extents_;

      PlanType			fwd_plan_           ;
      PlanType			bwd_plan_           ;
      value_type*		data_           ;
      ComplexType*       	data_transform_ ; // intermediate buffer
      size_t             	data_size_;
      size_t             	data_transform_size_;

      FftwImpl(const Extent& cextents)
	: n_(0),
	  n_allocated_(0),
	  extents_(),
	  allocated_extents_(),
	  fwd_plan_(nullptr),
	  bwd_plan_(nullptr),
	  data_(nullptr),
	  data_transform_(nullptr),
	  data_size_(0),
	  data_transform_size_(0)
      {
	extents_ = interpret_as::column_major(cextents);
	allocated_extents_ = extents_;
	
	n_ = std::accumulate(extents_.begin(),
			     extents_.end(),
			     1,
			     std::multiplies<std::size_t>());
	
	if(Padding){
	  allocated_extents_.back() = 2*(extents_.back()/2 + 1);
	}

	n_allocated_ = std::accumulate(allocated_extents_.begin(),
				     allocated_extents_.end(),
				     1,
				     std::multiplies<std::size_t>());

	data_size_ = sizeof(value_type)*n_allocated_;
	data_transform_size_ = IsInplace ? 0 : n_ * sizeof(ComplexType);

	traits::thread_api<TPrecision>::init_threads();
	traits::thread_api<TPrecision>::plan_with_threads(-1);

      }

      ~FftwImpl(){
	
	destroy();

	traits::thread_api<TPrecision>::cleanup_threads();
      }
      
      /**
       * Returns allocated memory for FFT
       */
      size_t getAllocSize() {
	return data_size_ + data_transform_size_;
      }

      // create FFT plan handle
      void init_forward() {

	//Note: these calls clear the content of data_ et al
	fwd_plan_ = traits::plan<TPrecision>::create(extents_, data_, data_transform_, traits::forward);
	bwd_plan_ = traits::plan<TPrecision>::create(extents_, data_transform_, data_, traits::backward);//leave this here for now

      }

      // recreates plan if needed
      void init_backward() {

	

      }

      /**
       * Returns estimated allocated memory on device for FFT plan, accessing the plan size in fftw is impossible
       following @tdd11235813 advice:
       getPlanSize() should return 0, Sizes of transform buffers are already returned by getAllocSize().
       */
      size_t getPlanSize() {
      
	return 0;
      }

      /**
       * Returns data to be transfered to and from device for FFT
       */
      size_t getTransferSize() {
	return data_size_ + data_transform_size_;
      }

      
      //////////////////////////////////////////////////////////////////////////////////////
      // --- next methods are benchmarked ---

      void malloc() {

	data_ = (value_type*)traits::memory_api<TPrecision>::malloc(data_size_);
	if(IsInplace){
	  data_transform_ = reinterpret_cast<ComplexType*>(data_);
	}
	else{
	  data_transform_ = (ComplexType*)traits::memory_api<TPrecision>::malloc(data_transform_size_);
	}


	
      }


      void execute_forward() {
	
	traits::plan<TPrecision>::execute(fwd_plan_);
	
      }

      void execute_backward() {

	traits::plan<TPrecision>::execute(bwd_plan_);
      }

      template<typename THostData>
      void upload(THostData* input) {

	
	if(!Padding){
	  memcpy(data_,input,data_size_);
	}
	else{
	  const std::size_t max_z = (NDim >= 3 ? extents_[NDim-3] : 1);
	  const std::size_t max_y = (NDim >= 2 ? extents_[NDim-2] : 1);
	  const std::size_t max_x = extents_[NDim-1];
	  const std::size_t allocated_x = allocated_extents_[NDim-1];
	  
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

	
	if(!Padding){
	  memcpy(output,data_,data_size_);
	}
	else{
	  const std::size_t max_z = (NDim >= 3 ? extents_[NDim-3] : 1);
	  const std::size_t max_y = (NDim >= 2 ? extents_[NDim-2] : 1);
	  const std::size_t max_x = extents_[NDim-1];
	  const std::size_t allocated_x = allocated_extents_[NDim-1];
	  
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
	
	if(data_transform_ && !IsInplace)
	  traits::memory_api<TPrecision>::free(data_transform_);
	data_transform_ = nullptr;
	
	if(fwd_plan_)
	  traits::plan<TPrecision>::destroy(fwd_plan_);
	fwd_plan_ = nullptr;

	if(bwd_plan_)
	  traits::plan<TPrecision>::destroy(bwd_plan_);
	bwd_plan_ = nullptr;

      }
    };

    typedef gearshifft::FFT<gearshifft::FFT_Inplace_Real, FftwImpl, TimerCPU> Inplace_Real;
    typedef gearshifft::FFT<gearshifft::FFT_Outplace_Real, FftwImpl, TimerCPU> Outplace_Real;
    typedef gearshifft::FFT<gearshifft::FFT_Inplace_Complex, FftwImpl, TimerCPU> Inplace_Complex;
    typedef gearshifft::FFT<gearshifft::FFT_Outplace_Complex, FftwImpl, TimerCPU> Outplace_Complex;

  } // namespace Fftw
} // namespace gearshifft


#endif /* FFTW_HPP_ */
