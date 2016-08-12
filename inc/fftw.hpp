#ifndef FFTW_HPP_
#define FFTW_HPP_

#include "types.hpp"
#include "application.hpp"
#include "timer.hpp"
#include "fft.hpp"
#include "benchmark_suite.hpp"
//#include "fftw_helper.hpp"

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
	  fftwf_destroy_plan(_plan);
	}

	template <size_t NDims>
	static PlanType create(const std::array<std::size_t,NDims>& _shape,
			     RealType* _in,
			     ComplexType* _out,
			     fftw_direction _dir = forward){
	  return fftwf_plan_dft_r2c(NDims,
				    (int*)_shape.data(),
				    _in,
				    _out,
				    FFTW_MEASURE );
	}

	template <size_t NDims>
	static PlanType create(const std::array<std::size_t,NDims>& _shape,
			     ComplexType* _in,
			     RealType* _out,
			     fftw_direction _dir = forward){
	  return fftwf_plan_dft_c2r(NDims,
				    (int*)_shape.data(),
				    _in,
				    _out, FFTW_MEASURE );
	}
	
	template <size_t NDims>
	static PlanType create(const std::array<std::size_t,NDims>& _shape,
			     ComplexType* _in,
			     ComplexType* _out,
			     fftw_direction _dir = forward){
	  return fftwf_plan_dft(NDims,
				(int*)_shape.data(),
				_in,
				_out,
				_dir, FFTW_MEASURE );
	}
	// //1D
	// static PlanType plan(const std::array<std::size_t,1>& _shape,
	// 		     RealType* _in,
	// 		     ComplexType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftwf_plan_dft_r2c(1,
	// 			    _shape[0],
	// 			    _in,
	// 			    _out, FFTW_MEASURE );
	// }
      
	// static PlanType plan(const std::array<std::size_t,1>& _shape,
	// 		     ComplexType* _in,
	// 		     RealType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftwf_plan_dft_c2r(1,
	// 			    _shape[0],
	// 			    _in,
	// 			    _out, FFTW_MEASURE );
	// }      

	// static PlanType plan(const std::array<std::size_t,1>& _shape,
	// 		     ComplexType* _in,
	// 		     ComplexType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftwf_plan_dft(1,
	// 			_shape[0],
	// 			_in,
	// 			_out,
	// 			_dir, FFTW_MEASURE );
	// }      

	// //2D
	// static PlanType plan(const std::array<std::size_t,2>& _shape,
	// 		     RealType* _in,
	// 		     ComplexType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftwf_plan_dft_r2c(2,
	// 			    _shape[1],_shape[0],
	// 			    _in,
	// 			    _out, FFTW_MEASURE );
	// }
      
	// static PlanType plan(const std::array<std::size_t,2>& _shape,
	// 		     ComplexType* _in,
	// 		     RealType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftwf_plan_dft_c2r(2,
	// 			    _shape[1],_shape[0],
	// 			    _in,
	// 			    _out, FFTW_MEASURE );
	// }      

	// static PlanType plan(const std::array<std::size_t,2>& _shape,
	// 		     ComplexType* _in,
	// 		     ComplexType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftwf_plan_dft(2,
	// 			_shape[1],_shape[0],
	// 			_in,
	// 			_out,
	// 			_dir, FFTW_MEASURE );
	// }

	// //3D      
	// static PlanType plan(const std::array<std::size_t,3>& _shape,
	// 		     RealType* _in,
	// 		     ComplexType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftwf_plan_dft_r2c(3,
	// 			    _shape[2],_shape[1],_shape[0],
	// 			    _in,
	// 			    _out, FFTW_MEASURE );
	// }
      
	// static PlanType plan(const std::array<std::size_t,3>& _shape,
	// 		     ComplexType* _in,
	// 		     RealType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftwf_plan_dft_c2r(3,
	// 			    _shape[2],_shape[1],_shape[0],
	// 			    _in,
	// 			    _out, FFTW_MEASURE );
	// }      

	// static PlanType plan(const std::array<std::size_t,3>& _shape,
	// 		     ComplexType* _in,
	// 		     ComplexType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftwf_plan_dft(3,
	// 			_shape[2],_shape[1],_shape[0],
	// 			_in,
	// 			_out,
	// 			_dir, FFTW_MEASURE );
	// }
	// //ND
	
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
	  fftw_destroy_plan(_plan);
	}

	// //ND
	template <size_t NDims>
	static PlanType create(const std::array<std::size_t,NDims>& _shape,
			     RealType* _in,
			     ComplexType* _out,
			     fftw_direction _dir = forward){
	  return fftw_plan_dft_r2c(NDims,
				    (int*)_shape.data(),
				    _in,
				    _out,
				    FFTW_MEASURE );
	}

	template <size_t NDims>
	static PlanType create(const std::array<std::size_t,NDims>& _shape,
			     ComplexType* _in,
			     RealType* _out,
			     fftw_direction _dir = forward){
	  return fftw_plan_dft_c2r(NDims,
				    (int*)_shape.data(),
				    _in,
				    _out, FFTW_MEASURE );
	}
	
	template <size_t NDims>
	static PlanType create(const std::array<std::size_t,NDims>& _shape,
			     ComplexType* _in,
			     ComplexType* _out,
			     fftw_direction _dir = forward){
	  return fftw_plan_dft(NDims,
				(int*)_shape.data(),
				_in,
				_out,
				_dir, FFTW_MEASURE );
	}
	
	//1D
	// static PlanType create(const std::array<std::size_t,1>& _shape,
	// 		     RealType* _in,
	// 		     ComplexType* _out ,
	// 		     fftw_direction _dir = forward){
	//   return fftw_plan_dft_r2c(1,
	// 			   _shape.data(),
	// 			   _in,
	// 			   _out,
	// 			   FFTW_MEASURE );
	// }
      
	// static PlanType create(const std::array<std::size_t,1>& _shape,
	// 		     ComplexType* _in,
	// 		     RealType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftw_plan_dft_c2r(1,
	// 			   _shape[0],
	// 			   _in,
	// 			   _out, FFTW_MEASURE );
	// }      

	// static PlanType create(const std::array<std::size_t,1>& _shape,
	// 		     ComplexType* _in,
	// 		     ComplexType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftw_plan_dft(1,
	// 		       _shape[0],
	// 		       _in,
	// 		       _out,
	// 		       _dir, FFTW_MEASURE );
	// }      

	// //2D
	// static PlanType create(const std::array<std::size_t,2>& _shape,
	// 		     RealType* _in,
	// 		     ComplexType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftw_plan_dft_r2c(2,
	// 			   _shape[1],_shape[0],
	// 			   _in,
	// 			   _out, FFTW_MEASURE );
	// }
      
	// static PlanType create(const std::array<std::size_t,2>& _shape,
	// 		     ComplexType* _in,
	// 		     RealType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftw_plan_dft_c2r(2,
	// 			   _shape[1],_shape[0],
	// 			   _in,
	// 			   _out, FFTW_MEASURE );
	// }      

	// static PlanType create(const std::array<std::size_t,2>& _shape,
	// 		     ComplexType* _in,
	// 		     ComplexType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftw_plan_dft(2,
	// 		       _shape[1],_shape[0],
	// 		       _in,
	// 		       _out,
	// 		       _dir, FFTW_MEASURE );
	// }

	// //3D      
	// static PlanType create(const std::array<std::size_t,3>& _shape,
	// 		     RealType* _in,
	// 		     ComplexType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftw_plan_dft_r2c(3,
	// 			   _shape[2],_shape[1],_shape[0],
	// 			   _in,
	// 			   _out, FFTW_MEASURE );
	// }
      
	// static PlanType create(const std::array<std::size_t,3>& _shape,
	// 		     ComplexType* _in,
	// 		     RealType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftw_plan_dft_c2r(3,
	// 			   _shape[2],_shape[1],_shape[0],
	// 			   _in,
	// 			   _out, FFTW_MEASURE );
	// }      

	// static PlanType create(const std::array<std::size_t,3>& _shape,
	// 		     ComplexType* _in,
	// 		     ComplexType* _out,
	// 		     fftw_direction _dir = forward){
	//   return fftw_plan_dft(3,
	// 		       _shape[2],_shape[1],_shape[0],
	// 		       _in,
	// 		       _out,
	// 		       _dir, FFTW_MEASURE );
	// }
      };
    }  // namespace traits

    /**
     * FFTW implicit context init and reset wrapper. Time is benchmarked.
     */
    struct Context {

      static const std::string title() {
	return "Fftw";
      }

      std::string getDeviceInfos() {
	//obtain number of processors
      
	std::ostringstream msg;
	msg << std::thread::hardware_concurrency() << "xCPU";
	return msg.str();
      }

      void create() {
	//TODO: set number of threads
      
      }

      void destroy() {
      
	//not needed
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
    
      // using FFTExecuteForward  = typename traits::plan::FFTExecuteForward;
      // using FFTExecuteBackward = typename traits::plan::FFTExecuteBackward;

      static constexpr
      bool IsInplace = TFFT::IsInplace;
      static constexpr
      bool IsComplex = TFFT::IsComplex;
      static constexpr
      bool Padding = IsInplace && IsComplex==false && NDim>1;

      using value_type  = typename std::conditional<IsComplex,ComplexType,RealType>::type;
    
      // static constexpr
      //  fftwType FFTForward  = IsComplex ? traits::plan::FFTComplex : traits::plan::FFTForward;
      // static constexpr
      //  fftwType FFTBackward = IsComplex ? traits::plan::FFTComplex : traits::plan::FFTBackward;

      //////////////////////////////////////////////////////////////////////////////////////
      // RUNTIME TIME FIELDS
    
      size_t n_;        // =[1]*..*[dim]
      size_t n_padded_; // =[1]*..*[dim-1]*([dim]/2+1)
      Extent extents_;

      PlanType			plan_           ;
      value_type*		data_           = nullptr;
      ComplexType*       	data_transform_ = nullptr; // intermediate buffer
      size_t             	data_size_;
      size_t             	data_transform_size_;

      FftwImpl(const Extent& cextents)
	: n_(0),
	  n_padded_(0),
	  extents_(),
	  plan_(),
	  data_(nullptr),
	  data_transform_(nullptr),
	  data_size_(0),
	  data_transform_size_(0)
      {
	extents_ = interpret_as::column_major(cextents);
	
	n_ = std::accumulate(extents_.begin(),
			     extents_.end(),
			     1,
			     std::multiplies<unsigned>());
	if(Padding)
	  n_padded_ = n_ / extents_[NDim-1] * (extents_[NDim-1]/2 + 1);

	data_size_ = ( Padding ? 2*n_padded_ : n_ ) * sizeof(value_type);
	data_transform_size_ = IsInplace ? 0 : n_ * sizeof(ComplexType);

	traits::thread_api<TPrecision>::init_threads();
	traits::thread_api<TPrecision>::plan_with_threads(-1);
      }

      ~FftwImpl(){
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

	//NOTE: contents of data_ et al are overwritten by this call
	plan_ = traits::plan<TPrecision>::create(extents_, data_, data_transform_, traits::forward);
      }

      // recreates plan if needed
      void init_backward() {

	traits::plan<TPrecision>::destroy(plan_);

	//NOTE: contents of data_ et al are overwritten by this call
	plan_ = traits::plan<TPrecision>::create(extents_, data_, data_transform_, traits::backward);
      
      }

      /**
       * Returns estimated allocated memory on device for FFT plan, accessing the plan size in fftw is impossible
       
       */
      size_t getPlanSize() {
      
	if(IsInplace)
	  return data_size_;
	else
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
	traits::plan<TPrecision>::execute(plan_);
      }

      void execute_backward() {
	traits::plan<TPrecision>::execute(plan_);
      }

      template<typename THostData>
      void upload(THostData* input) {

	//TODO: unnecessary?
      
	// if(Padding) // real + inplace + ndim>1
	// {
	//   size_t w      = extents_[NDim-1] * sizeof(THostData);
	//   size_t h      = n_ * sizeof(THostData) / w;
	//   size_t pitch  = (extents_[NDim-1]/2+1) * sizeof(ComplexType);
	//   CHECK_CUDA(cudaMemcpy2D(data_, pitch, input, w, w, h, cudaMemcpyHostToDevice);
	// }else{
	//   CHECK_CUDA(cudaMemcpy(data_, input, data_size_, cudaMemcpyHostToDevice);
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
	//   CHECK_CUDA(cudaMemcpy2D(output, w, data_, pitch, w, h, cudaMemcpyDeviceToHost);
	// }else{
	//   CHECK_CUDA(cudaMemcpy(output, data_, data_size_, cudaMemcpyDeviceToHost);
	// }
      }

      void destroy() {

	traits::memory_api<TPrecision>::free(data_);
	if(IsInplace==false)
	  traits::memory_api<TPrecision>::free(data_transform_);

	traits::plan<TPrecision>::destroy(plan_);
      
      }
    };

    typedef gearshifft::FFT<gearshifft::FFT_Inplace_Real, FftwImpl, TimerCPU> Inplace_Real;
    typedef gearshifft::FFT<gearshifft::FFT_Outplace_Real, FftwImpl, TimerCPU> Outplace_Real;
    typedef gearshifft::FFT<gearshifft::FFT_Inplace_Complex, FftwImpl, TimerCPU> Inplace_Complex;
    typedef gearshifft::FFT<gearshifft::FFT_Outplace_Complex, FftwImpl, TimerCPU> Outplace_Complex;

  } // namespace Fftw
} // namespace gearshifft


#endif /* FFTW_HPP_ */
