#ifndef RESULT_BENCHMARK_HPP_
#define RESULT_BENCHMARK_HPP_

#include <math.h>
#include <iostream>
#include <array>
#include <vector>
#include <assert.h>
#include <regex>

namespace gearshifft
{
/** Result data generated after a benchmark has completed the runs
 */
  template<int T_NumberRuns, int T_NumberValues>
  class ResultBenchmark  {
  public:
    using ValuesT = std::array<std::array<double, T_NumberValues >, T_NumberRuns >;

    template<bool isComplex, bool isInplace, size_t T_NDim>
    void init(const std::array<size_t, T_NDim>& ce, const char* precision) {
      total_ = 1;
      for(auto i=0; i<T_NDim; ++i) {
        extents_[i] = ce[i];
        total_ *= ce[i];
      }
      dim_ = T_NDim;
      dimkind_ = computeDimkind();
      run_ = 0;
      isInplace_ = isInplace;
      isComplex_ = isComplex;
      precision_.assign(precision);
      error_.clear();
      errorRun_ = -1;
    }

    /**
     * @retval 1 arbitrary extents
     * @retval 2 power-of-two extents
     * @retval 3 extents are combination of powers of (3,5,7) {3^r * 5^s * 7^t}
     */
    size_t computeDimkind() {
      bool p2=true;
      for( size_t k=0; k<dim_; ++k ) {
        p2 &= powerOf(extents_[k], 2.0);
      }
      if(p2)
        return 2;
      for( size_t k=0; k<dim_; ++k ) {
        if( isNotOnlyDivBy_2_3_5_7(extents_[k]) ){
          return 1;
        }
      }
      return 3;
    }
    /* setters */

    void setRun(int run) {
      run_ = run;
    }

    template<typename T_Index>
    void setValue(T_Index idx_val, double val) {
      int idx = static_cast<int>(idx_val);
      assert(idx<T_NumberValues);
      values_[run_][idx] = val;
    }

    void setError(int run, const std::string& what) {
      assert(errorRun_<T_NumberRuns);
      errorRun_ = run;
      error_ = what;
      // remove path informations of source file location
      std::regex e ("([^ ]*/|[^ ]*\\\\)([^/\\\\]*)(hpp|cpp)");
      error_ = std::regex_replace(error_,e,"$2$3");
    }

    /* getters */


    template<typename T_Index>
    double getValue(T_Index idx_val) const {
      int idx = static_cast<int>(idx_val);
      assert(idx<T_NumberValues);
      return values_[run_][idx];
    }
    std::string getPrecision() const { return precision_; }
    size_t getDim() const { return dim_; }
    size_t getDimKind() const { return dimkind_; }
    std::string getDimKindStr() const {
      switch(dimkind_) {
      case 1: return "oddshape";
      case 2: return "powerof2";
      case 3: return "radix357";
      }
    }
    std::array<size_t,3> getExtents() const { return extents_; }
    size_t getExtentsTotal() const { return total_; }
    bool isInplace() const { return isInplace_; }
    bool isComplex() const { return isComplex_; }
    bool hasError() const { return error_.empty()==false; }
    const std::string& getError() const { return error_; }
    int getErrorRun() const { return errorRun_; }

  private:
    /// result object id
    //size_t id_ = 0;
    int run_ = 0;
    /// fft dimension
    size_t dim_ = 0;
    /// extents are 1=Arbitrary, 2=PowerOfTwo, 3=CombRadix357
    size_t dimkind_ = 0;
    /// fft extents
    std::array<size_t,3> extents_ = { {1} };
    /// all extents multiplied
    size_t total_ = 1;
    /// each run w values ( data[idx_run][idx_val] )
    ValuesT values_ = { {{ {0.0} }} };
    /// FFT Kind Inplace
    bool isInplace_ = false;
    /// FFT Kind Complex
    bool isComplex_ = false;
    /// Precision as string
    std::string precision_;
    /// Error message
    std::string error_;
    /// Run where error occurred
    int errorRun_;

  private:
    bool powerOf(size_t e, double b) {
      if(e==0)
        return false;
      double a = static_cast<double>(e);
      double p = floor(log(a)/log(b)+0.5);
      return fabs(pow(b,p)-a)<0.0001;
    }

    bool isNotOnlyDivBy_2_3_5_7(size_t e) {
      size_t t = e;
      size_t sqr = static_cast<size_t>(sqrt(e));
      for( size_t d=2; d<=sqr; ++d ) {
        // if e contains divisors > 7
        if( e%d == 0 && d>7 )
          return true;
        // e = e / d^q
        while( e%d == 0 ) {
          e /= d;
        }
      }
      return e==t || e>7; // e might be a prime or a divisor>7
    }

  };
}

#endif
