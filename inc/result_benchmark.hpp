#ifndef RESULT_BENCHMARK_HPP_
#define RESULT_BENCHMARK_HPP_

#include <math.h>
#include <iostream>
#include <array>
#include <vector>
#include <assert.h>

namespace gearshifft
{
/** Result data generated after a benchmark has completed the runs
 */
  template<int T_NumberRuns, int T_NumberValues>
  class ResultBenchmark  {
  public:
    using ValuesT = std::array<std::array<double, T_NumberValues >, T_NumberRuns >;

    template<bool isComplex, bool isInplace, size_t T_NDim>
    void init(const std::array<unsigned, T_NDim>& ce) {
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
    }

    /**
     * @retval 1 arbitrary extents
     * @retval 2 power-of-two extents
     * @retval 3 power-of-other extents ( {3,5,7}^p )
     */
    size_t computeDimkind() {
      bool p2 = true;
      bool p3 = false;
      for(const auto e : extents_)
        p2 = p2 && PowerOf(e,2.0);
      if(!p2) {
        for(const auto e : extents_)
          p3 = p3 || PowerOf(e,3.0) || PowerOf(e,5.0) || PowerOf(e,7.0);
      }
      return p2 ? 2 : p3 ? 3 : 1;
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

    /* getters */


    template<typename T_Index>
    double getValue(T_Index idx_val) const {
      int idx = static_cast<int>(idx_val);
      assert(idx<T_NumberValues);
      return values_[run_][idx];
    }

    size_t getDim() const { return dim_; }
    size_t getDimKind() const { return dimkind_; }
    auto getExtents() const { return extents_; }
    size_t getExtentsTotal() const { return total_; }
    bool isInplace() const { return isInplace_; }
    bool isComplex() const { return isComplex_; }

  private:
    /// result object id
    //size_t id_ = 0;
    int run_ = 0;
    /// fft dimension
    size_t dim_ = 0;
    /// extents are 1=Arbitrary, 2=PowerOfTwo, 3=PowerOfOther
    size_t dimkind_ = 0;
    /// fft extents
    std::array<unsigned,3> extents_ = { {1} };
    /// all extents multiplied
    size_t total_ = 1;
    /// each run w values ( data[idx_run][idx_val] )
    ValuesT values_ = { {{ {0.0} }} };
    /// FFT Kind Inplace
    bool isInplace_ = false;
    /// FFT Kind Complex
    bool isComplex_ = false;

  private:
    bool PowerOf(unsigned e, double b) {
      if(!e)
        return false;
      double p = floor(log(e)/log(b)+0.5);
      return fabs(pow(b,p)-e)<0.000001;
    }

  };
}

#endif
