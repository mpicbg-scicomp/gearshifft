#ifndef RESULT_ALL_HPP_
#define RESULT_ALL_HPP_

#include "result_benchmark.hpp"
#include "result_writer.hpp"
#include "traits.hpp"
#include "types.hpp"

#include <algorithm>
#include <mutex>
#include <vector>


namespace gearshifft {

  template<int T_NumberRuns,  // runs per benchmark including warmup
           int T_NumberWarmups,
           int T_NumberValues // recorded values per run
           >
  class ResultAll {

    friend class ResultWriter<T_NumberRuns, T_NumberWarmups, T_NumberValues>;

    using ResultBenchmarkT = ResultBenchmark<T_NumberRuns, T_NumberValues>;

  public:

    void add(const ResultBenchmarkT& result) {
      std::lock_guard<std::mutex> g(resultsMutex_);
      results_.push_back(result);
    }

    std::size_t size() const {
      return results_.size();
    }

    /*
     * sort order:  fftkind -> dimkind -> dim -> nx*ny*nz
     */
    void sort() {
      std::lock_guard<std::mutex> g(resultsMutex_);
      std::stable_sort(
        results_.begin( ), results_.end( ),
        [ ]( const ResultBenchmarkT& lhs, const ResultBenchmarkT& rhs )
        {
          if(lhs.getPrecision()==rhs.getPrecision())
            if(lhs.isInplace()==rhs.isInplace())
              if(lhs.isComplex()==rhs.isComplex())
                return lhs.getDimKind()<rhs.getDimKind() ||
                                        (lhs.getDimKind()==rhs.getDimKind() &&
                                         (
                                          lhs.getDim()<rhs.getDim() ||
                                          (lhs.getDim()==rhs.getDim() &&
                                           lhs.getExtentsTotal()<rhs.getExtentsTotal())
                                         ));
              else
                return lhs.isComplex();
            else
              return lhs.isInplace();
          else
            return false;
        });
    }

  private:
    std::mutex resultsMutex_;
    std::vector< ResultBenchmarkT > results_;

  };

}
#endif
