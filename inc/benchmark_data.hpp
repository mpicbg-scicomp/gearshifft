#ifndef BENCHMARK_DATA_HPP_
#define BENCHMARK_DATA_HPP_

#include "types.hpp"

#include <stdlib.h>
#include <math.h>
#include <numeric>
#include <vector>

// http://www.boost.org/doc/libs/1_56_0/doc/html/align/tutorial.html
#include <boost/align/aligned_allocator.hpp>
#include <boost/range/counting_range.hpp>
#include <boost/container/vector.hpp>
#include <boost/noncopyable.hpp>

namespace gearshifft {

/**
 * Singleton test data helper and container.
 * Creates real and complex data on first access or when dimensions have changed.
 * Provides check_deviation() for accuracy comparison.
 * @note FFT works with accuracy like O(eps*log(N)).
 * @note max(data)-min(data) should fit into realtypes precision.
 */
  template<typename RealType, size_t Dimensions>
  class BenchmarkData : boost::noncopyable {
  public:
    using ComplexType    = Real2D<RealType>;
    using BenchmarkDataT = BenchmarkData<RealType, Dimensions>;
    using Extent         = std::array<unsigned, Dimensions>;
    using RealVector     = std::vector<RealType, boost::alignment::
                                       aligned_allocator<RealType,
                                                         alignof(RealType)> >;
    using ComplexVector  = std::vector<ComplexType, boost::alignment::
                                       aligned_allocator<ComplexType,
                                                         alignof(ComplexType)> >;

    size_t getSize() const { return size_; }
    const Extent getExtents() const { return extents_; }

    static BenchmarkDataT* getInstancePtr(const Extent& extents) {
      static BenchmarkDataT instance;
      instance.init_if_dim_changed(extents);
      return &instance;
    }

    void copyTo(RealVector& vec) {
      vec.reserve(size_);
      for( unsigned i : boost::counting_range(size_t(0), size_) ){
        vec[i] = data_linear_[i];
      }
    }

    void copyTo(ComplexVector& vec) {
      vec.reserve(size_);
      for( unsigned i : boost::counting_range(size_t(0), size_) ){
        vec[i].x = data_linear_[i];
        vec[i].y = 0;
      }
    }

    template<bool Normalize, typename TVector>
    double check_deviation(const TVector& data) const
      {
        double diff_sum = 0;
        double diff;
        for( unsigned i : boost::counting_range(size_t(0), size_) ){
          diff = sub<Normalize>(data,i);
          diff_sum += diff*diff;
        }
        return sqrt(diff_sum/(size_-1.0));
      }

  private:

    template<bool Normalize>
    constexpr double sub(const ComplexVector& vector, unsigned i) const {
      if(Normalize)
        return 1.0/size_ * (vector[i].x) - data_linear_[i];
      else
        return static_cast<double>( vector[i].x - data_linear_[i] );
    }

    template<bool Normalize>
    constexpr double sub(const RealVector& vector, unsigned i) const {
      if(Normalize)
        return 1.0/size_ * (vector[i]) - data_linear_[i];
      else
        return static_cast<double>( vector[i] - data_linear_[i] );
    }

    void init_if_dim_changed(const Extent& extents)
      {
        if(extents_ == extents) // nothing changed
          return;
        //std::cout << "Create inital data"<<std::endl;
        extents_ = extents;
        size_ = std::accumulate(extents_.begin(), extents_.end(), 1, std::multiplies<unsigned>());

        srand(2016); // seed for random number stream

        // allocate variables for all test cases
        data_linear_.resize(size_);
        for( unsigned i : boost::counting_range(size_t(0), size_) )
        {
          data_linear_[i] = 1.0*rand()/RAND_MAX;
        }
      }

    BenchmarkData() = default;
    ~BenchmarkData() = default;

  private:
    RealVector data_linear_;
    Extent extents_;
    size_t size_ = 0;

  };
} // gearshifft
#endif
