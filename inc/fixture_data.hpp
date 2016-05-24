/**
 * Creates data and generic functions for benchmarks.
 * Class is a singleton to control data allocations. 
 */
#ifndef FIXTURE_DATA_HPP_
#define FIXTURE_DATA_HPP_

#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <array>
#include <numeric>
#include <vector>
// http://www.boost.org/doc/libs/1_56_0/doc/html/align/tutorial.html
#include <boost/align/aligned_allocator.hpp>
#include <boost/range/counting_range.hpp>
#include <boost/container/vector.hpp>
#include <boost/noncopyable.hpp>

namespace gearshifft {
/**
 * Basic 2D vector with template type.
 * Used for test data for complex FFTs 
 */
  template<typename REAL>
  struct Real2D {
    using type = REAL;
    REAL x, y;
  };

/**
 * Singleton test data helper and container.
 * Creates real and complex data on first access or when dimensions have changed.
 * Provides check_deviation() for accuracy comparison.
 * @note FFT works with accuracy like O(eps*log(N)).
 * @note max(data)-min(data) should fit into realtypes precision.
 */
  template<typename RealType, size_t Dimensions>
  class FixtureData : boost::noncopyable
  {
  public:
    using ComplexType = Real2D<RealType>;
    using FixtureDataT = FixtureData<RealType, Dimensions>;
    using Extent = std::array<unsigned, Dimensions>;
    using RealVector = std::vector<RealType, boost::alignment::
                                   aligned_allocator<RealType, alignof(RealType)> >;
    using ComplexVector = std::vector<ComplexType, boost::alignment::
                                      aligned_allocator<ComplexType, alignof(ComplexType)> >;

  private:
    RealVector data_linear_;
    Extent extents_;
    size_t size_ = 0;

  public:
    size_t getSize() const { return size_; }
    const Extent getExtents() const { return extents_; }

    static FixtureDataT* getInstancePtr(const Extent& extents) {
      static FixtureDataT instance;
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

    FixtureData() {
      //std::cout << "FFTFixtureData created" << std::endl;// with " << __PRETTY_FUNCTION__ << std::endl;
    }

    ~FixtureData(){
      //std::cout << "FFTFixtureData destroyed." << std::endl;
    }
  };
} // gearshifft
#endif
