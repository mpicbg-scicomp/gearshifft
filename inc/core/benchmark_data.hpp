#ifndef BENCHMARK_DATA_HPP_
#define BENCHMARK_DATA_HPP_

#include "types.hpp"

// http://www.boost.org/doc/libs/1_56_0/doc/html/align/tutorial.html

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <boost/align/aligned_allocator.hpp>
#include <boost/range/counting_range.hpp>
#include <boost/container/vector.hpp>
#include <boost/noncopyable.hpp>
#pragma GCC diagnostic pop

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace gearshifft {

  static constexpr size_t LIMIT16 = 1 << 15;

  template <typename RealType>
  struct BenchmarkDataGenerator {
    constexpr RealType operator()() {
      return 0.125 * (i_++ & 7);
    }

  private:
    size_t i_ = 0;
  };

  // To avoid overflows float16 data non-zero points are limited (y[0] of FFT(x)
  // is sum of input values). Overflow leads to nan or inf values and iFFT(FFT())
  // cannot be validated. This method still leads to nan's when size_ >= (1<<20).
  template <typename RealType>
  struct BenchmarkDataGeneratorLimited {
    BenchmarkDataGeneratorLimited(const size_t size) : size_(size) {}

    constexpr RealType operator()() {
      return (i_++ % (size_ / LIMIT16) == 0) ? 0.1 : 0.0;
    }

  private:
    size_t i_ = 0;
    const size_t size_;
  };

/**
 * Singleton test data helper and container.
 * Creates real and complex data on first access or when dimensions have changed.
 * Provides check_deviation() for accuracy comparison.
 * \note FFT works with accuracy like O(eps*log(N)).
 * \note max(data)-min(data) should fit into realtypes precision.
 */
  template<typename RealType, size_t Dimensions>
  class BenchmarkData : private boost::noncopyable {
  public:
    using ComplexType    = Real2D<RealType>;
    using BenchmarkDataT = BenchmarkData<RealType, Dimensions>;
    using Extent         = std::array<size_t, Dimensions>;
    using RealVector     = std::vector<RealType, boost::alignment::
                                       aligned_allocator<RealType,
                                                         alignof(RealType)> >;
    using ComplexVector  = std::vector<ComplexType, boost::alignment::
                                       aligned_allocator<ComplexType,
                                                         alignof(ComplexType)> >;


    static const BenchmarkDataT& data(const Extent& extents) {
      static BenchmarkDataT data;
      data.init_if_dim_changed(extents);
      return data;
    }

    size_t size() const {
      return size_;
    }

    void copyTo(RealVector& vec) const {
      vec.resize(size_);
      std::copy(data_linear_.begin(), data_linear_.end(), vec.begin());
    }

    void copyTo(ComplexVector& vec) const {
      vec.resize(size_);
      for( size_t i=0; i<size_; ++i ){
        vec[i].real(data_linear_[i]);
        vec[i].imag(0.0);
      }
    }

    // deviation = sample standard deviation
    template<bool Normalize, typename TVector>
    void check_deviation(double& deviation,
                         size_t& mismatches,
                         const TVector& data,
                         double error_bound) const {
      double diff_sum = 0;
      double diff;
      for( size_t i=0; i<size_; ++i ){
        diff = sub<Normalize>(data, i);
        if(std::isnan(diff) || diff > error_bound)
          ++mismatches;
        diff_sum += diff*diff;
      }
      deviation = sqrt(diff_sum/(size_-1.0));
    }

  private:

    template<bool Normalize>
    constexpr double sub(const ComplexVector& vector, size_t i) const {
      return Normalize ? 1.0/size_ * (vector[i].real()) - static_cast<double>(data_linear_[i]) : static_cast<double>( vector[i].real() - data_linear_[i] );
    }

    template<bool Normalize>
    constexpr double sub(const RealVector& vector, size_t i) const {
      return Normalize ? 1.0/size_ * (vector[i]) - static_cast<double>(data_linear_[i]) : static_cast<double>( vector[i] - data_linear_[i] );
    }

    void init_if_dim_changed(const Extent& extents) {
      if(extents_ == extents) // nothing changed
        return;
      extents_ = extents;
      size_ = std::accumulate(extents_.begin(), extents_.end(), 1, std::multiplies<size_t>());

      // allocate variables for all test cases
      data_linear_.resize(size_);

      if (std::is_same<RealType, float16>::value && size_ > LIMIT16) {
        std::generate(data_linear_.begin(), data_linear_.end(), BenchmarkDataGeneratorLimited<RealType>{size_});
      } else {
        std::generate(data_linear_.begin(), data_linear_.end(), BenchmarkDataGenerator<RealType>{});
      }

    }

    BenchmarkData() = default;
    ~BenchmarkData() = default;

  private:
    RealVector data_linear_;
    Extent extents_ = {{0}};
    size_t size_ = 0;

  };
} // gearshifft

#endif
