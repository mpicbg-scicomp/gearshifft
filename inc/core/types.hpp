#ifndef TYPES_HPP_
#define TYPES_HPP_

// https://github.com/mpicbg-scicomp/gearshifft/issues/145
#include <boost/predef.h>
#if BOOST_VERSION < 106600
#define BOOST_SYSTEM_NO_DEPRECATED
#endif

#include "complex-half.hpp"

#include <boost/mpl/list.hpp>

#include <array>
#include <vector>
#include <iostream>
#include <algorithm>
#include <complex>
#include <type_traits>

namespace gearshifft {
  using Extents1D = std::array<size_t,1>;
  using Extents2D = std::array<size_t,2>;
  using Extents3D = std::array<size_t,3>;
  using Extents1DVec = std::vector< Extents1D >;
  using Extents2DVec = std::vector< Extents2D >;
  using Extents3DVec = std::vector< Extents3D >;

/// List alias
  template<typename... Types>
  using List = boost::mpl::list<Types...>;

  namespace interpret_as {
    template<typename T_Extents>
    T_Extents column_major(T_Extents e) {
      return e;
    }
    template<typename T_Extents>
    T_Extents row_major(T_Extents e) {
      std::reverse(e.begin(), e.end());
      return e;
    }
  }

  template<typename T>
  struct ErrorBound {
    static constexpr int PrecisionDigits =
      std::conditional<std::is_same<float16, T>::value, std::integral_constant<int, 3>, std::integral_constant<int, std::numeric_limits<T>::digits10>>::type::value;

    double operator()() {
      return pow(10.0,-PrecisionDigits);
    }
  };


  template<size_t NDim>
  inline
  std::ostream& operator<<(std::ostream& os, const std::array<size_t,NDim>& e) {
    os << e[0];
    for(size_t k=1; k<NDim; ++k) {
      if(e[k]==0)
        break;
      os << "x" << e[k];
    }
    return os;
  }

  /**
   * Basic 2D vector with template type.
   * Used for test data for complex FFTs
   */
#if GEARSHIFFT_FLOAT16_SUPPORT == 0
  template<typename REAL>
  using Real2D = std::complex<REAL>;
#else
  template<typename REAL>
  using Real2D = typename std::conditional<std::is_same<REAL,float16>::value, gearshifft::complex<float16>, std::complex<REAL> >::type;
#endif

#if GEARSHIFFT_PRECISION_DOUBLE_ONLY==1
  using DefaultPrecisions                     = List<double>;
  using DefaultPrecisionsWithoutHalfPrecision = DefaultPrecisions;
#elif GEARSHIFFT_PRECISION_SINGLE_ONLY==1
  using DefaultPrecisions                     = List<float>;
  using DefaultPrecisionsWithoutHalfPrecision = DefaultPrecisions;
#else
# if GEARSHIFFT_FLOAT16_SUPPORT==1
#  if GEARSHIFFT_PRECISION_HALF_ONLY==1
  using DefaultPrecisions                     = List<float16>;
  using DefaultPrecisionsWithoutHalfPrecision = List<>;
#  else
  using DefaultPrecisions                     = List<float, double, float16>;
  using DefaultPrecisionsWithoutHalfPrecision = List<float, double>;
#  endif
# else
  using DefaultPrecisions                     = List<float, double>;
  using DefaultPrecisionsWithoutHalfPrecision = DefaultPrecisions;
# endif
#endif

  enum struct RecordType {
    Allocation = 0,
    PlanInitFwd,
    PlanInitInv,
    Upload,
    FFT,
    FFTInv,
    Download,
    PlanDestroy,
    Total,
    DevBufferSize,
    DevPlanSize,
    DevTransferSize,
    Deviation,
    Mismatches,
    _NrRecords
  };

  inline
  std::ostream& operator<< (std::ostream & os, RecordType r) {
    switch (r) {
    case RecordType::Allocation: return os << "Time_Allocation [ms]";
    case RecordType::PlanInitFwd: return os << "Time_PlanInitFwd [ms]";
    case RecordType::PlanInitInv: return os << "Time_PlanInitInv [ms]";
    case RecordType::Upload: return os << "Time_Upload [ms]";
    case RecordType::FFT: return os << "Time_FFT [ms]";
    case RecordType::FFTInv: return os << "Time_iFFT [ms]";
    case RecordType::Download: return os << "Time_Download [ms]";
    case RecordType::PlanDestroy: return os << "Time_PlanDestroy [ms]";
    case RecordType::Total: return os << "Time_Total [ms]";
    case RecordType::DevBufferSize: return os << "Size_DeviceBuffer [bytes]";
    case RecordType::DevPlanSize: return os << "Size_DevicePlan [bytes]";
    case RecordType::DevTransferSize: return os << "Size_DeviceTransfer [bytes]";
    case RecordType::Deviation: return os << "Error_StandardDeviation";
    case RecordType::Mismatches: return os << "Error_Mismatches";
    case RecordType::_NrRecords:
    default:
      ;
    }
    return os << static_cast<int>(r);
  }
}
#endif
