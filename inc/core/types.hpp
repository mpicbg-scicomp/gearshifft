#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <array>
#include <vector>
#include <iostream>
#include <algorithm>
#include <complex>

namespace gearshifft {
  using Extents1D = std::array<size_t,1>;
  using Extents2D = std::array<size_t,2>;
  using Extents3D = std::array<size_t,3>;
  using Extents1DVec = std::vector< Extents1D >;
  using Extents2DVec = std::vector< Extents2D >;
  using Extents3DVec = std::vector< Extents3D >;

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
  // template<typename REAL>
  // struct Real2D {
  //   using type = REAL;
  //   REAL x, y;
  // };

  template<typename REAL>
  using Real2D = std::complex<REAL>;

  enum struct RecordType {
    Device = 0,
    Allocation,
    PlanInit,
    Upload,
    FFT,
    FFTInverse,
    Download,
    PlanDestroy,
    Total,
    DevBufferSize,
    DevPlanSize,
    DevTransferSize,
    _NrRecords
  };

  inline
  std::ostream& operator<< (std::ostream & os, RecordType r) {
    switch (r) {
    case RecordType::Device: return os << "Time_Device [ms]";
    case RecordType::Allocation: return os << "Time_Allocation [ms]";
    case RecordType::PlanInit: return os << "Time_PlanInit [ms]";
    case RecordType::Upload: return os << "Time_Upload [ms]";
    case RecordType::FFT: return os << "Time_FFT [ms]";
    case RecordType::FFTInverse: return os << "Time_iFFT [ms]";
    case RecordType::Download: return os << "Time_Download [ms]";
    case RecordType::PlanDestroy: return os << "Time_PlanDestroy [ms]";
    case RecordType::Total: return os << "Time_Total [ms]";
    case RecordType::DevBufferSize: return os << "Size_DeviceBuffer [bytes]";
    case RecordType::DevPlanSize: return os << "Size_DevicePlan [bytes]";
    case RecordType::DevTransferSize: return os << "Size_DeviceTransfer [bytes]";
    };
    return os << static_cast<int>(r);
  }
}
#endif
