#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <array>
#include <vector>
#include <iostream>

namespace gearshifft {
  using Extents1D = std::array<unsigned,1>;
  using Extents2D = std::array<unsigned,2>;
  using Extents3D = std::array<unsigned,3>;
  using Extents1DVec = std::vector< Extents1D >;
  using Extents2DVec = std::vector< Extents2D >;
  using Extents3DVec = std::vector< Extents3D >;

  template<size_t NDim>
  inline
  std::ostream& operator<<(std::ostream& os, const std::array<unsigned,NDim>& e) {
    os << e[0];
    for(size_t k=1; k<NDim; ++k)
      os << "x" << e[k];
  }

  enum struct RecordType{
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
    _NrRecords
  };

  inline
  std::ostream& operator<< (std::ostream & os, RecordType r)
  {
    switch (r)
    {
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
    };
    return os << static_cast<int>(r);
  }
}
#endif
