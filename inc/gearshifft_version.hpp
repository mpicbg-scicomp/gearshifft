#ifndef GEARSHIFFT_VERSION_HPP
#define GEARSHIFFT_VERSION_HPP

#include <string>

namespace gearshifft {
  std::string gearshifft_version();
  unsigned gearshifft_version_major();
  unsigned gearshifft_version_minor();
  unsigned gearshifft_version_patch();
  unsigned gearshifft_version_tweak();
}

#endif /* GEARSHIFFT_VERSION_HPP */
