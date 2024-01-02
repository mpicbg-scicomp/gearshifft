#ifndef COMPLEX_HALF_H
#define COMPLEX_HALF_H

#if GEARSHIFFT_FLOAT16_SUPPORT == 0

struct float16 {};
inline float16 operator""_h(long double) { return float16{}; }

#else

#include "half.hpp"

#include <complex>

using float16 = half_float::half;
using namespace half_float::literal;

namespace gearshifft {

template<typename T> class complex;

  struct complex_float16 {
    float16 real;
    float16 imag;
  };

template<>
class complex<float16> {

public:
  typedef float16 value_type;
  typedef complex_float16 _ComplexT;


  constexpr complex(_ComplexT __z) : _M_value(__z) { }

  constexpr complex(float16 __r = float16(0), float16 __i = float16(0)) {
    _M_value.real = __r;
    _M_value.imag = __i;
  }

  constexpr float16
  real() const { return _M_value.real; }

  constexpr float16
  imag() const { return _M_value.imag; }


  template<typename T>
  void
  real(T __val) { _M_value.real = static_cast<float16>(__val); }

  template<typename T>
  void
  imag(T __val) { _M_value.imag = static_cast<float16>(__val); }

private:
  _ComplexT _M_value;
  };
}

#endif /* GEARSHIFFT_FLOAT16_SUPPORT */
#endif /* COMPLEX-HALF_H */
